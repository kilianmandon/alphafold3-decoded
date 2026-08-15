# Training Command:
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc-per-node gpu -m training.torchrun_training

import contextlib
import datetime
import os
from pathlib import Path
import pickle
import time
from common import utils
from torch_snapkit import memory_snapshot
import tqdm

import torch
from torch.profiler import ProfilerActivity, profile
import wandb
from atomworks.io.utils.io_utils import to_cif_file

from config import Config
from diffusion.model import Model
from feature_extraction.feature_extraction import collate_batch, tree_map
from training.af3_dataset import af3_pipeline_none_on_error, build_af3_dataset, build_eval_dataset, build_sampler, collate_batch_drop_none
from training.training_module import AF3TrainingModule, mse_loss
import torch.distributed as dist
import torch._dynamo

torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 256

def setup_ddp():
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)

    return rank, local_rank, world_size, device


def evaluation(training_model, eval_loader, device, step, rank, cif_dir):
    item_counter = 0
    total_loss = torch.tensor(0, device=device, dtype=torch.float32)
    for batch_with_labels in eval_loader:
        batch_with_labels = tree_map(lambda x: x.to(device), batch_with_labels, skip_unconvertible_entries=True)
        batch = batch_with_labels['batch']
        batch.reference_features.setup_block_mask(num_diffusion_samples=1)
        batch.token_features.setup_block_mask()
        batch.reference_features.materialize()

        x_flat = training_model.model(batch)

        x_gt = [torch.tensor(atom_array.coord, device=device) for atom_array in batch_with_labels["atom_array"]]
        x_gt = utils.pad_to_shape(collate_batch(x_gt), batch.reference_features.positions.shape)
        ca_mask = [torch.tensor([name=='CA' for name in atom_array.atom_name], device=device) for atom_array in batch_with_labels["atom_array"]]
        ca_mask = utils.pad_to_shape(collate_batch(ca_mask), batch.reference_features.mask.shape)
        x_gt_mask = ~(x_gt.isnan().any(dim=-1))
        x_gt_mask = x_gt_mask & ca_mask
        x_gt[~x_gt_mask] = 0

        for i, atom_array in enumerate(batch_with_labels['atom_array']):
            atom_mask = batch.reference_features.mask[i].cpu().numpy()
            atom_array.coord = x_flat[i][atom_mask].cpu().numpy()
            filename = f'{cif_dir}/step_{step:04d}_rank_{rank}_idx_{item_counter}.cif'
            try:
                to_cif_file(atom_array, filename)  
            except:
                print(f'[rank{rank}]: Error generating cif {item_counter}.')
            item_counter += 1

        loss = mse_loss(x_flat, x_gt, x_gt_mask, batch)
        total_loss += loss.sum()
    total_loss = total_loss / item_counter

    dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)

    return total_loss




def main():
    config = Config()
    config.global_config.n_cycle = 1
    model = Model(config)


    rank, local_rank, world_size, device = setup_ddp()
    # rank = 0; world_size=1; device='cuda:0'

    # train_ds = build_af3_dataset(config)
    train_ds_pickle_path = Path('train_ds.pkl')
    eval_ds_pickle_path = Path('eval_ds.pkl')
    if not train_ds_pickle_path.exists() and rank==0:
        print('No training dataset pickle found, building...')
        train_ds = build_eval_dataset(config, samples_per_group=8, is_inference=False)
        for ds in train_ds.datasets:
            ds.transform = None
        with open(train_ds_pickle_path, 'wb') as f:
            pickle.dump(train_ds, f)
    if not eval_ds_pickle_path.exists() and rank==0:
        print('No eval dataset pickle found, building...')
        eval_ds = build_eval_dataset(config, samples_per_group=8)
        for ds in eval_ds.datasets:
            ds.transform = None
        with open(eval_ds_pickle_path, 'wb') as f:
            pickle.dump(eval_ds, f)


    dist.barrier()

    with open(train_ds_pickle_path, 'rb') as f:
        train_ds = pickle.load(f)
        for ds in train_ds.datasets:
            ds.transform = af3_pipeline_none_on_error(config, is_inference=False)

    with open(eval_ds_pickle_path, 'rb') as f:
        eval_ds = pickle.load(f)
        for ds in eval_ds.datasets:
            ds.transform = af3_pipeline_none_on_error(config, is_inference=True)

    eval_token_counts = []
    for batch in torch.utils.data.DataLoader(eval_ds, num_workers=10, collate_fn=lambda x: collate_batch_drop_none(x, config)):
        eval_token_counts.append(batch['batch'].token_features.token_count)
    if rank==0:
        print(f'Eval token counts: {eval_token_counts}')

    sampler = build_sampler(train_ds)
    avail_workers = len(os.sched_getaffinity(0))
    num_workers = min(max(1, (avail_workers-2)//world_size - 1), 6)
    print(f'[rank{rank}]: {num_workers}/{avail_workers} workers used')
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers=num_workers, batch_size=config.training_config.micro_batch_size, sampler=sampler, collate_fn=lambda x: collate_batch_drop_none(x, config))

    eval_ds = torch.utils.data.Subset(eval_ds, indices=range(rank, len(eval_ds), world_size))
    eval_dl = torch.utils.data.DataLoader(eval_ds, num_workers=num_workers, batch_size=config.training_config.micro_batch_size, collate_fn=lambda x: collate_batch_drop_none(x, config))
    base_eval_cif_dir = 'eval_cif_samples'

    Path(base_eval_cif_dir).mkdir(exist_ok=True)

    model.to(device)
    model.regional_compile()
    if rank==0:
        print(f'Total world size: {world_size}')

    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)

    h = sum(p.data.double().sum() for p in model.parameters())
    print(f'rank{rank}', h.item())

    training_model = AF3TrainingModule(model, config, num_devices=world_size)
    
    opt = training_model.configure_optimizers()


    n_steps = 500
    n_steps_eval = 50

    if rank==0:
        wandb.init(
            project='af3',
            config = {
                'micro_bs': config.training_config.micro_batch_size,
                'diff_micro_bs': config.training_config.diffusion_micro_batch_size,
                'batch_size': config.training_config.batch_size,
                'world_size': world_size,
            }
        )


    t0 = time.perf_counter()

    def loop_train_dl():
        while True:
            yield from train_dl

    pbar = tqdm.tqdm(loop_train_dl(), total=n_steps)

    for batch_idx, batch in enumerate(pbar):
        t_data = time.perf_counter() - t0

        torch.cuda.reset_peak_memory_stats()

        batch = tree_map(lambda x: x.to(device), batch, skip_unconvertible_entries=True)
        loss = training_model.training_step(batch, batch_idx)

        if world_size>1:
            dist.all_reduce(loss, dist.ReduceOp.AVG)

        if rank==0:
            pbar.set_postfix(train_loss=loss.item())

        if (batch_idx+1) % training_model.global_grad_accum_steps == 0:
            grad_norm = training_model.sync_grads()
            opt.step()
            opt.zero_grad()

        if rank==0:
            log = {
                'train/loss': loss.item(),
                'time/step': time.perf_counter() - t0,
                'time/data_wait': t_data,
                'mem/peak_alloc_gb': torch.cuda.max_memory_allocated()/1024**3,
                'mem/peak_reserved_gb': torch.cuda.max_memory_reserved()/1024**3,
            }

            if (batch_idx+1) % training_model.global_grad_accum_steps==0:
                log['train/grad_norm'] = grad_norm
            wandb.log(log, step=batch_idx)

        if (batch_idx+1)%n_steps_eval==0:
            step_eval_cif_dir = f'{base_eval_cif_dir}/step_{batch_idx:04d}'
            Path(step_eval_cif_dir).mkdir(exist_ok=True)
            training_model.model.eval()
            with torch.no_grad():
                total_loss = evaluation(training_model, eval_dl, device, batch_idx, rank, step_eval_cif_dir)
            training_model.model.train()
            if rank == 0:
                wandb.log({'eval/loss': total_loss.item()}, step=batch_idx)
                cif_artifact = wandb.Artifact(name=f'eval_predictions-{wandb.run.id}', type='cif', metadata={'step': batch_idx})
                cif_artifact.add_dir(step_eval_cif_dir)
                wandb.run.log_artifact(cif_artifact)


        t0 = time.perf_counter()
        
        if batch_idx+1>=n_steps:
            break





if __name__=='__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()