import contextlib
import datetime
import os
import pickle
import time
from torch_snapkit import memory_snapshot
import tqdm

import torch
from torch.profiler import ProfilerActivity, profile
import wandb

from config import Config
from diffusion.model import Model
from feature_extraction.feature_extraction import tree_map
from training.af3_dataset import build_af3_dataset, build_sampler, collate_batch_drop_none
from training.training_module import AF3TrainingModule
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



def main():
    config = Config()
    config.global_config.n_cycle = 1
    config.diffusion_config.denoising_steps = 1
    model = Model(config)


    rank, local_rank, world_size, device = setup_ddp()
    # rank = 0; world_size=1; device='cuda:0'

    train_ds = build_af3_dataset(config)
    sampler = build_sampler(train_ds)
    avail_workers = len(os.sched_getaffinity(0))
    num_workers = min(max(1, (avail_workers-2)//world_size - 1), 6)
    print(f'[rank{rank}]: {num_workers}/{avail_workers} workers used')
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers=num_workers, batch_size=config.training_config.micro_batch_size, sampler=sampler, collate_fn=lambda x: collate_batch_drop_none(x, config))

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


    n_steps = 50
    pbar = tqdm.tqdm(train_dl, total=n_steps)

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
    for batch_idx, batch in enumerate(pbar):
        t_data = time.perf_counter() - t0
        batch = tree_map(lambda x: x.to(device), batch, skip_unconvertible_entries=True)
        loss = training_model.training_step(batch, batch_idx)

        # print(f'[rank{rank}] ', torch.cuda.memory_allocated() / 1024**3, "GB allocated")
        # print(f'[rank{rank}] ', torch.cuda.memory.max_memory_allocated() / 1024**3, "GB peak allocated")
        # print(f'[rank{rank}] ', torch.cuda.memory_reserved() / 1024**3, "GB reserved")
        # print(f'[rank{rank}] ', torch.cuda.max_memory_reserved() / 1024**3, "GB peak reserved")


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
            wandb.log(log)


        t0 = time.perf_counter()
        
        if batch_idx+1>=n_steps:
            break




if __name__=='__main__':
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()