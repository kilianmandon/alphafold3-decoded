import datetime
import os
import pickle
from torch_snapkit import memory_snapshot
import tqdm

import torch

from config import Config
from diffusion.model import Model
from feature_extraction.feature_extraction import tree_map
from training.af3_dataset import build_af3_dataset, build_sampler, collate_batch_drop_none
from training.training_module import AF3TrainingModule
import torch.distributed as dist

def setup_ddp():
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=10))

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

    train_ds = build_af3_dataset(config)
    sampler = build_sampler(train_ds)
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers=15, batch_size=1, sampler=sampler, collate_fn=lambda x: collate_batch_drop_none(x, config))

    rank, local_rank, world_size, device = setup_ddp()
    model.to(device)
    model.regional_compile()
    if rank==0:
        print(f'Total world size: {world_size}')

    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    training_model = AF3TrainingModule(ddp_model, config, num_devices=1)
    opt = training_model.configure_optimizers()

    n_steps =  10
    # samples = [batch for _, batch in zip(range(n_steps), train_dl)]
    # with open('test_samples.pkl', 'wb') as f:
    #     pickle.dump(samples, f)

    with open('test_samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    pbar = tqdm.tqdm(samples, total=n_steps, smoothing=1)


    for batch_idx, batch in enumerate(pbar):
        batch = tree_map(lambda x: x.to(device), batch, skip_unconvertible_entries=True)
        loss = training_model.training_step(batch, batch_idx)

        dist.all_reduce(loss, dist.ReduceOp.AVG)
        if rank==0:
            pbar.set_postfix(train_loss=loss.item())


        if (batch_idx+1) % training_model.global_grad_accum_steps:
            opt.step()
            opt.zero_grad()
        
        if batch_idx+1>=n_steps:
            break




if __name__=='__main__':
    # with memory_snapshot('torchrun_basic_two_devices', share=True, share_code='supersecret-kili-sharing-0'):
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()