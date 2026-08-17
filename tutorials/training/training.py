import os

import tqdm
import lightning as L
# Set so that Atomworks does not raise a warning, we don't need to actually download the mirrors for this notebook.
os.environ["PDB_MIRROR_PATH"] = ""
os.environ["CCD_MIRROR_PATH"] = ""

import time
import tensortrace as ttr

import torch
import pickle


from torch_snapkit import memory_snapshot

from common import utils
from diffusion.model import Model
from training.training_module import AF3TrainingModule, mse_loss
from training import af3_dataset
from config import Config
from feature_extraction.feature_extraction import Batch, collate_batch, tree_map
from training.af3_dataset import build_af3_dataset, build_sampler, collate_batch_drop_none

def training_forward(model: Model, batch_with_labels: dict, config: Config, ):
    diffusion_batch_size = config.training_config.diffusion_micro_batch_size
    total_diffusion_batch_size = config.training_config.diffusion_batch_size
    assert total_diffusion_batch_size % diffusion_batch_size == 0, 'Total and per-micro-batch diffusion_batch_size need to be equal.'
    t0 = time.time()
    batch = batch_with_labels['batch']
    x_gt_shape = batch.reference_features.positions.shape
    batch_shape = batch.reference_features.positions.shape[:-2]
    n_atoms = batch.reference_features.positions.shape[-2]
    device = batch.reference_features.positions.device

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        s_input, s_trunk, z_trunk, rel_feat = model.evoformer(batch)
    t1 = time.time()
    print(f'Evoformer complete {t1-t0:.1f} s')

    x_gt = [torch.tensor(atom_array.coord, device=device) for atom_array in batch_with_labels["atom_array"]]
    x_gt = utils.pad_to_shape(collate_batch(x_gt), x_gt_shape)
    x_gt_mask = ~(x_gt.isnan().any(dim=-1))
    x_gt[~x_gt_mask] = 0

    diffusion_batch_shape = (diffusion_batch_size,) + batch_shape
    x_gt = x_gt[None, ...].broadcast_to(diffusion_batch_shape + (n_atoms, 3))
    sigma_data = config.diffusion_config.sigma_data

    def expand_batch_to_diffusion_shape(x):
        return x[None, ...].broadcast_to((diffusion_batch_size,) + x.shape)


    batch = tree_map(expand_batch_to_diffusion_shape, batch, skip_unconvertible_entries=True)
    s_input, s_trunk, z_trunk, rel_feat = tree_map(expand_batch_to_diffusion_shape, [s_input, s_trunk, z_trunk, rel_feat])

    s_input_d = s_input.detach().requires_grad_(True)
    s_trunk_d = s_trunk.detach().requires_grad_(True)
    z_trunk_d = z_trunk.detach().requires_grad_(True)
    rel_feat_d = rel_feat.detach().requires_grad_(True)

    num_repeats = total_diffusion_batch_size // diffusion_batch_size
    total_loss = torch.tensor(0, device=device, dtype=float)
    for _ in range(num_repeats):
        noise_amount = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(diffusion_batch_shape, device=device))
        noise = torch.randn(x_gt.shape, device=device) * noise_amount[..., None, None]
        x_gt_randaug = model.diffusion_sampler.center_random_aug(x_gt, batch.reference_features)
        x_gt_noisy = x_gt_randaug + noise

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_denoised = model.diffusion_module.forward(x_gt_noisy, noise_amount, s_input_d, s_trunk_d, z_trunk_d, rel_feat_d, batch)
        loss = mse_loss(x_denoised, x_gt, x_gt_mask, batch).mean() / num_repeats
        loss.backward()
        total_loss += loss.detach()

    t2 = time.time()
    print(f'Diff back {t2 - t1:.1f} s')

    torch.autograd.backward([s_input, s_trunk, z_trunk], [s_input_d.grad, s_trunk_d.grad, z_trunk_d.grad])
    print(f'Evo back {time.time()-t2:.1f} s')
    
    return total_loss

def basic_test():
    config = Config()
    config.global_config.n_cycle = 1
    config.diffusion_config.denoising_steps = 1
    config.global_config.c_m = 32
    config.global_config.c_z = 64
    config.global_config.c_s = 384
    config.evoformer_config.msa_module_config.n_blocks = 1
    
    device='cuda'
    model = Model(config)
    model.to(device)
    block = model.evoformer.msa_module.blocks[0].core.triangle_att_starting
    block.compile()

    for i in tqdm.tqdm(range(5)):
        z = ttr.load('z_msa_mod').to(device)
        single_mask = ttr.load('single_mask_msa_mod').to(device)

        # out = block(z, single_mask, activation_checkpointing=False)
        out = block(z, single_mask)
        out.sum().backward()

def basic_step():
    config = Config()
    config.global_config.n_cycle = 1
    config.diffusion_config.denoising_steps = 1
    # config.global_config.c_m = 32
    # config.global_config.c_z = 64
    # config.global_config.c_s = 384
    # config.evoformer_config.msa_module_config.n_blocks = 1

    # dataset = build_af3_dataset(config)
    # torch.random.manual_seed(35)
    # sampler = build_sampler(dataset)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8, collate_fn=lambda x: collate_batch_drop_none(x, config))

    # dl_iter = iter(loader)


    # samples = [next(dl_iter) for i in range(5)]

    # with open('test_samples.pkl', 'wb') as f:
    #     pickle.dump(samples, f)

    model = Model(config)
    model.to('cuda')
    model.regional_compile()

    with open('test_samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    samples = tree_map(lambda x: x.to(device='cuda'), samples, skip_unconvertible_entries=True)
    optim = torch.optim.Adam(model.parameters())

    for i in range(5):
        t = time.time()

        it_samples = samples[i]

        if it_samples is None:
            raise ValueError(f'Failed in iteration {i}')
        batch = it_samples['batch']
        batch.reference_features.setup_block_mask(config.training_config.diffusion_micro_batch_size)
        batch.token_features.setup_block_mask()
        
        optim.zero_grad()
        
        training_forward(model, it_samples, config)
        print(f'Tokk {time.time()-t:.1f} s')
        optim.step()





def main():
    config = Config()
    config.global_config.n_cycle = 1
    config.diffusion_config.denoising_steps = 1
    # config.evoformer_config.pairformer_config.n_blocks = 1
    # config.evoformer_config.pairformer_config.n_transition_pairstack = 1
    # config.evoformer_config.pairformer_config.n_transition = 1
    # config.diffusion_config.denoising_steps = 2
    # config.global_config.c_m = 4
    # config.global_config.c_z = 32
    # config.global_config.c_s = 32
    # config.evoformer_config.pairformer_config.n_head_pairstack = 1
    # config.evoformer_config.pairformer_config.n_head_att_pair_bias = 1
    # config.evoformer_config.msa_module_config.n_head_pairstack = 1
    # config.evoformer_config.msa_module_config.n_transition = 1
    # config.evoformer_config.msa_module_config.n_transition_pairstack = 1
    # config.diffusion_config.n_head_diffusion_transformer = 1
    # config.diffusion_config.n_block_diffusion_transformer = 1
    # config.diffusion_config.atom_attention_config.c_token = 64

    t0 = time.time()
    dataset = build_af3_dataset(config)
    sampler = build_sampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8, collate_fn=lambda x: collate_batch_drop_none(x, config))

    # Force initialization by accessing dynamo first
    # Currently only works with export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
    # _ = torch._dynamo
    # torch._functorch.config.activation_memory_budget = 0.99

    model = Model(config)
    # params = torch.load('data/params/af3_pytorch.pt')
    # model.load_state_dict(params)
    
    # model.evoformer.compile(fullgraph=True)
    # model.diffusion_module.compile(fullgraph=True)
    # torch.compiler.reset()
    model.regional_compile()

    af3_training_module = AF3TrainingModule(model, config, num_devices=2)
    trainer = L.Trainer(max_steps=4, accelerator='gpu', devices=[0])

    # TODO: check for correctness (does checkpointing use kwargs?)
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    trainer.fit(af3_training_module, train_dataloaders=loader)




if __name__=='__main__':
    # Use this to get frame-tracing for allocations in backward pass
    # with torch.autograd.detect_anomaly():
    # with memory_snapshot('training_mixed', share=True, share_code='kilis_new_af3_secret3'):
    # main()
    # with memory_snapshot('pl_training', share=True, share_code='kilis_new_af3_secret4'):
    basic_step()



