import contextlib
import pickle

import torch
from torch_snapkit import memory_snapshot

from config import Config
from diffusion.model import Model
from feature_extraction.feature_extraction import Batch, tree_map
from common.block_sparse_tensor import BlockSparseTensor
from training.af3_dataset import af3_pipeline_none_on_error, build_af3_dataset, build_sampler, collate_batch_drop_none
from training.training_module import AF3TrainingModule

def profile_module(name, module, args=None, kwargs=None, do_backward=False, device='cuda', n_cycle=1):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], schedule=torch.profiler.schedule(warmup=5, active=1, wait=0, repeat=1)) as p:
        for _ in range(6):
            module.zero_grad()
            for _ in range(n_cycle):
                out = module(*args, **kwargs)
                if do_backward:
                    loss = torch.zeros((), device=device)
                    def inc_loss(x):
                        nonlocal loss
                        loss = loss + x.sum()
                    tree_map(inc_loss, out, skip_unconvertible_entries=True)
                    loss.backward()

            p.step()

    resp = p.key_averages().table(sort_by='self_cuda_time_total', row_limit=30)
    with open(f'profiling_{name}.txt', 'w') as f:
        f.write(resp)

    print()
    print(f'========= Profiling {name} ===========')
    print(resp)
    print('')

def profile_evoformer(model, batch, config, bf16=False):
    evoformer = model.evoformer

    sub_batch = tree_map(lambda x: torch.clone(x), batch, skip_unconvertible_entries=True)
    sub_batch.msa_features.msa_feat = sub_batch.msa_features.msa_feat[..., 0]
    sub_batch.msa_features.msa_mask = sub_batch.msa_features.msa_mask[..., 0]

    device='cuda'

    n_tokens = batch.token_features.token_count
    c_s, c_s_input, c_z = config.global_config.c_s, config.global_config.c_s_input, config.global_config.c_z

    s_input = torch.randn((1, n_tokens, c_s_input), device=device)
    s = torch.randn((1, n_tokens, c_s), device=device)
    z = torch.randn((1, n_tokens, n_tokens, c_z), device=device)

    context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

    with context:
        profile_module('bf16_template_embedder', evoformer.template_embedder, args=(batch, z), do_backward=True)
        profile_module('bf16_msa_module', evoformer.msa_module, args=(sub_batch, s_input, z), do_backward=True)
        profile_module('bf16_pairformer', evoformer.pairformer, args=(s, z, sub_batch.token_features), do_backward=True)
        profile_module('bf16_evoformer', evoformer, args=(batch,), do_backward=True)

    

def main():
    config = Config()
    config.global_config.n_cycle = 1
    config.diffusion_config.denoising_steps = 1

    config.global_config.c_s = 64
    config.global_config.c_z = 32
    config.global_config.c_m = 32

    config.evoformer_config.msa_module_config.n_blocks = 1
    config.evoformer_config.template_module_config.n_blocks = 1
    config.evoformer_config.pairformer_config.n_blocks = 8
    config.evoformer_config.pairformer_config.n_head_pairstack = 2
    config.evoformer_config.msa_module_config.n_head_pairstack = 2
    config.evoformer_config.pairformer_config.n_head_att_pair_bias = 2
    config.diffusion_config.n_head_diffusion_transformer = 2

    config.diffusion_config.n_block_diffusion_transformer = 4
    config.diffusion_config.atom_attention_config.c_token = 32
    config.diffusion_config.atom_attention_config.n_block_atom_transformer = 1


    model = Model(config)

    # train_ds = build_af3_dataset(config)
    with open('train_ds.pkl', 'rb') as f:
        train_ds = pickle.load(f)
        train_ds.transform = af3_pipeline_none_on_error(config, is_inference=True)
    # sampler = build_sampler(train_ds)
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers=4, batch_size=config.training_config.micro_batch_size, collate_fn=lambda x: collate_batch_drop_none(x, config))

    batch_with_labels = next(iter(train_dl))

    with open('test_batch.pkl', 'wb') as f:
        pickle.dump(batch_with_labels, f)

    with open('test_batch.pkl', 'rb') as f:
        batch_with_labels = pickle.load(f)

    rank = 0; world_size=1; device='cuda:0'
    config.training_config.batch_size = config.training_config.micro_batch_size * world_size
    model.to(device)
    model.regional_compile()
    
    training_model = AF3TrainingModule(model, config, num_devices=world_size)
    opt = training_model.configure_optimizers()



    batch_with_labels = tree_map(lambda x: x.to(device), batch_with_labels, skip_unconvertible_entries=True)

    batch: Batch = batch_with_labels['batch']
    batch.reference_features.setup_block_mask(config.training_config.diffusion_micro_batch_size)
    batch.token_features.setup_block_mask()
    batch.reference_features.materialize()

    # profile_evoformer(model, batch, config, bf16=True)
    profile_diffusion_module(model, batch, config, bf16=True)



def profile_diffusion_module(model, batch, config, bf16=False):
    n_atoms = batch.reference_features.atom_count
    n_tokens = batch.token_features.token_count
    c_s, c_s_input, c_z = config.global_config.c_s, config.global_config.c_s_input, config.global_config.c_z
    c_token = config.diffusion_config.atom_attention_config.c_token
    c_atom = config.diffusion_config.atom_attention_config.c_atom
    c_atompair = config.diffusion_config.atom_attention_config.c_atompair
    diff_batch_shape = (config.training_config.diffusion_micro_batch_size, 1)

    device = 'cuda'

    def expand_to_diffusion_shape(x):
        return x[None, ...].broadcast_to((config.training_config.diffusion_micro_batch_size,)+x.shape)

    x_gt = torch.randn((1, n_atoms, 3), device=device)
    t_hat = torch.randn(diff_batch_shape, device=device)
    s_input = torch.randn((1, n_tokens, c_s_input), device=device)
    s_trunk = torch.randn((1, n_tokens, c_s), device=device)
    z_trunk = torch.randn((1, n_tokens, n_tokens, c_z), device=device)
    rel_feat = torch.randn((1, n_tokens, n_tokens, config.global_config.rel_feat_dim), device=device)

    x_gt, s_input, s_trunk, z_trunk, rel_feat, batch = tree_map(expand_to_diffusion_shape, (x_gt, s_input, s_trunk, z_trunk, rel_feat, batch), skip_unconvertible_entries=True)

    batch.reference_features.materialize()
    reference_features = batch.reference_features
    token_features = batch.token_features

    r = torch.randn(diff_batch_shape + (n_atoms, 3), device=device)
    a = torch.randn(diff_batch_shape + (n_tokens, c_token), device=device)
    q_skip = torch.randn(diff_batch_shape + (n_atoms, c_atom), device=device)
    c_skip = torch.randn(diff_batch_shape + (n_atoms, c_atom), device=device)
    p_skip_prep = torch.randn(diff_batch_shape + (n_atoms, 1, c_atompair), device=device)
    p_skip = BlockSparseTensor.broadcast_up(p_skip_prep, reference_features.block_mask_diffusion, diff_batch_shape) 


    context = contextlib.nullcontext() if not bf16 else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

    with context:
        profile_module('bf16_diffusion_module', model.diffusion_module, args=(x_gt, t_hat, s_input, s_trunk, z_trunk, rel_feat, batch), do_backward=True, n_cycle=8)
        profile_module('bf16_diff_cond', model.diffusion_module.diffusion_conditioning, args=(t_hat, s_input, s_trunk, z_trunk, rel_feat), do_backward=True, n_cycle=8)
        profile_module('bf16_diff_att_enc', model.diffusion_module.atom_att_enc, args=(reference_features,), kwargs={'r': r, 's_trunk': s_trunk, 'z': z_trunk}, do_backward=True, n_cycle=8)
        profile_module('bf16_diff_trans', model.diffusion_module.diffusion_transformer, args=(a, s_trunk, z_trunk, token_features.block_mask), do_backward=True, n_cycle=8)
        profile_module('bf16_diff_att_dec', model.diffusion_module.atom_att_dec, args=(a, q_skip, c_skip, p_skip, reference_features), do_backward=True, n_cycle=8)


        
if __name__=='__main__':
    main()