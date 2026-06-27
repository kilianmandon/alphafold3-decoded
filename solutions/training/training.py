from pathlib import Path
import time
import ast
import copy

import torch
import pickle


from common import utils
from diffusion.model import Model
from training import af3_dataset
from config import Config
from feature_extraction.feature_extraction import Batch, collate_batch, tree_map
from training.af3_dataset import build_af3_dataset, build_sampler, collate_batch_drop_none

import os
# Set so that Atomworks does not raise a warning, we don't need to actually download the mirrors for this notebook.
os.environ["PDB_MIRROR_PATH"] = ""
os.environ["CCD_MIRROR_PATH"] = ""


def weighted_align(x_src, x_tgt, w):
    # x_src has shape (**batch_shape, n_atoms, 3)
    # x_tgt has shape (**batch_shape, n_atoms, 3)
    # w has shape (**batch_shape, n_atoms)
    batch_shape = x_src.shape[:-2]
    n_atoms = x_src.shape[-2]
    device = x_src.device


    mu_x_src = torch.sum(x_src * w[..., None], dim=-2) / torch.sum(w[..., None], dim=-2)
    mu_x_tgt = torch.sum(x_tgt * w[..., None], dim=-2) / torch.sum(w[..., None], dim=-2)

    mu_x_src = mu_x_src[..., None, :]
    mu_x_tgt = mu_x_tgt[..., None, :]

    x_src = x_src - mu_x_src
    x_tgt = x_tgt - mu_x_tgt

    H = torch.einsum('...l,...li,...lj->...ij', w, x_tgt, x_src)
    U, _, Vh = torch.linalg.svd(H)

    F = torch.eye(3, device=device).reshape((1,) * len(batch_shape) + (3, 3))
    F = F.broadcast_to(batch_shape + (3, 3)).contiguous()
    F[..., 2, 2] = torch.linalg.det(U@Vh)

    R = U@F@Vh
    x_aligned = torch.einsum('...ij,...nj->...ni', R, x_src) + mu_x_tgt

    return x_aligned.detach()

def mse_loss(x_out, x_gt, x_gt_mask, batch: Batch):

    alpha_dna = 5
    alpha_rna = 5
    alpha_ligand = 10
    w = 1 + batch.token_features.is_dna * alpha_dna + batch.token_features.is_rna * alpha_rna + batch.token_features.is_ligand * alpha_ligand


    w = batch.reference_features.to_atom_layout(w, has_atom_dimension=False)
    w = w * batch.reference_features.mask * x_gt_mask

    with torch.autocast(device_type="cuda", enabled=False):
        x_gt_aligned = weighted_align(x_gt, x_out, w)

    mse = 1/3 * torch.sum(w * (x_out - x_gt_aligned).square().sum(dim=-1), axis=-1) / x_gt_mask.sum(dim=-1)
    
    return mse


def add_code_file_content_to_snapshot(snapshot_filename):
    with open(snapshot_filename, 'rb') as f:
        snapshot = pickle.load(f)
    
    trace_entries = snapshot['device_traces'][0]

    files_to_analyze = set()
    for entry in trace_entries:
        for frame in entry['frames']:
            files_to_analyze.add(frame['filename'])

    file_data = {}
    skipped_files = []
    for filename in files_to_analyze:
        if filename and Path(filename).exists():
            file_data[filename] = Path(filename).read_text()
        elif filename not in skipped_files:
            # print(f'Skipping file {filename}')
            skipped_files.append(filename)

    snapshot['source_code'] = file_data
    
    with open(snapshot_filename, 'wb') as f:
        pickle.dump(snapshot, f)

    
def training_forward(model: Model, batch_with_labels: dict, config: Config, diffusion_batch_size=24):
    batch = batch_with_labels['batch']
    x_gt_shape = batch.reference_features.positions.shape
    batch_shape = batch.reference_features.positions.shape[:-2]
    n_atoms = batch.reference_features.positions.shape[-2]
    device = batch.reference_features.positions.device

    s_input, s_trunk, z_trunk, rel_feat = model.evoformer(batch)
    print('Evoformer complete')

    x_gt = [torch.tensor(data['atom_array'].coord, device=device) for data in batch_with_labels["original_data"]]
    x_gt = utils.pad_to_shape(collate_batch(x_gt), x_gt_shape)
    x_gt_mask = ~(x_gt.isnan().any(dim=-1))
    x_gt[~x_gt_mask] = 0

    diffusion_batch_shape = (diffusion_batch_size,) + batch_shape
    x_gt = x_gt[None, ...].broadcast_to(diffusion_batch_shape + (n_atoms, 3))
    sigma_data = config.diffusion_config.sigma_data
    noise_amount = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(diffusion_batch_shape, device=device))
    noise = torch.randn(x_gt.shape, device=device) * noise_amount[..., None, None]

    def expand_batch_to_diffusion_shape(x):
        return x[None, ...].broadcast_to((diffusion_batch_size,) + x.shape)


    batch = tree_map(expand_batch_to_diffusion_shape, batch, skip_unconvertible_entries=True)
    s_input, s_trunk, z_trunk, rel_feat = tree_map(expand_batch_to_diffusion_shape, [s_input, s_trunk, z_trunk, rel_feat])


    x_gt_randaug = model.diffusion_sampler.center_random_aug(x_gt, batch.reference_features)
    x_gt_noisy = x_gt_randaug + noise

    x_denoised = model.diffusion_module.forward(x_gt_noisy, noise_amount, s_input, s_trunk, z_trunk, rel_feat, batch)

    loss = mse_loss(x_denoised, x_gt, x_gt_mask, batch).mean()
    return loss

    # x_flat = self.diffusion_sampler(model.diffusion_module,

def main():
    torch.cuda.memory._record_memory_history(
        # True,
        # trace_alloc_max_entries=1_000_000,
        # trace_alloc_record_context=True,
    )

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print('Saving memory snapshot after OOM.')
        filename = f"oom_memory_snapshot.pkl"
        torch.cuda.memory._dump_snapshot(filename)

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

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

    # t0 = time.time()
    # dataset = build_af3_dataset(config)
    # sampler = build_sampler(dataset)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=collate_batch_drop_none)
    # samples = next(iter(loader))
    # print(f'Featurization complete. Took {time.time() - t0:.1f} seconds.')
    # with open('test_samples_384.pkl', 'wb') as f:
    #     pickle.dump(samples, f)

    with open('test_samples_384.pkl', 'rb') as f:
        samples = pickle.load(f)

    device = 'cuda:0'
    samples['batch'] = tree_map(lambda x: x.to(device=device), samples['batch'])

    # Force initialization by accessing dynamo first
    # Currently only works with export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
    # _ = torch._dynamo
    # torch._functorch.config.activation_memory_budget = 0.99

    model = Model(config)
    # params = torch.load('data/params/af3_pytorch.pt')
    # model.load_state_dict(params)
    model = model.to(device=device)
    
    # model.evoformer.compile(fullgraph=True)
    # model.diffusion_module.compile(fullgraph=True)
    # torch.compiler.reset()
    # model.regional_compile()

    batch = samples['batch']
    diffusion_batch_size=24
    batch.reference_features.setup_block_mask(num_diffusion_samples=diffusion_batch_size)
    batch.token_features.setup_block_mask()
    n_seq = batch.token_features.mask.shape[1]

    # TODO: check for correctness (does checkpointing use kwargs?)


    for i in range(1):
        print(f'Iteration {i}...')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            t0 = time.time()
            loss = training_forward(model, samples, config, diffusion_batch_size=diffusion_batch_size)
            print('Forward complete.')
            loss.backward()
            print(f'Backward complete. Took {time.time()-t0:.1f} seconds.')


    snapshot_filename = f'memory_snapshot_x{diffusion_batch_size}_small_regional_compile_bf16_b1_no_backward.pkl'
    torch.cuda.memory._dump_snapshot(snapshot_filename)
    add_code_file_content_to_snapshot(snapshot_filename)
    
    torch.cuda.memory._record_memory_history(enabled=None)

def test():
    a = torch.zeros((5,), device='cuda').long()
    x = torch.zeros((3,), device='cuda', requires_grad=True)
    a = torch.nn.functional.one_hot(a, 3)

    s = (a@x).sum()
    return s

if __name__=='__main__':
    # Use this to get frame-tracing for allocations in backward pass
    # with torch.autograd.detect_anomaly():
    main()



