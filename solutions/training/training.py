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
from training.af3_dataset import build_af3_dataset, build_sampler

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

    w = w.unsqueeze(-1)

    mu_x_src = torch.sum(x_src * w, dim=-2) / torch.sum(w, dim=-2)
    mu_x_tgt = torch.sum(x_tgt * w, dim=-2) / torch.sum(w, dim=-2)

    mu_x_src = mu_x_src[..., None, :]
    mu_x_tgt = mu_x_tgt[..., None, :]

    x_src = x_src - mu_x_src
    x_tgt = x_tgt - mu_x_tgt

    w = w.squeeze()
    H = torch.einsum('...l,...li,...lj->...ij', w, x_tgt, x_src)
    U, _, Vh = torch.linalg.svd(H)

    F = torch.eye(3, device=device).reshape((1,) * len(batch_shape) + (3, 3))
    F = F.broadcast_to(batch_shape + (3, 3)).contiguous()
    F[..., 2, 2] = torch.linalg.det(U@Vh)

    R = U@F@Vh
    x_aligned = torch.einsum('...ij,...nj->...ni', R, x_src) + mu_x_tgt

    return x_aligned.detach()

def mse_loss(x_out, batch_with_labels: dict):
    batch = batch_with_labels['batch']

    
    x_gt = [torch.tensor(data['atom_array'].coord, device=x_out.device) for data in batch_with_labels["original_data"]]
    x_gt = utils.pad_to_shape(collate_batch(x_gt), x_out.shape)

    alpha_dna = 5
    alpha_rna = 5
    alpha_ligand = 10
    w = 1 + batch.token_features.is_dna * alpha_dna + batch.token_features.is_rna * alpha_rna + batch.token_features.is_ligand * alpha_ligand

    x_gt_mask = ~(x_gt.isnan().any(dim=-1))
    x_gt[~x_gt_mask] = 0

    w = batch.reference_features.to_atom_layout(w, has_atom_dimension=False)
    w = w * batch.reference_features.mask * x_gt_mask

    x_gt_aligned = weighted_align(x_gt, x_out, w)
    mse = 1/3 * torch.sum(w * (x_out - x_gt_aligned).square().sum(dim=-1), axis=-1) / x_gt_mask.sum(dim=-1)
    
    return mse

def stack_trace_analysis(memory_snapshot, workspace_root='alphafold3-decoded/solutions'):
    trace_entries = memory_snapshot['device_traces'][0]

    files_to_analyze = set()

    dedup_counter = 0
    for entry in trace_entries:
        new_frames = []
        new_frames_ids = []
        for frame in entry['frames']:
            if workspace_root in frame['filename'] and not id(frame) in new_frames_ids:
                new_frames.append(frame)
                new_frames_ids.append(id(frame))
            elif workspace_root in frame['filename']:
                dedup_counter += 1

        entry['frames'] = copy.deepcopy(new_frames)
        files_to_analyze |= set(f['filename'] for f in new_frames)
    
    print(f'Dedup: {dedup_counter}')
    class_ranges = {
        f: {} for f in files_to_analyze
    }
    for filename in files_to_analyze:
        src_tree = ast.parse(Path(filename).read_text())
        for node in ast.walk(src_tree):
            if isinstance(node, ast.ClassDef):
                start, end = node.lineno, node.end_lineno
                class_ranges[filename][(start, end)] = node.name

    def get_class(filename, lineno):
        for (start, end), name in class_ranges[filename].items():
            if start <= lineno <= end:
                return name
        return None

    for j, entry in enumerate(trace_entries):
        for i, frame in enumerate(entry['frames']):
            cls = get_class(frame['filename'], frame['line'])
            frame['old_filename'] = frame['filename']
            if cls:
                frame['filename'] = f'{cls}.{frame["name"]} ({frame["old_filename"]})'
            else:
                frame['filename'] = f'{frame["name"]} ({frame["old_filename"]})'



    
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
        if Path(filename).exists():
            file_data[filename] = Path(filename).read_text()
        elif filename not in skipped_files:
            # print(f'Skipping file {filename}')
            skipped_files.append(filename)

    snapshot['source_code'] = file_data
    
    with open(snapshot_filename, 'wb') as f:
        pickle.dump(snapshot, f)

    


def main():
    torch.cuda.memory._record_memory_history(
        True,
        trace_alloc_max_entries=1_000_000,
        trace_alloc_record_context=True,
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

    t0 = time.time()
    dataset = build_af3_dataset(config)
    sampler = build_sampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=collate_batch)
    samples = next(iter(loader))
    # samples['batch'].reference_features.setup_block_mask()
    print(f'Featurization complete. Took {time.time() - t0:.1f} seconds.')
    with open('test_samples_384.pkl', 'wb') as f:
        pickle.dump(samples, f)

    # Currently, only working with 256 and torch version 2.9 or 2.10
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
    
    model.compile(fullgraph=True)
    print('Compiled.')

    batch = samples['batch']
    batch.reference_features.setup_block_mask()
    batch.token_features.setup_block_mask()
    n_seq = batch.token_features.mask.shape[1]

    for i in range(1):
        print(f'Iteration {i}...')
        t0 = time.time()
        x_pred = model(samples['batch'])
        print('Forward complete.')
        # loss = x_pred[0].sum()
        loss = mse_loss(x_pred, samples)
        loss.backward()
        print(f'Backward complete. Took {time.time()-t0:.1f} seconds.')

    # model.evoformer.forward(samples['batch'])
    # loss = mse_loss(x_pred, samples)

    snapshot_filename = 'memory_snapshot_pair_offloaded_compiled_48_blocks.pkl'
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
    main()



