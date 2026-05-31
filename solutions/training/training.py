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



def main():
    # torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    torch.cuda.memory._record_memory_history(
        True,
        trace_alloc_max_entries=1_000_000,
        trace_alloc_record_context=True,
    )
    config = Config()
    config.global_config.n_cycle = 1
    config.evoformer_config.pairformer_config.n_blocks = 2
    config.diffusion_config.denoising_steps = 1

    # af3_dataset.extract_top1000_entries()
    # dataset = build_af3_dataset(config)
    # sampler = build_sampler(dataset)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=8, collate_fn=collate_batch)
    # samples = next(iter(loader))
    # with open('test_samples.pkl', 'wb') as f:
    #     pickle.dump(samples, f)

    with open('test_samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    device = 'cuda:0'
    samples['batch'] = tree_map(lambda x: x.to(device=device), samples['batch'])
    model = Model(config)
    # params = torch.load('data/params/af3_pytorch.pt')
    # model.load_state_dict(params)
    model = model.to(device=device)
    model.eval()
    x_pred = model.forward(samples['batch'])
    # loss = mse_loss(x_pred, samples)

    try:
        torch.cuda.memory._dump_snapshot('memory_snapshot.pkl')
    except:
        print(f'Error saving memory snapshot.')
    
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__=='__main__':
    main()


