import contextlib
import torch.distributed as dist

import lightning as L
import torch
from common import utils
from torch._subclasses.fake_tensor import unset_fake_temporarily

from config import Config
from diffusion.model import Model
from feature_extraction.feature_extraction import Batch, collate_batch, tree_map

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

class AF3TrainingModule:
    def __init__(self, model, config: Config, num_devices: int):
        super().__init__()
        self.distributed = num_devices > 1
        self.model = model
        
        self.config = config
        self.num_devices = num_devices
        assert config.training_config.batch_size % (num_devices*config.training_config.micro_batch_size) == 0, 'Batch size must be divisible by num_devices*micro_batch_size.'
        assert config.training_config.diffusion_batch_size % config.training_config.diffusion_micro_batch_size == 0, 'diffusion_batch_size must be divisible by diffusion_micro_batch_size.'
        self.global_grad_accum_steps = config.training_config.batch_size // (num_devices * config.training_config.micro_batch_size)
        self.automatic_optimization = False

    def _diffusion_step(self, x_gt, x_gt_mask, s_input, s_trunk, z_trunk, rel_feat, batch, bf16: bool, do_backward: bool):
        batch_shape = x_gt.shape[:-2]
        device = x_gt.device
        training_config = self.config.training_config
        diffusion_batch_shape = (training_config.diffusion_micro_batch_size,) + batch_shape
        sigma_data = self.config.diffusion_config.sigma_data
        amp_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

        model = self.model

        def expand_to_diffusion_shape(x):
            return x[None, ...].broadcast_to((training_config.diffusion_micro_batch_size,)+x.shape)

        x_gt, s_input, s_trunk, z_trunk, rel_feat, batch = tree_map(expand_to_diffusion_shape, (x_gt, s_input, s_trunk, z_trunk, rel_feat, batch), skip_unconvertible_entries=True)
        batch.reference_features.materialize()

        assert not rel_feat.requires_grad

        num_repeats = training_config.diffusion_batch_size // training_config.diffusion_micro_batch_size

        total_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for _ in range(num_repeats):
            noise_amount = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(diffusion_batch_shape, device=device))
            noise = torch.randn(x_gt.shape, device=device) * noise_amount[..., None, None]
            x_gt_randaug = model.diffusion_sampler.center_random_aug(x_gt, batch.reference_features)
            x_gt_noisy = x_gt_randaug + noise


            with amp_context:
                x_denoised = model.diffusion_module.forward(x_gt_noisy, noise_amount, s_input, s_trunk, z_trunk, rel_feat, batch)

                loss = mse_loss(x_denoised, x_gt, x_gt_mask, batch).mean() / (num_repeats * self.global_grad_accum_steps)
            if do_backward:
                loss.backward()
            total_loss += loss.detach()

        return total_loss

    def sync_grads(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        for p in params:
            if p.requires_grad is None:
                p.grad = torch.zeros_like(p)
        grads = [p.grad for p in params]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)

        sq_grad_norm = torch.zeros((), device=params[0].device)
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)
            sq_grad_norm += g.double().pow(2).sum()

        return sq_grad_norm.sqrt().item()


    def _shared_step(self, batch_with_labels: dict, batch_idx: int, stage: str, bf16: bool=True):
        do_backward = stage=='train'
        batch: Batch = batch_with_labels['batch']
        x_gt_shape = batch.reference_features.positions.shape
        device = batch.reference_features.positions.device

        model = self.model

        with unset_fake_temporarily():
            batch.reference_features.setup_block_mask(self.config.training_config.diffusion_micro_batch_size)
            batch.token_features.setup_block_mask()
            batch.reference_features.materialize()
        

        amp_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

        with amp_context:
            s_input, s_trunk, z_trunk, rel_feat = model.evoformer(batch)

        x_gt = [torch.tensor(atom_array.coord, device=device) for atom_array in batch_with_labels["atom_array"]]
        x_gt = utils.pad_to_shape(collate_batch(x_gt), x_gt_shape)
        x_gt_mask = ~(x_gt.isnan().any(dim=-1))
        x_gt[~x_gt_mask] = 0

        s_input_d = s_input.detach().requires_grad_(True)
        s_trunk_d = s_trunk.detach().requires_grad_(True)
        z_trunk_d = z_trunk.detach().requires_grad_(True)

        loss = self._diffusion_step(x_gt, x_gt_mask, s_input_d, s_trunk_d, z_trunk_d, rel_feat, batch, bf16=bf16, do_backward=do_backward)

        if do_backward:
            torch.autograd.backward([s_input, s_trunk, z_trunk], [s_input_d.grad, s_trunk_d.grad, z_trunk_d.grad])

        return loss * self.global_grad_accum_steps

    def training_step(self, batch_with_labels: dict, batch_idx: int):
        return self._shared_step(batch_with_labels, batch_idx, stage='train')

    def validation_step(self, batch_with_labels: dict, batch_idx: int):
        return self._shared_step(batch_with_labels, batch_idx, stage='val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


