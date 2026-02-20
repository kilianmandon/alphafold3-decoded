from torch import nn
import torch
import tqdm

from config import Config
from input_embedding.atom_attention import AtomAttentionDecoder, AtomAttentionEncoder
from feature_extraction.feature_extraction import Batch
from feature_extraction.ref_struct_features import RefStructFeatures

from common.modules import Transition, DiffusionTransformer
import common.utils as utils


class DiffusionModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        diffusion_config = config.diffusion_config
        global_config = config.global_config
        sigma_data = diffusion_config.sigma_data
        c_token = diffusion_config.atom_attention_config.c_token
        c_s = global_config.c_s
        c_z = global_config.c_z
        target_feat_dim = global_config.target_feat_dim
        rel_feat_dim = global_config.rel_feat_dim
        c_fourier = diffusion_config.c_fourier
        n_head_diffusion_transformer = diffusion_config.n_head_diffusion_transformer
        n_block_diffusion_transformer = diffusion_config.n_block_diffusion_transformer

        self.diffusion_conditioning = DiffusionConditioning(c_s, c_z, target_feat_dim, rel_feat_dim, sigma_data, c_fourier)
        self.atom_att_enc = AtomAttentionEncoder(c_s, c_z, diffusion_config.atom_attention_config, use_trunk=True)
        self.diffusion_transformer = DiffusionTransformer(c_token, c_z, n_head=n_head_diffusion_transformer, c_s=c_s, n_blocks=n_block_diffusion_transformer)
        self.atom_att_dec = AtomAttentionDecoder(diffusion_config.atom_attention_config)

        self.layer_norm_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = nn.Linear(c_s, c_token, bias=False)
        self.layer_norm_a = nn.LayerNorm(c_token, bias=False)

        self.sigma_data = sigma_data

    def forward(self, x_noisy, t_hat, s_inputs, s_trunk, z_trunk, rel_enc, batch: Batch):
        # x_noisy has shape (**batch_shape, N_blocks, 32, 3)
        # t_hat has shape (**batch_shape, )

        ref_struct = batch.ref_struct
        token_features = batch.token_features
        s, z = self.diffusion_conditioning(t_hat, s_inputs, s_trunk, z_trunk, rel_enc)
        r=x_noisy / torch.sqrt(t_hat**2+self.sigma_data**2)[..., None, None]


        a, (q_skip, c_skip, p_skip) = self.atom_att_enc(ref_struct, r=r, s_trunk=s_trunk, z=z)


        a += self.linear_s(self.layer_norm_s(s))
        a = self.diffusion_transformer(a, s, z, token_features.block_mask)

        a = self.layer_norm_a(a)

        r_update = self.atom_att_dec.forward(a, q_skip, c_skip, p_skip, ref_struct)

        d_skip = self.sigma_data**2 / (self.sigma_data**2+t_hat**2)
        d_scale = self.sigma_data * t_hat / torch.sqrt(self.sigma_data**2 + t_hat**2)
        d_skip = d_skip[..., None, None]
        d_scale = d_scale[..., None, None]
        
        x_out = d_skip * x_noisy + d_scale * r_update

        return x_out



class DiffusionConditioning(nn.Module):
    def __init__(self, c_s, c_z, target_feat_dim, rel_feat_dim, sigma_data, c_fourier):
        super().__init__()

        self.sigma_data = sigma_data
        self.linear_z = nn.Linear(rel_feat_dim + c_z, c_z, bias=False)
        self.layer_norm_z = nn.LayerNorm(rel_feat_dim + c_z, bias=False)
        self.z_transition = nn.ModuleList([Transition(c_z, n=2) for _ in range(2)])

        self.layer_norm_s = nn.LayerNorm(target_feat_dim + c_s, bias=False)
        self.linear_s = nn.Linear(target_feat_dim + c_s, c_s, bias=False)
        self.s_transition = nn.ModuleList([Transition(c_s, n=2) for _ in range(2)])

        self.layer_norm_fourier = nn.LayerNorm(c_fourier, bias=False)
        self.linear_fourier = nn.Linear(c_fourier, c_s, bias=False)

        self.fourier_w = nn.Parameter(torch.randn((c_fourier,)), requires_grad=False)
        self.fourier_b = nn.Parameter(torch.randn((c_fourier,)), requires_grad=False)

    def fourier_embedding(self, t_hat):
        # t_hat has shape (**batch_shape,)
        # out should have shape (**batch_shape, 1, c_fourier)

        c_noise = 1/4 * torch.log(t_hat/self.sigma_data)
        c_noise = c_noise[..., None, None]
        x = c_noise * self.fourier_w + self.fourier_b
        return torch.cos(2 * torch.pi * x)

    def forward(self, t_hat, s_inputs, s_trunk, z_trunk, rel_feat):
        z = torch.cat((z_trunk, rel_feat), dim=-1)
        z = self.linear_z(self.layer_norm_z(z))
        for block in self.z_transition:
            z += block(z)

        s = torch.cat((s_trunk, s_inputs), dim=-1)
        tf_mask = torch.ones(s.shape[-1], device=s.device, dtype=bool)
        tf_mask[415] = tf_mask[447] = False
        s = self.linear_s(apply_layernorm_masked(s, self.layer_norm_s, tf_mask))
        # s = self.linear_s(self.layer_norm_s(s))
        n = self.fourier_embedding(t_hat)
        s += self.linear_fourier(self.layer_norm_fourier(n))

        for block in self.s_transition:
            s += block(s)
        
        return s, z


def apply_layernorm_masked(inp, layer_norm, mask):
    masked_inp = inp[:, mask]
    # masked_mean, masked_var = masked_inp.mean(-1, keepdim=True), masked_inp.var(-1, keepdim=True)
    # return (inp - masked_mean) / torch.sqrt(masked_var + layer_norm.eps) * layer_norm.weight
    fake_layernorm = nn.LayerNorm((masked_inp.shape[-1],), eps=layer_norm.eps, bias=False)
    fake_layernorm.weight.copy_(layer_norm.weight[mask])
    fake_layernorm.to(inp.device, dtype=inp.dtype)
    masked_out = fake_layernorm(masked_inp)
    full_out = torch.zeros_like(inp)
    full_out[:, mask] = masked_out
    return full_out




class DiffusionSampler(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        diffusion_config = config.diffusion_config

        self.denoising_steps = diffusion_config.denoising_steps
        self.gamma_0 = diffusion_config.gamma_0
        self.gamma_min = diffusion_config.gamma_min
        self.noise_scale = diffusion_config.noise_scale
        self.step_scale = diffusion_config.step_scale

        self.sigma_data = diffusion_config.sigma_data
        self.s_min = diffusion_config.s_min
        self.s_max = diffusion_config.s_max
        self.rho = diffusion_config.rho

        self.center_random_aug = CenterRandomAugmentation(diffusion_config.s_trans_center_randaug)

    def noise_schedule(self, t):
        return self.sigma_data * (self.s_max ** (1/self.rho) + t * (self.s_min**(1/self.rho) - self.s_max**(1/self.rho))) ** self.rho

    def forward(self, diffusion_module, s_inputs, s_trunk, z_trunk, rel_enc, batch: Batch, noise_data=None):
        ref_struct = batch.ref_struct
        # q2k_mask has shape (**batch_shape, N_block, 32,)
        batch_shape = s_trunk.shape[:-2]
        device = s_trunk.device
        
        noise_levels = self.noise_schedule(torch.linspace(0, 1, self.denoising_steps+1, device=device))
        x_shape = batch_shape + (ref_struct.atom_count, 3)

        if noise_data is not None:
            x = noise_levels[0] * noise_data['init_pos'].to(dtype=torch.float32)
        else:
            x = noise_levels[0] * torch.randn(x_shape, device=device)

        for i, (c_prev, c) in tqdm.tqdm(enumerate(zip(noise_levels[:-1], noise_levels[1:])), total=self.denoising_steps):

            if noise_data is not None:
                rand_rot = noise_data['aug_rot'][i].to(dtype=torch.float32)
                rand_trans = noise_data['aug_trans'][i].to(dtype=torch.float32)
            else:
                rand_rot = rand_trans = None

            x = self.center_random_aug(x, ref_struct, rand_rot=rand_rot, rand_trans=rand_trans)

            gamma = self.gamma_0 if c > self.gamma_min else 0
            t_hat = c_prev * (gamma + 1)

            if noise_data is not None:
                noise = self.noise_scale * torch.sqrt(t_hat**2 - c_prev**2) * noise_data['noise'][i]
            else:
                noise = self.noise_scale * torch.sqrt(t_hat**2 - c_prev**2) * torch.randn(x_shape, device=device)

            x_noisy = x+noise
            x_denoised = diffusion_module.forward(x_noisy, t_hat, s_inputs, s_trunk, z_trunk, rel_enc, batch)


            delta = (x_noisy-x_denoised)/t_hat
            dt = c - t_hat
            x = x_noisy + self.step_scale * dt * delta

        return x

class CenterRandomAugmentation(nn.Module):
    def __init__(self, s_trans):
        super().__init__()
        self.s_trans = s_trans

    def forward(self, x, ref_struct: RefStructFeatures, rand_rot=None, rand_trans=None):
        # x has shape (**batch_dims, N_atoms, 3)
        device = x.device
        batch_shape = x.shape[:-2]

        x = x - utils.masked_mean(x, ref_struct.mask[..., None], axis=(-2), keepdims=True)

        if rand_rot is None:
            rand_rot = utils.rand_rot(batch_shape, device=device)
            rand_trans = self.s_trans * torch.randn(batch_shape+(1,3), device=device)

        x = torch.einsum('ji,...j->...i', rand_rot, x) + rand_trans

        return x


