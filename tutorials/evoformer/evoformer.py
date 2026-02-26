import copy

from config import Config, MSAModuleConfig, PairformerConfig, TemplateModuleConfig
from feature_extraction.feature_extraction import Batch
from feature_extraction.token_features import TokenFeatures
import torch
from torch import nn
from torch.nn import functional as F
import tqdm
from torch.nn.attention.flex_attention import flex_attention

from common.modules import AttentionPairBias, Transition
from input_embedding.input_embedder import InputEmbedder


class Evoformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        global_config = config.global_config
        evoformer_config = config.evoformer_config
        input_embedding_config = config.input_embedding_config
        n_cycle = global_config.n_cycle
        c_s = global_config.c_s
        c_m = global_config.c_m
        c_z = global_config.c_z
        msa_feat_dim = global_config.msa_feat_dim
        target_feat_dim = global_config.c_s_input
        rel_feat_dim = global_config.rel_feat_dim
        

        self.input_embedder = InputEmbedder(c_s, c_z, target_feat_dim, rel_feat_dim, input_embedding_config)

        self.layer_norm_prev_z = nn.LayerNorm(c_z)
        self.prev_z_embedding = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_prev_s = nn.LayerNorm(c_s)
        self.prev_s_embedding = nn.Linear(c_s, c_s, bias=False)

        self.template_embedder = TemplateEmbedder(c_z, evoformer_config.template_module_config)
        self.msa_module = MSAModule(c_m, c_z, msa_feat_dim, target_feat_dim, evoformer_config.msa_module_config)
        self.pairformer = PairFormer(c_s, c_z, evoformer_config.pairformer_config)
        self.n_cycle = n_cycle
        self.c_s = c_s
        self.c_z = c_z

    def forward(self, batch: Batch):
        c_s = self.c_s
        c_z = self.c_z

        msa_features = batch.msa_features
        token_features = batch.token_features

        batch_shape = msa_features.target_feat.shape[:-2]
        N_token = msa_features.target_feat.shape[-2]
        device = msa_features.target_feat.device

        s_input, s_init, z_init, rel_feat = self.input_embedder(batch)

        prev_s = torch.zeros(batch_shape+(N_token, c_s), device=device, dtype=torch.float32)
        prev_z = torch.zeros(batch_shape+(N_token, N_token, c_z), device=device, dtype=torch.float32)

        for i in tqdm.tqdm(range(self.n_cycle)):
            sub_batch = copy.deepcopy(batch)
            sub_batch.msa_features.msa_feat = sub_batch.msa_features.msa_feat[..., i]
            sub_batch.msa_features.msa_mask = sub_batch.msa_features.msa_mask[..., i]

            z = z_init + self.prev_z_embedding(self.layer_norm_prev_z(prev_z))
            z += self.template_embedder(batch, z)
            # Note: += in the paper for the next line, not +
            z = self.msa_module(sub_batch, s_input, z)
            s = s_init + self.prev_s_embedding(self.layer_norm_prev_s(prev_s))

            s, z = self.pairformer(s, z, token_features)
            prev_s, prev_z = s, z

        return s_input, s, z, rel_feat


class TemplateEmbedder(nn.Module):
    def __init__(self, c_z, config: TemplateModuleConfig):
        super().__init__()
        self.n_templates = config.n_templates
        self.c_in = config.c_in
        self.c = config.c

        self.linear_a = nn.Linear(self.c_in, config.c, bias=False)
        self.linear_z = nn.Linear(c_z, config.c, bias=False)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.layer_norm_v = nn.LayerNorm(config.c)
        self.linear_out = nn.Linear(config.c, c_z, bias=False)
        self.pair_stack = nn.ModuleList(
            [PairStack(config.c, config.n_head_pairstack, config.n_transition_pairstack, config.p_dropout_pairstack) for _ in range(config.n_blocks)])

    def forward(self, batch: Batch, z: torch.Tensor):
        target_feat = batch.msa_features.target_feat
        batch_shape = target_feat.shape[:-2]
        n_tokens = target_feat.shape[-2]
        n_templates = self.n_templates
        single_mask = batch.token_features.mask
        device = target_feat.device

        dummy_a = torch.zeros(batch_shape+(n_tokens, n_tokens, n_templates, self.c_in), device=device, dtype=torch.float32)
        dummy_aatype = torch.zeros(batch_shape+(n_tokens,), device=device).long()
        dummy_aatype = F.one_hot(dummy_aatype, 31)
        dummy_a[..., 40:71] = dummy_aatype[..., None, :, None, :]
        dummy_a[..., 71:102] = dummy_aatype[..., :, None, None, :]
        u = torch.zeros(batch_shape+(n_tokens, n_tokens, self.c), device=device, dtype=torch.float32)
        for i in range(n_templates):
            v = self.linear_z(self.layer_norm_z(z)) + \
                self.linear_a(dummy_a[..., i, :])
            for block in self.pair_stack:
                v = block(v, single_mask)
            u += self.layer_norm_v(v)

        u = u / n_templates
        u = self.linear_out(torch.relu(u))

        return u


class OuterProductMean(nn.Module):
    # def __init__(self, c_m=64, c=32, c_z=128):
    def __init__(self, c_m, c_z, c):
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_a = nn.Linear(c_m, c, bias=False)
        self.linear_b = nn.Linear(c_m, c, bias=False)
        self.linear_out = nn.Linear(c**2, c_z)

    def forward(self, msa_feat: torch.Tensor, msa_mask: torch.Tensor):
        # msa_feat: Shape (*, N_seq, N_tokens, c_m)
        # mask: Shape (*, N_seq, N_tokens)
        N_seq, N_tokens = msa_feat.shape[-3:-1]
        m = self.layer_norm(msa_feat)
        m_a = self.linear_a(m) * msa_mask[..., None]
        m_b = self.linear_b(m) * msa_mask[..., None]

        ab = torch.einsum('...sic,...sjd->...ijcd', m_a, m_b)
        ab = ab.flatten(start_dim=-2)
        z = self.linear_out(ab)

        norm = torch.einsum('...si,...sj->...ij', msa_mask, msa_mask)
        z = z / (norm[..., None] + 1e-3)
        return z


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, c_m, c_z, c, n_head):
        super().__init__()

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_v = nn.Linear(c_m, c*n_head, bias=False)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_m, c*n_head, bias=False)
        self.linear_out = nn.Linear(c*n_head, c_m, bias=False)
        self.N_head = n_head
        self.c = c

    def forward(self, m, z, single_mask):
        # m has shape (*, N_seq, N_token, c_m)
        # z has shape (*, N_token, N_token, c_z)
        m = self.layer_norm_m(m)
        v = self.linear_v(m).unflatten(-1, (self.N_head, self.c))
        b = self.linear_b(self.layer_norm_z(z))
        g = torch.sigmoid(self.linear_g(m))

        b += -1e9 * ~single_mask[..., None, :, None]

        w = torch.softmax(b, dim=-2)
        o = torch.einsum('...ijh,...sjhc->...sihc', w, v)
        o = g * o.flatten(start_dim=-2)
        m = self.linear_out(o)
        return m



class TriangleMultiplication(nn.Module):
    def __init__(self, c_z, c, outgoing=True):
        super().__init__()

        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_a1 = nn.Linear(c_z, c, bias=False)
        self.linear_a2 = nn.Linear(c_z, c, bias=False)
        self.linear_b1 = nn.Linear(c_z, c, bias=False)
        self.linear_b2 = nn.Linear(c_z, c, bias=False)
        self.linear_g = nn.Linear(c_z, c_z, bias=False)
        self.layer_norm_out = nn.LayerNorm(c)
        self.linear_out = nn.Linear(c, c_z, bias=False)
        self.outgoing = outgoing

    def forward(self, z, single_mask):
        pair_mask = single_mask[..., :, None] * single_mask[..., None, :]
        z = self.layer_norm_z(z)
        a = torch.sigmoid(self.linear_a1(z)) * self.linear_a2(z)
        b = torch.sigmoid(self.linear_b1(z)) * self.linear_b2(z)
        g = torch.sigmoid(self.linear_g(z))

        a = a * pair_mask[..., None]
        b = b * pair_mask[..., None]

        if self.outgoing:
            o = torch.einsum('...ikc,...jkc->...ijc', a, b)
        else:
            # Note: Correct Indexing would look like this
            # o = torch.einsum('...kic,...kjc->...ijc', a, b)
            o = torch.einsum('...kjc,...kic->...ijc', a, b)
        z = g * self.linear_out(self.layer_norm_out(o))

        return z


class TriangleAttention(nn.Module):
    def __init__(self, c_z, c, n_head, starting_node=True):
        super().__init__()

        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_z, n_head*c, bias=False)
        self.linear_k = nn.Linear(c_z, n_head*c, bias=False)
        self.linear_v = nn.Linear(c_z, n_head*c, bias=False)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_z, n_head*c, bias=False)
        self.linear_out = nn.Linear(c*n_head, c_z, bias=False)
        self.N_head = n_head
        self.c = c
        self.starting_node = starting_node

        if torch.cuda.is_available():
            self.flex_attention = torch.compile(flex_attention)
        else:
            self.flex_attention = flex_attention

    def forward(self, z, single_mask):
        N_head = self.N_head
        c = self.c
        N_token = z.shape[-3]
        batch_shape = z.shape[:-3]

        z = self.layer_norm_z(z)
        q = self.linear_q(z).unflatten(-1, (N_head, c))
        k = self.linear_k(z).unflatten(-1, (N_head, c))
        v = self.linear_v(z).unflatten(-1, (N_head, c))
        g = self.linear_g(z).unflatten(-1, (N_head, c))

        bias = self.linear_b(z)

        if self.starting_node:
            bias = bias[..., None, :, :, :]
            bias += -1e9 * ~single_mask[..., None, None, :, None]
            q = torch.einsum('...ijhc->...ihjc', q)
            k = torch.einsum('...ikhc->...ihkc', k)
            v = torch.einsum('...ikhc->...ihkc', v)
            bias = torch.einsum('...ijkh->...ihjk', bias)
            
        else:
            # I'm pretty sure this would be the correct variant for indexing
            # bias = bias[..., None, :, :].transpose(-2, -4)
            bias = bias[..., None, :, :, :].transpose(-3, -4)
            bias += -1e9 * ~single_mask[..., None, None, :, None]
            # Layout conversion
            q = torch.einsum('...ijhc->...jhic', q)
            k = torch.einsum('...kjhc->...jhkc', k)
            v = torch.einsum('...kjhc->...jhkc', v)
            bias = torch.einsum('...ijkh->...jhik', bias)

        q = torch.flatten(q, end_dim=-4)
        k = torch.flatten(k, end_dim=-4)
        bias = torch.flatten(bias, end_dim=-4)

        def bias_score_mod(score, b, h, q_idx, kv_idx):
            # Broadcasting of bias index over missing token dimension
            b = b // N_token
            return score + bias[b, h, q_idx, kv_idx]

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        o = self.flex_attention(q, k, v, score_mod=bias_score_mod)

        o = o.reshape(batch_shape + (N_token, N_head, N_token, c))

        if self.starting_node:
            o = torch.einsum('...ihjc->...ijhc', o)
        else:
            o = torch.einsum('...jhic->...ijhc', o)

        o = torch.sigmoid(g) * o
        o = o.flatten(-2)
        z = self.linear_out(o)

        return z


class SharedDropout(nn.Module):
    def __init__(self, p, shared_dim):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x):
        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)
        return x*mask


class DropoutRowwise(SharedDropout):
    def __init__(self, p):
        super().__init__(p, shared_dim=-2)


class DropoutColumnwise(SharedDropout):
    def __init__(self, p):
        super().__init__(p, shared_dim=-3)


class PairStack(nn.Module):
    def __init__(self, c, n_head, n_transition, p_dropout):
        super().__init__()
        c_att = c//n_head
        self.dropout_rowwise = DropoutRowwise(p_dropout)
        self.dropout_columnwise = DropoutColumnwise(p_dropout)
        self.triangle_mult_outgoing = TriangleMultiplication(
            c_z=c, c=c, outgoing=True)
        self.triangle_mult_incoming = TriangleMultiplication(
            c_z=c, c=c, outgoing=False)
        self.triangle_att_starting = TriangleAttention(
            c_z=c, c=c_att, n_head=n_head, starting_node=True)
        self.triangle_att_ending = TriangleAttention(
            c_z=c, c=c_att, n_head=n_head, starting_node=False)
        self.transition = Transition(c, n=n_transition)

    def forward(self, z, single_mask):
        z += self.dropout_rowwise(self.triangle_mult_outgoing(z, single_mask))
        z += self.dropout_rowwise(self.triangle_mult_incoming(z, single_mask))
        z += self.dropout_rowwise(self.triangle_att_starting(z, single_mask))
        z += self.dropout_columnwise(self.triangle_att_ending(z, single_mask))
        z += self.transition(z)
        return z


class MSAModuleBlock(nn.Module):
    def __init__(self, c_m, c_z, config: MSAModuleConfig):
        super().__init__()
        self.dropout_rowwise = DropoutRowwise(config.p_dropout)
        self.opm = OuterProductMean(c_m, c_z, config.c_opm)
        self.msa_pair_weighted = MSAPairWeightedAveraging(c_m, c_z, config.c_msa_ave, config.n_head_msa_ave)
        self.transition = Transition(c_m, config.n_transition)
        self.core = PairStack(c_z, p_dropout=config.p_dropout_pairstack, n_transition=config.n_transition_pairstack, n_head=config.n_head_pairstack)

    def forward(self, m, z, msa_mask, single_mask):
        z += self.opm(m, msa_mask)
        m += self.dropout_rowwise(self.msa_pair_weighted(m, z, single_mask))
        m += self.transition(m)

        z = self.core(z, single_mask)
        return m, z


class MSAModule(nn.Module):
    def __init__(self, c_m, c_z, msa_feat_dim, c_s_input, config: MSAModuleConfig):
        super().__init__()
        self.linear_m = nn.Linear(msa_feat_dim, c_m, bias=False)
        self.linear_s = nn.Linear(c_s_input, c_m, bias=False)
        self.blocks = nn.ModuleList(
            [MSAModuleBlock(c_m, c_z, config) for _ in range(config.n_blocks)])

    def forward(self, batch: Batch, s_input, z):
        msa_feat = batch.msa_features.msa_feat
        msa_mask = batch.msa_features.msa_mask
        single_mask = batch.token_features.mask
        m = self.linear_m(msa_feat)
        m += self.linear_s(s_input)

        for block in self.blocks:
            m, z = block(m, z, msa_mask, single_mask)
        return z


class PairFormerBlock(nn.Module):
    def __init__(self, c_s, c_z, config: PairformerConfig):
        super().__init__()
        self.core = PairStack(c_z, n_head=config.n_head_pairstack, n_transition=config.n_transition_pairstack, p_dropout=config.p_dropout_pairstack)
        self.att_pair_bias = AttentionPairBias(c_s, c_z, n_head=config.n_head_att_pair_bias)
        self.single_transition = Transition(c_s, n=config.n_transition)

    def forward(self, s, z, token_features: TokenFeatures):
        single_mask = token_features.mask
        block_mask = token_features.block_mask
        z = self.core(z, single_mask)
        s += self.att_pair_bias(s, z, block_mask)
        s += self.single_transition(s)
        return s, z


class PairFormer(nn.Module):
    def __init__(self, c_s, c_z, config: PairformerConfig):
        super().__init__()
        self.blocks = nn.ModuleList([PairFormerBlock(c_s, c_z, config)
                                    for _ in range(config.n_blocks)])

    def forward(self, s, z, token_features: TokenFeatures):
        for block in tqdm.tqdm(self.blocks):
            s, z = block(s, z, token_features)
        return s, z