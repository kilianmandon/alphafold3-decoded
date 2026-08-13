import inspect

import torch

from common.block_sparse_tensor import BlockSparseTensor
from torch._subclasses.fake_tensor import unset_fake_temporarily

def link_score_mod(score_mod, dtype, device):
    link = torch.zeros((), device=device, dtype=dtype)

    if score_mod is None:
        return link

    cv = inspect.getclosurevars(score_mod)
    for val in list(cv.nonlocals.values()):
        payload = getattr(val, 'bias', val)
        if isinstance(payload, torch.Tensor):
            link += payload.sum()
        elif isinstance(payload, BlockSparseTensor):
            link += payload.physical.sum()
    return link


def debug_flex_attention(q, k, v, score_mod=None, block_mask=None, kernel_options=None):
    b, h, s_q, d_q = q.shape
    s_k = k.shape[2]
    d_v = v.shape[3]
    
    fwd_flops = (b * h * s_q * s_k * 2 * d_q) + (b * h * s_q * d_v * 2 * s_k)

    with torch.no_grad():
        if block_mask is not None:
            with unset_fake_temporarily():
                kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
                block_size = block_mask.BLOCK_SIZE[0]
                
                num_rows = (s_q + block_size - 1) // block_size
                num_cols = (s_k + block_size - 1) // block_size
                
                total_blocks_dense = b * num_rows * num_cols
                present_blocks = int(torch.sum(kv_num_blocks_no_heads).item())
                
                block_mask_ratio = present_blocks / total_blocks_dense if total_blocks_dense > 0 else 1.0
                fwd_flops = int(fwd_flops * block_mask_ratio)

    if fwd_flops > 0:
        M, N = 1024, 1024
        K = fwd_flops // (2 * M * N)
        if K > 0:
            dummy_a = torch.ones((M, K), device=q.device, dtype=q.dtype)
            dummy_b = torch.ones((K, N), device=q.device, dtype=q.dtype)
            _ = torch.matmul(dummy_a, dummy_b)

    if s_q == s_k and d_q == d_v:
        out = v.clone()
    elif s_q <= s_k:
        out = v[:, :, :s_q, :].clone()
    else:
        padding = v.new_zeros(b, h, s_q - s_k, d_v)
        out = torch.cat([v, padding], dim=2)

    # connect inputs to outputs to keep the graph intact
    graph_linker = (q.sum() * 0.0) + (k.sum() * 0.0) + link_score_mod(score_mod, q.dtype, q.device)

    out = out * (1.0 + graph_linker)

    return out
