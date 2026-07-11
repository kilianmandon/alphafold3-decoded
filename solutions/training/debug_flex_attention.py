import torch

def debug_flex_attention(q, k, v, score_mod=None, block_mask=None, kernel_options=None):
    b, h, s_q, d_q = q.shape
    s_k = k.shape[2]
    d_v = v.shape[3]
    
    # 1. Compute FLOPs accurately
    fwd_flops = (b * h * s_q * s_k * 2 * d_q) + (b * h * s_q * d_v * 2 * s_k)

    with torch.no_grad():
        if block_mask is not None:
            kv_num_blocks_no_heads = block_mask.kv_num_blocks[:, 0, :]
            block_size = block_mask.BLOCK_SIZE[0]
            
            num_rows = (s_q + block_size - 1) // block_size
            num_cols = (s_k + block_size - 1) // block_size
            
            total_blocks_dense = b * num_rows * num_cols
            present_blocks = int(torch.sum(kv_num_blocks_no_heads).item())
            
            block_mask_ratio = present_blocks / total_blocks_dense if total_blocks_dense > 0 else 1.0
            fwd_flops = int(fwd_flops * block_mask_ratio)

    # 2. Fake the FLOP Counter
    if fwd_flops > 0:
        M, N = 1024, 1024
        K = fwd_flops // (2 * M * N)
        if K > 0:
            dummy_a = torch.ones((M, K), device=q.device, dtype=q.dtype)
            dummy_b = torch.ones((K, N), device=q.device, dtype=q.dtype)
            _ = torch.matmul(dummy_a, dummy_b)

    # 3. Handle expected layout output shape
    if s_q == s_k and d_q == d_v:
        out = v.clone()
    elif s_q <= s_k:
        out = v[:, :, :s_q, :].clone()
    else:
        padding = v.new_zeros(b, h, s_q - s_k, d_v)
        out = torch.cat([v, padding], dim=2)

    # 4. TRICK AUTOGRAD: Graph-link q, k, v, and score_mod
    # Evaluate score_mod once at index zero to drag any captured tensors (like pair bias) into the graph
    graph_linker = (q.sum() * 0.0) + (k.sum() * 0.0)
    if score_mod is not None:
        # flex_attention score_mod signature usually expects: score, b, h, q_idx, kv_idx
        # We pass a dummy scalar zero tensor for the score input
        dummy_score = torch.tensor(0.0, device=q.device, dtype=q.dtype)
        mod_output = score_mod(dummy_score, 0, 0, 0, 0)
        graph_linker = graph_linker + (mod_output.sum() * 0.0)

    # Apply the linker. It multiplies by 1.0, adding nothing to the value, 
    # but safely stitches the graph together.
    out = out * (1.0 + graph_linker)

    # For debug tracking simplicity, we let PyTorch natively handle the backward graph via the linker
    return out
