from dataclasses import dataclass

from torch import nn
import torch
from torch_snapkit import memory_snapshot

from torch.utils.flop_counter import FlopCounterMode


class AttentionPairBiasSimple(nn.Module):
    def __init__(self, c_a, c_z):
        super().__init__()

        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_v = nn.Linear(c_a, c_a, bias=False)
        self.linear_b = nn.Linear(c_z, c_z, bias=False)
        self.linear_out = nn.Linear(c_a, c_a, bias=False)


    

    def forward(self, a, z):
        def inner_forward(self, a, z):
            v = self.linear_v(a)
            v = v + self.linear_b(self.layer_norm_z(z)).sum()
            v = self.linear_out(v)

            return v
        return torch.utils.checkpoint.checkpoint(inner_forward, self, a, z, use_reentrant=False)


@dataclass
class DebugExtBlockMask:
    block_mask = None

def main():
    c_a = 128
    c_z = 128
    n_seq = 384
    n_batch = 12

    test_module = nn.ModuleList([AttentionPairBiasSimple(c_a, c_z) for _ in range(12)])
    test_module.to('cuda')

    a = torch.randn((n_batch, n_seq, c_a)).to('cuda')
    z = torch.randn((n_batch, n_seq, n_seq, c_z)).to('cuda')
    with FlopCounterMode(display=True):
        for block in test_module:
            a = block(a, z)
        a.sum().backward()


if __name__=='__main__':
    # with memory_snapshot('flopcount_bug_simple', share=True, share_code='daily-secret-05', log_shapes=True):
    torch.cuda.memory._record_memory_history()
    main()
    torch.cuda.memory._dump_snapshot('test_bug_snapshot.pkl')
    torch.cuda.memory._record_memory_history(enabled=None)