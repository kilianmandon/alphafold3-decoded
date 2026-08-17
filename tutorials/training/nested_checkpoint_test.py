from torch import nn
import torch
from torch_snapkit import memory_snapshot
from common.modules import Transition
from common.utils import activation_checkpointing


class WrapperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.transitions = nn.ModuleList([Transition(32, 4) for _ in range(2)])
        self.linears = nn.ModuleList([nn.Linear(32, 512), nn.Linear(512, 32)])

    @activation_checkpointing
    def forward(self, x):
        for transition in self.transitions:
            x = x + transition(x, activation_checkpointing=False)
        for linear in self.linears:
            x = linear(x)
        return x

class WrapperWrapperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([WrapperModule() for _ in range(2)])

    def forward(self, x):
        for mod in self.mods:
            x = x + mod(x)
        return x

def main():
    with torch.autograd.detect_anomaly(), memory_snapshot('debug_outer_checkpointing', share=True, share_code='kilisaf3_secret', log_shapes=True):
        x = torch.randn((100_000, 32), device='cuda')
        mod = WrapperWrapperModule().to(device='cuda')
        out = mod(x)
        out.mean().backward()


if __name__=='__main__':
    main()
