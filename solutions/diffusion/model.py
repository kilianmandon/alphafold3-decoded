from torch import nn
from config import Config
from evoformer.evoformer import Evoformer
from diffusion.diffusion import DiffusionModule, DiffusionSampler
from feature_extraction.feature_extraction import Batch

class Model(nn.Module):
    # Implements Algorithm 1 from the paper
    def __init__(self, config: Config):
        super().__init__()
        self.evoformer = Evoformer(config)
        self.diffusion_module = DiffusionModule(config)
        self.diffusion_sampler = DiffusionSampler(config)

    def forward(self, batch: Batch):
        s_input, s_trunk, z_trunk, rel_enc = self.evoformer(batch)

        x_flat = self.diffusion_sampler(self.diffusion_module,
                            s_input, s_trunk, z_trunk, rel_enc, batch)


        return x_flat

