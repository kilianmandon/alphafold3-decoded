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
        s_input, s_trunk, z_trunk, rel_feat = self.evoformer(batch)

        x_flat = self.diffusion_sampler(self.diffusion_module,
                            s_input, s_trunk, z_trunk, rel_feat, batch)


        return x_flat

    def regional_compile(self, fullgraph=True):
        # Evoformer modules
        # # self.evoformer.input_embedder.compile(fullgraph=fullgraph)
        # for block in self.evoformer.template_embedder.pair_stack:
        #     block.compile(fullgraph=fullgraph)

        for block in self.evoformer.msa_module.blocks:
            block.compile(fullgraph=fullgraph)

        for block in self.evoformer.pairformer.blocks:
            block.compile(fullgraph=fullgraph)

        # Diffusion modules
        self.diffusion_module.diffusion_conditioning.compile(fullgraph=fullgraph)
        # self.diffusion_module.atom_att_enc.atom_transformer.compile(fullgraph=fullgraph)
        # self.diffusion_module.atom_att_enc.compile(fullgraph=fullgraph)

        for att_pair_bias_block in self.diffusion_module.diffusion_transformer.att_pair_bias:
            att_pair_bias_block.compile(fullgraph=fullgraph)

        for cond_trans_block in self.diffusion_module.diffusion_transformer.cond_trans:
            cond_trans_block.compile(fullgraph=fullgraph)
        ...




