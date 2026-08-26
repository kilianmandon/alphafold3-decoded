import torch
from config import InputEmbeddingConfig
from feature_extraction.feature_extraction import Batch
from torch import nn
import torch.nn.functional as F

from input_embedding.atom_attention import AtomAttentionEncoder


class InputEmbedder(nn.Module):
    """
    Implements the input embedding module of AlphaFold 3, which corresponds to the lines 1 to 5 of 
    Algorithm 1 in the paper. Algorithm 3 from the paper is also implemented in this module, it is used to compute
    the initial pair embedding z_init.
    """
    def __init__(self, c_s, c_z, c_s_input, rel_feat_dim, config: InputEmbeddingConfig):
        super().__init__()

        self.r_max = config.r_max
        self.s_max = config.s_max

        """
        TODO: Initialize the layers you will need for input embedding. This includes:
        position_activations: LinearNoBias for the relative encoding in Algorithm 3
        single_embedding: LinearNoBias for the single embedding of s_input, Algorithm 1
        left_single and right_single: LinearNoBias layers for the pair embedding of s_input, Algorithm 1
        bond_embedding: LinearNoBias layer for the bond matrix embedding in Algorithm 1
        atom_cross_att: AtomAttentionEncoder, uses c_s, c_z, and config.atom_attention_config. In the InputEmbedder,
          we don't use the trunk (this means we don't recycle features from previous blocks, because there 
          are no previous blocks. This is only used in the Diffusion Module)

        """

        self.position_activations = nn.Linear(rel_feat_dim, c_z, bias=False)
        self.single_embedding = nn.Linear(c_s_input, c_s, bias=False)
        self.left_single = nn.Linear(c_s_input, c_z, bias=False)
        self.right_single = nn.Linear(c_s_input, c_z, bias=False)
        self.bond_embedding = nn.Linear(1, c_z, bias=False)
        self.atom_cross_att = AtomAttentionEncoder(c_s, c_z, config.atom_attention_config, use_trunk=False)

        """ End of your code """


    def relative_encoding(self, batch: Batch):
        """
        Implements "Algorithm 3: Relative position encoding" from the paper.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                rel_enc: Tensor of shape (**batch_shape, n_tokens, n_tokens, c_z) representing the relative encoding, 
                    the return value of the algorithm as stated in the paper.
                rel_feat: Tensor of shape (**batch_shape, n_tokens, n_tokens, rel_feat_dim), the relative features 
                    before the final linear layer. This is used in the diffusion module in AlphaFold 3.
        """
        token_features = batch.token_features
        token_index = token_features.token_index
        residue_index = token_features.residue_index
        asym_id = token_features.asym_id
        entity_id = token_features.entity_id
        sym_id = token_features.sym_id
        r_max = self.r_max
        s_max = self.s_max

        # This is p before the linear layer in the paper, e.g. concat(a_rel_pos, ...)
        rel_feat = None
        # This is p in the paper
        rel_enc = None

        """
        TODO: Implement Algorithm 3 from the paper. Look at the variables defined above this comment for the values 
        you will need.
        """

        left_token_index, right_token_index = token_index[...,
                                                          None], token_index[..., None, :]
        left_residue_index, right_residue_index = residue_index[...,
                                                                None], residue_index[..., None, :]
        left_asym_id, right_asym_id = asym_id[..., None], asym_id[..., None, :]
        left_entity_id, right_entity_id = entity_id[...,
                                                    None], entity_id[..., None, :]
        left_sym_id, right_sym_id = sym_id[..., None], sym_id[..., None, :]

        same_chain = left_asym_id == right_asym_id
        same_residue = left_residue_index == right_residue_index
        same_entity = left_entity_id == right_entity_id

        residue_dist = torch.clip(
            left_residue_index-right_residue_index+r_max, 0, 2*r_max)
        residue_dist[~same_chain] = 2*r_max+1

        token_dist = torch.clip(
            left_token_index-right_token_index + r_max, 0, 2*r_max)
        token_dist[~(same_chain & same_residue)] = 2*r_max+1

        chain_dist = torch.clip(left_sym_id-right_sym_id+s_max, 0, 2*s_max)
        chain_dist[~same_entity] = 2*s_max+1

        a_rel_pos = F.one_hot(residue_dist, 2*r_max+2)
        a_rel_token = F.one_hot(token_dist, 2*r_max+2)
        a_rel_chain = F.one_hot(chain_dist, 2*s_max+2)

        p = torch.cat(
            (a_rel_pos, a_rel_token, same_entity[..., None], a_rel_chain), dim=-1)
        rel_feat = p.to(dtype=torch.float32)
        rel_enc = self.position_activations(rel_feat)

        """ End of your code """

        return rel_enc, rel_feat

    def forward(self, batch):
        """
        Implements lines 1 to 5 of Algorithm 1 in the paper. 

        Returns:
            tuple[torch.Tensor]: A tuple containing:
                s_input: Tensor of shape (**batch_shape, n_tokens, c_s_input), the initial single feature before
                    the linear layer. 
                s_init: Tensor of shape (**batch_shape, n_tokens, c_s), the single embedding, after the linear layer.
                z_init: Tensor of shape (**batch_shape, n_tokens, n_tokens, c_z), the initial pair embedding.
                rel_feat: Tensor of shape (**batch_shape, n_tokens, n_tokens, rel_feat_dim), the relative features 
                    before the final linear layer. This is used in the diffusion module in AlphaFold 3.
        """

        # This is concat(f_restype, f_profile, f_deletion_mean) in Algorithm 2 in the paper.
        target_feat = batch.msa_features.target_feat
        reference_features = batch.reference_features

        s_input, s_init, z_init, rel_feat = None, None, None, None

        """
        TODO: Implement lines 1 to 5 of Algorithm 1 in the paper. This includes the steps in "Algorithm 2: InputFeatureEmbedder"
        for which you will need to call the AtomAttentionEncoder module, and the steps in "Algorithm 3: Relative position encoding" 
        which you implemented above in the relative_encoding function. Note that self.relative_encoding returns a 
        tuple (rel_enc, rel_feat). rel_enc is the return value in the paper, and rel_feat needs to be returned
        by this forward function, because it is used in the diffusion module of AlphaFold 3 (the paper does not state this explicitly).
        """

        token_act, _ = self.atom_cross_att(reference_features)
        s_input = torch.cat((target_feat, token_act), dim=-1)

        s_init = self.single_embedding(s_input)
        a = self.left_single(s_input)
        b = self.right_single(s_input)
        z_init = a[..., None, :] + b[..., None, :, :]

        rel_enc, rel_feat = self.relative_encoding(batch)
        z_init = z_init + rel_enc
        z_init = z_init + self.bond_embedding(batch.bond_matrix)

        """ End of your code """

        return s_input, s_init, z_init, rel_feat
