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

        # Replace 'pass' with your code
        pass

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

        # Replace 'pass' with your code
        pass

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

        # Replace 'pass' with your code
        pass

        """ End of your code """

        return s_input, s_init, z_init, rel_feat
