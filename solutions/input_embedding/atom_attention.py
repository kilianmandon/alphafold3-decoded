import torch
from torch import nn
from config import AtomAttentionConfig
from feature_extraction.reference_features import ReferenceFeatures
from common.block_sparse_tensor import BlockSparseTensor
from common.modules import DiffusionTransformer

import common.utils as utils

def hotfix_mangle_layout(ref_space_uid, reference_features: ReferenceFeatures):
    """
    AlphaFold 3 does conversion of token-layout to atom-layout ("key-layout" in their code) incorrectly. 
    Because of that, the flat atom-layout is instead a flattened version of the token-layout. This function imitates
    that behavior. Note that they are using the unmasked version of ref_space_uid (which matters, because 
    the masked positions are spilling into unmakred positions due to the wrong conversion). To create this unmasked, 
    token-layout version, we take the first row in token-layout and broadcast that over the whole token-layout tensor.
    """
    ref_space_uid = reference_features.to_token_layout(ref_space_uid)
    ref_space_uid[..., :, :] = ref_space_uid[..., :, :1]
    ref_space_uid = torch.flatten(ref_space_uid, start_dim=-2)
    return ref_space_uid


class SingleAtomConditioning(nn.Module):
    """
    Implements Line 1 from "Algorithm 5: Atom attention encoder"
    """
    def __init__(self, config: AtomAttentionConfig):
        super().__init__()
        c_atom = config.c_atom
        self.atom_element_dim = config.atom_element_dim
        self.atom_chars_dim = config.atom_chars_dim
        atom_chars_full_dim = 4 * self.atom_chars_dim # for all four characters in name
        atom_element_dim = config.atom_element_dim

        """
        TODO: Initialize the following modules (we split the conat -> linear layer into separate linear layers):
        embed_ref_pos, embed_ref_mask, embed_ref_element, embed_ref_charge, embed_ref_atom_name
        """

        self.embed_ref_pos = nn.Linear(3, c_atom, bias=False)
        self.embed_ref_mask = nn.Linear(1, c_atom, bias=False)
        self.embed_ref_element = nn.Linear(atom_element_dim, c_atom, bias=False)
        self.embed_ref_charge = nn.Linear(1, c_atom, bias=False)
        self.embed_ref_atom_name = nn.Linear(atom_chars_full_dim, c_atom, bias=False)

        """ End of your code """
    
    def forward(self, reference_features: ReferenceFeatures):
        mask = reference_features.mask[..., None].to(torch.float32)
        element = reference_features.element.long()
        charge = reference_features.charge[..., None].to(torch.float32)
        name_chars = reference_features.atom_name_chars.long()
        atom_element_dim = self.atom_element_dim
        atom_chars_dim = self.atom_chars_dim

        act = None

        """
        TODO: Embed all of the features using the corresponding linear layers. elements and atom_names need to be
        one-hot encoded, and the trailing dimensions of atom_names_1h must be flattened, so that the shapes work out.
        We embed arcsinh(charge) instead of directly using charge.
        """

        elements_1h = nn.functional.one_hot(element, atom_element_dim).to(torch.float32)
        atom_names_1h = nn.functional.one_hot(name_chars, atom_chars_dim).to(torch.float32)
        atom_names_1h = atom_names_1h.flatten(start_dim=-2)

        act = self.embed_ref_pos(reference_features.positions)
        act = act + self.embed_ref_mask(mask)
        act = act + self.embed_ref_element(elements_1h)
        act = act + self.embed_ref_charge(torch.arcsinh(charge))
        act = act + self.embed_ref_atom_name(atom_names_1h)
        act = act * mask

        """ End of your code """

        return act


class AtomPairConditioning(nn.Module):
    """
    Implements lines 2 to 6 from "Algorithm 5: Atom attention encoder" in the paper.
    """

    def __init__(self, config: AtomAttentionConfig, use_block_mask_diffusion: bool):
        """
        Initializes AtomPairConditioning.
        Args:
            config (AtomAttentionConfig): Used to set the channel dimensions of the layers.
            use_block_mask_diffusion (bool): Whether the module is used in the Evoformer part or 
                the DiffusionModule part. Concretely, in the Evoformer part, inputs will be of shape
                (**batch_shape, **feat_shape) and the module should use `reference_features.block_mask`,
                while in the DiffusionModule part, inputs will be (n_diffusion_samples, **batch_shape, **feat_shape),
                and this module should use `reference_features.block_mask_diffusion`, which includes the 
                diffusion_samples dimension. 

        """
        super().__init__()
        c_atompair = config.c_atompair
        self.use_block_mask_diffusion = use_block_mask_diffusion

        """
        TODO: Initialize embed_pair_offsets (line 4 in the algorithm), embed_pair_distances (line 5 in the algorithm),
        and embed_pair_mask (line 6 in the algorithm).
        """

        self.embed_pair_offsets = nn.Linear(3, c_atompair, bias=False)
        self.embed_pair_distances = nn.Linear(1, c_atompair, bias=False)
        self.embed_pair_mask = nn.Linear(1, c_atompair, bias=False)

        """ End of your code """

    def forward(self, reference_features: ReferenceFeatures):
        ref_space_uid = reference_features.ref_space_uid
        ref_pos = reference_features.positions

        if self.use_block_mask_diffusion:
            block_mask = reference_features.block_mask_diffusion
        else:
            block_mask = reference_features.block_mask

        batch_shape = ref_space_uid.shape[:-1]

        # This is p in the paper
        pair_cond = None

        """
        TODO: Implement lines 2 to 6 from Algorithm 5 in the paper. We want all the involved pair activations 
        (that is d_lm, v_lm, and p_lm in the paper) to be of our custom type BlockSparseTensor (BST). 
        You can create the BSTs for f_l_ref_pos, f_m_ref_pos, f_l_ref_space_uid, f_m_ref_space_uid using the method
        BlockSparseTensor.broadcast_up(...). It takes a tensor v with a shape that's broadcastable to 
        (**batch_shape, n_tokens, n_tokens, c) (the last dimension is optional), and creates a block-sparse tensor
        that is equivalent to broadcasting v to that shape (only materializing the parts of it that are not masked).

        Important: f_m_ref_pos (or ref_space_right in the variable naming I use here) is constructed not from
        ref_space_uid, but from a wrong conversion atom_layout -> token_layout -/-> atom_layout. The last conversion
        step is done wrong in AlphaFold 3, as a simple reshape. To emulate this, you cann call the function
        hotfix_mangle_Layout(ref_space_uid, reference_features), to create the "wrong" version of ref_space_uid,
        to be used in ref_space_right.
        """

        # This needs to be used for the construction of ref_space_right, e.g. for f_m_ref_space_uid in Algorithm 5.
        wrong_ref_space_uid = hotfix_mangle_layout(ref_space_uid, reference_features)

        ref_space_left = BlockSparseTensor.broadcast_up(ref_space_uid[..., :, None], block_mask, batch_shape)
        ref_space_right = BlockSparseTensor.broadcast_up(wrong_ref_space_uid[..., None, :], block_mask, batch_shape)
        ref_pos_left = BlockSparseTensor.broadcast_up(ref_pos[..., :, None, :], block_mask, batch_shape)
        ref_pos_right = BlockSparseTensor.broadcast_up(ref_pos[..., None, :, :], block_mask, batch_shape)

        offsets_valid = (ref_space_left == ref_space_right).to(dtype=torch.float32)

        offsets = ref_pos_left - ref_pos_right

        pair_cond = offsets.map(self.embed_pair_offsets) * offsets_valid

        sq_dists = offsets.map(lambda x: torch.sum(x**2, dim=-1, keepdim=True))

        pair_cond = pair_cond + (1/(1+sq_dists)).map(self.embed_pair_distances) * offsets_valid
        pair_cond = pair_cond + offsets_valid.map(self.embed_pair_mask)

        """ End of your code """

        return pair_cond


class AtomAttentionEncoder(nn.Module):
    """
    Implements "Algorithm 5: Atom attention encoder" from the paper.
    """
    def __init__(self, c_s, c_z, config: AtomAttentionConfig, use_trunk=False):
        super().__init__()
        c_atom = config.c_atom
        c_atompair = config.c_atompair

        """
        TODO: Initialize all the modules for Algorithm 5:
        single_atom_conditioning: SingleAtomConditioning module, for line 1 
        atom_pair_conditioning: AtomPairConditioning module, for line 2 to 6
        single_to_pair_left and single_to_pair_right: Linear, line 13
        pair_mlp: nn.Sequential of ReLUs and Linears, line 14
        atom_transformer: DiffusionTransformer, the activation dimension is c_atom, pair conditioning dimension is
          c_atom_pair, n_head is config.n_head_atom_transformer, single conditioning dimension is c_atom, 
          n_blocks is config.n_block_atom_transformer, and atom_level is True.
        linear_out: Linear, line 16

        if use_trunk, initialize:
          trunk_layer_norm_s: LayerNorm, bias=False, line 9
          trunk_linear_s: Linear, line 9
          trunk_layer_norm_z: LayerNorm, bias=False, line 10
          trunk_linear_z: Linear, line 10
          trunk_linear_r: Linear, line 11
        """

        self.single_atom_conditioning = SingleAtomConditioning(config)
        self.atom_pair_conditioning = AtomPairConditioning(config, use_block_mask_diffusion=use_trunk)

        self.single_to_pair_left = nn.Linear(c_atom, c_atompair, bias=False)
        self.single_to_pair_right = nn.Linear(c_atom, c_atompair, bias=False)

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(c_atompair, c_atompair, bias=False)
        )

        self.atom_transformer = DiffusionTransformer(c_a=c_atom, c_z=c_atompair, n_head=config.n_head_atom_transformer, c_s=c_atom, n_blocks=config.n_block_atom_transformer, atom_level=True)
        self.linear_out = nn.Linear(c_atom, config.c_token, bias=False)

        self.use_trunk = use_trunk
        if use_trunk:
            self.trunk_layer_norm_s = nn.LayerNorm(c_s, bias=False)
            self.trunk_linear_s = nn.Linear(c_s, c_atom, bias=False)
            self.trunk_layer_norm_z = nn.LayerNorm(c_z, bias=False)
            self.trunk_linear_z = nn.Linear(c_z, c_atompair, bias=False)
            self.trunk_linear_r = nn.Linear(3, c_atom, bias=False)


    def forward(self, reference_features: ReferenceFeatures, r=None, s_trunk=None, z=None):
        ref_space_uid = reference_features.ref_space_uid
        if self.use_trunk:
            block_mask = reference_features.block_mask_diffusion
        else:
            block_mask = reference_features.block_mask

        batch_shape = ref_space_uid.shape[:-1]

        token_act = None
        single_act, single_cond, pair_cond = None, None, None

        """
        TODO: Implement Algorithm 5 from the paper.
        Variable names:
          q: single_act (the activation, in contrast to conditioning inputs for the diffusion transformer)
          c: single_cond (the single conditioning input for the diffusion transformer)
          p: pair_cond (the pair conditioning input for the diffusion transformer)
          a: token_act (the token-level activation created by the method, the main output)
        Discrepancy from the paper: Line 11 in the paper states q += LinearNoBias(r). Here, q is a copy of c before 
          the trunk-update, e.g. before c is changed in line 9. We follow the alphafold3 repository, which does
          q = c + LinearNoBias(r) instead.

        Hints: Except for the trunk update, most steps just follow the paper. Make sure to clone single_cond
        when initializing single_act so single_act is not affected if you do in-place updates on single_cond.
        For the trunk update:
          s_trunk[tok_idx(l)] in the paper can be constructed by using reference_features.to_atom_layout on s_trunk
            by setting has_atom_dimension=False (that's basically the use-case we created the method for: broadcasting
            the s_trunk so it includes the atom dimension, then doing token_layout -> atom_layout)
          z[tok_idx(l), tok_idx(m)] can be constructed in three steps:
            Compute the inverse_indices batch_idx, l_idx, m_idx using BlockSparseTensor.get_inverse_indices. 
              These satisfy, for any tensor x of shape (batch_dim, n_token, n_token, c), that 
              x[batch_idx, l_idx, m_idx] = bst_physical_storage, where bst_physical_storage is the flat block tensor 
              used by BlockSparseTensor internally.
            Map l_idx and m_idx to i_idx=token_idx(l_idx) and j_idx = token_idx(m_idx). This can be done by 
              indexing with them (and with batch_idx) into reference_features.token_index
            Construct z as a BlockSparseTensor using the constructor, with physical as z[batch_idx, i_idx, j_idx]
          Note that all indexing with batch_idx, l_idx, and m_idx, expects the data to be of shape 
          (batch_dim, **layout_and_feat_dims), not of shape (**batch_shape, **layout_and_feat_dims). You can unify
          the batch shape (e.g. flatten if there are too many batch dimensions, unsqueeze if there is none) 
          using utils.unify_batch_dimension.
        
        """

        single_cond = self.single_atom_conditioning(reference_features)
        single_act = single_cond.clone()
        pair_cond = self.atom_pair_conditioning(reference_features)

        if self.use_trunk:
            s_trunk = reference_features.to_atom_layout(s_trunk, has_atom_dimension=False)

            batch_idx, l_idx, m_idx = pair_cond.inverse_indices
            token_indices = utils.unify_batch_dimension(reference_features.token_index, batch_shape)
            z = utils.unify_batch_dimension(z, batch_shape)
            i_idx = token_indices[batch_idx, l_idx]
            j_idx = token_indices[batch_idx, m_idx]
            z = self.trunk_linear_z(self.trunk_layer_norm_z(z))
            z = pair_cond._wrap(z[batch_idx, i_idx, j_idx])

            single_cond = single_cond + self.trunk_linear_s(self.trunk_layer_norm_s(s_trunk))
            pair_cond = pair_cond + z

            # Note: The paper uses the old, non-trunk-updated value
            # for single_cond here
            single_act = single_cond + self.trunk_linear_r(r)


        row_act = self.single_to_pair_left(torch.relu(single_cond))
        row_act = BlockSparseTensor.broadcast_up(row_act[..., :, None, :], block_mask, batch_shape)
        col_act = self.single_to_pair_right(torch.relu(single_cond))
        col_act = BlockSparseTensor.broadcast_up(col_act[..., None, :, :], block_mask, batch_shape)

        pair_cond = pair_cond + row_act + col_act
        pair_cond = pair_cond + pair_cond.map(self.pair_mlp)

        single_act = self.atom_transformer(
            single_act,
            single_cond,
            pair_cond,
            block_mask
        )


        token_act = reference_features.to_token_layout(single_act)
        token_act = torch.relu(self.linear_out(token_act))

        token_act = utils.masked_mean(token_act, reference_features.token_layout_ref_mask[..., None], axis=-2)

        """ End of your code """

        skip = (single_act, single_cond, pair_cond)

        return token_act, skip



class AtomAttentionDecoder(nn.Module):
    """
    Implements "Algorithm 6: Atom attention decoder" from the paper.
    """
    def __init__(self, config: AtomAttentionConfig):
        super().__init__()
        c_atom = config.c_atom
        c_atomapair = config.c_atompair

        """
        TODO: Initialize the necessary modules:
        linear_a: Linear layer for line 1 in the algorithm
        atom_transformer: DiffusionTransformer for line 2 in the algorithm. The activation dimension is c_atom,
          the pair condition dimension is c_atompair, the single condition dimension is c_atom, the number of blocks
          is config.n_block_atom_transformer, atom_level is True.
        layer_norm_q: LayerNorm (no bias) for line 3 in the algorithm.
        linear_out: Linear layer for line 3 in the algorithm.
        """

        self.linear_a = nn.Linear(config.c_token, c_atom, bias=False)
        self.atom_transformer = DiffusionTransformer(c_a=c_atom, c_z=c_atomapair, n_head=config.n_head_atom_transformer, c_s=c_atom, n_blocks=config.n_block_atom_transformer, atom_level=True)
        self.layer_norm_q = nn.LayerNorm(c_atom, bias=False)
        self.linear_out = nn.Linear(c_atom, 3, bias=False)

        """ End of your code """

    def forward(self, a, q_skip, c_skip, p_skip, reference_features: ReferenceFeatures):
        """
        Implements Algorithm 6 from the paper.

        Args:
            a (torch.Tensor): Token activation of shape (**batch_shape, n_token, c_token).
            q_skip (torch.Tensor): Single activation from a skip connection, of shape (**batch_shape, n_atoms, c_atom).
            c_skip (torch.Tensor): Single condition from a skip connection, of shape (**batch_shape, n_atoms, c_atom).
            p_skip (BlockSparseTensor): Pair condition from a skip connection, of logical 
              shape (**batch_shape, n_atoms, n_atoms, c_atompair).
            reference_features (ReferenceFeatures): Reference features from the feature extraction stage. Used for 
              the required atom-layout -> token-layout conversion.

        Returns:
            torch.Tensor: Position update r of shape (**batch_shape, n_atoms, 3).
        """

        r = None
        # AtomAttentionDecoder is only used in the diffusion module
        block_mask = reference_features.block_mask_diffusion

        """
        TODO: Implement Algorithm 6 from the paper. For the conversion of a to atom-layout, you can use 
        reference_features.to_atom_layout. 
        """

        a = self.linear_a(a)
        a_q = reference_features.to_atom_layout(a, has_atom_dimension = False)
        q = a_q + q_skip
        q = self.atom_transformer(q, c_skip, p_skip, block_mask)
        r = self.linear_out(self.layer_norm_q(q))

        """ End of your code """

        return r
