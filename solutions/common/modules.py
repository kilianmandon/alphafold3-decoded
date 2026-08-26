from torch.nn.attention.flex_attention import flex_attention
import common.utils as utils
from common.utils import activation_checkpointing
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from common.block_sparse_tensor import ExtendedBlockMask

class AdaptiveLayerNorm(nn.Module):
    """
    Implements "Algorithm 26: Adaptive LayerNorm" from the paper.
    """
    def __init__(self, c_a, c_s):
        super().__init__()
        """
        TODO: Initialize the required modules:
        layer_norm: LayerNorm with elementwise_affine=False, line 1 in the algorithm
        single_cond_layer_norm: LayerNorm with bias=False, line 2
        single_cond_scale: Linear, first Linear in line 3 in the algorithm
        single_cond_bias: Linear, second Linear in line 3 in the algorithm
        """

        self.layer_norm = nn.LayerNorm(c_a, elementwise_affine=False)
        self.single_cond_layer_norm = nn.LayerNorm(c_s, bias=False)
        self.single_cond_scale = nn.Linear(c_s, c_a)
        self.single_cond_bias = nn.Linear(c_s, c_a, bias=False)

    def forward(self, x, single_cond):
        out = None
        """
        TODO: Implement the forward pass of Algorithm 6.
        Variable naming: x is a in the paper, single_cond is s in the paper.
        """

        x = self.layer_norm(x)
        single_cond = self.single_cond_layer_norm(single_cond)
        single_scale = self.single_cond_scale(single_cond)
        single_bias = self.single_cond_bias(single_cond)
        out = F.sigmoid(single_scale) * x + single_bias

        """ End of your code """

        return out


class ConditionedTransitionBlock(nn.Module):
    """
    Implements "Algorithm 25: Conditioned Transition Block" from the paper.
    """

    def __init__(self, c_a, c_s, n=2):
        super().__init__()

        """
        TODO: Initialize the required modules:
        adaptive_layernorm: AdaptiveLayerNorm, line 1 in the algorithm
        linear_a1, linear_a2: Linear, line 2
        linear_transition: Linear, second Linear in line 3
        linear_cond: Linear, first Linear in line 3 (init with weight=0, bias=-2)

        You can do the special initialization with
        init.constant_(module_name.weight, 0) and the same for bias
        """
        self.adaptive_layernorm = AdaptiveLayerNorm(c_a, c_s)
        self.linear_a1 = nn.Linear(c_a, n*c_a, bias=False)
        self.linear_a2 = nn.Linear(c_a, n*c_a, bias=False)

        self.linear_transition = nn.Linear(n*c_a, c_a, bias=False)
        self.linear_cond = nn.Linear(c_s, c_a)
        init.constant_(self.linear_cond.weight, 0)
        init.constant_(self.linear_cond.bias, -2)

        """ End of your code """
    
    @activation_checkpointing
    def forward(self, a, s):
        out = None
        """
        TODO: Implement the forward pass of Algorithm 25.
        """
        a = self.adaptive_layernorm(a, s)
        b = F.silu(self.linear_a1(a)) * self.linear_a2(a)

        out = F.sigmoid(self.linear_cond(s)) * self.linear_transition(b)

        """ End of your code """

        return out

class Transition(nn.Module):
    """
    Implements "Algorithm 11: Transition layer" from the paper
    """
    def __init__(self, c, n):
        super().__init__()
        """
        TODO: Initialize the required modules:
        layer_norm: LayerNorm, line 1 in the algorithm
        linear_a, linear_b: Linear, lines 3 and 4
        linear_out: Linear, line 4
        """
        self.layer_norm = nn.LayerNorm(c)
        self.linear_a = nn.Linear(c, n*c, bias=False)
        self.linear_b = nn.Linear(c, n*c, bias=False)
        self.linear_out = nn.Linear(n*c, c, bias=False)
        """ End of your code """

    @activation_checkpointing(checkpoint_by_default=False)
    def forward(self, x):
        out = None
        """
        TODO: Implement the forward pass of Algorithm 11. The swish function in the paper is called F.silu in torch, and it computes silu(x) = x * sigmoid(x)
        """
        x = self.layer_norm(x)
        a = self.linear_a(x)
        b = self.linear_b(x)
        out = self.linear_out(F.silu(a) * b)

        """ End of your code """
        
        return out


class AttentionPairBias(nn.Module):
    """
    Implements "Algorithm 24: DiffusionAttention with pair bias and mask" from the paper. 

    AttentionPairBias is used at three points in AF3 with different configuration:
      - Through the DiffusionTransformer in AtomAttentionEncoder (InputEmbedder and DiffusionModule) and 
        AtomAttentionDecoder (DiffusionModule)
      - Through the PairFormerBlock (within the Evoformer)
      - Through the DiffusionTransformer in the DiffusionModule

    Special parameters:
      c_s: Dimension of the single conditioning. If None, no conditioning is used. Conditioning is used in all
        parts except within the Pairformer.
      biased_layer_norm_z: Whether or not layer_norm_z should include a bias. This should be False except 
        within the Pairformer.
      atom_level: Whether attention operates on atom-level features (e.g. n_tokens*24) or token-level
        features (n_tokens). This is True in the AtomAttentionEncoder and AtomAttentionDecoder and has two effects:
        - Within the atom-level AttentionPairBias, AF3 actually uses separate AdaLN modules to operate on the 
          query and key embeddings (this is not noted in the paper).
        - We need to adjust how we add the bias to the attention logits in atom-level AttentionPairBias,
          as it is a block-sparse tensor instead of a regular tensor in that case.
    """
    def __init__(self, c_a, c_z, n_head, c_s=None, biased_layer_norm_z=True, atom_level=False):
        super().__init__()
        self.n_head = n_head
        c = c_a // n_head
        self.c = c
        self.use_conditioning = c_s is not None
        self.atom_level = atom_level

        """
        TODO: Initialize the required modules:
        General:
          linear_q, linear_k, linear_v: Linear, lines 6 and 7 in the algorithm. Note that we are creating the
            embeddings for the attention heads jointly, so the output dimension should be c*n_head. We will unflatten them afterward.
          layer_norm_z: LayerNorm, line 8, should respect whether biased_layer_norm_z=True
          linear_b, linear_g: Linear, lines 8 and 9
        Attention:
          Set self.flex_attention either to flex_attention (if cuda is not available) or 
            torch.compile(flex_attention) (if cuda is available)
        If using conditioning, e.g. if use_conditioning is True:
          linear_out_adaptive: Linear, line 13. Should be initialized with bias=-2 and weight=0
          if atom_level: layer_norm_q, layer_norm_k: AdaptiveLayerNorm (line 2 in the Algorithm)
          else: layer_norm_a: AdaptiveLayerNorm (line 2 in the Algorithm)
          Note that the paper does not mention this split of layer_norm, but it is done in the implementation
            within AtomAttentionEncoder.
        If not, e.g. only if use_conditioning is False:
          layer_norm_a: LayerNorm, line 4
        """

        if self.use_conditioning:
            if self.atom_level:
                self.layer_norm_q = AdaptiveLayerNorm(c_a, c_s)
                self.layer_norm_k = AdaptiveLayerNorm(c_a, c_s)
            else:
                self.layer_norm_a = AdaptiveLayerNorm(c_a, c_s)

            # Should be initialized with bias=-2
            self.linear_out_adaptive = nn.Linear(c_s, c_a)
            init.constant_(self.linear_out_adaptive.weight, 0)
            init.constant_(self.linear_out_adaptive.bias, -2)
        else:
            self.layer_norm_a = nn.LayerNorm(c_a)

        self.linear_q = nn.Linear(c_a, c*n_head)
        self.linear_k = nn.Linear(c_a, c*n_head, bias=False)
        self.linear_v = nn.Linear(c_a, c*n_head, bias=False)
        self.layer_norm_z = nn.LayerNorm(c_z, bias=biased_layer_norm_z)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_a, c*n_head, bias=False)
        self.linear_out = nn.Linear(c*n_head, c_a, bias=False)

        if torch.cuda.is_available():
            self.flex_attention = torch.compile(flex_attention, dynamic=True)
        else:
            self.flex_attention = flex_attention

    """ End of your code """

    def prepare_qkvbg(self, a, z, s=None):
        """
        Implements Lines 1-8 from Algorithm 24 (AttentionPairBias)  from the paper,
        e.g. prepares the query, key, value, bias, and gate embeddings.
        """
        n_head = self.n_head
        c = self.c
        batch_shape = a.shape[:-2]

        q, k, v, bias, g = None, None, None, None, None

        """
        TODO: Implement the following steps to create the embeddings:
        - Create the pre-query and pre-key embeddings a_q and a_k using either layer_norm_a for both (if 
            non-adaptive, e.g. s is None, or if split_ada_qk is False) or with layer_norm_q and layer_norm_k (if adaptive
            and split_ada_qk are True)
        - Use the linear layers to compute the logits for q, k, v, and g. q and g use a_q (pre-query embeddings),
            k and v use a_k (pre-key embeddings). Unflatten their channels (except for g) from n_head*c to (n_head, c).
        - Apply a sigmoid to g
        - q, k, and v, have shape (**batch_shape, n_tokens, n_heads, c). Permute their axes to match 
            shape (**batch_shape, n_heads, n_tokens, c).
        - Create the bias (shape (**batch_shape, n_tokens, n_tokens, n_heads)) from the pair representation 
            (shape (**batch_shape, n_tokens, n_tokens, n_heads)). If the pair representation is 
            block-sparse (e.g. if atom_level is True), you need to apply the linear layer and layernorm 
            through z.map(layer) instead of layer(z), since linear layers don't know how to 
            handle our block-sparse tensor class.
            You *do not* need to add the mask (the betas in the paper) to the bias, as we mask explicitly 
            by passing a block_mask to flex_attention.
        - flex_attention expects the inputs to have the fixed shape (batch_dim, n_heads, n_tokens, c). You can use
            utils.unify_batch_dimension on q, k, v, and bias, to flatten the batch_shape accordingly
        - Important! q, k, and v, need to be contiguous, otherwise you might get confusing error messages from
            flex_attention. Make them so using e.g. q = q.contiguous().
        """

        if self.use_conditioning:
            if self.atom_level:
                a_q = self.layer_norm_q(a, s)
                a_k = self.layer_norm_k(a, s)
            else:   
                a_q = self.layer_norm_a(a, s)
                a_k = a_q
        else:
            a_q = self.layer_norm_a(a)
            a_k = a_q

        g = torch.sigmoid(self.linear_g(a_q))
        q = self.linear_q(a_q).unflatten(-1, (n_head, c))
        k = self.linear_k(a_k).unflatten(-1, (n_head, c))
        v = self.linear_v(a_k).unflatten(-1, (n_head, c))

        if self.atom_level:
            bias = z.map(self.layer_norm_z).map(self.linear_b)
        else:
            bias = self.linear_b(self.layer_norm_z(z))

        q = torch.einsum('...ihc->...hic', q)
        k = torch.einsum('...jhc->...hjc', k)
        v = torch.einsum('...jhc->...hjc', v)

        q = utils.unify_batch_dimension(q, batch_shape)
        k = utils.unify_batch_dimension(k, batch_shape)
        v = utils.unify_batch_dimension(v, batch_shape)
        bias = utils.unify_batch_dimension(bias, batch_shape)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

        """ End of your code """

        return q, k, v, bias, g


    @activation_checkpointing
    def forward(self, a, z, extended_block_mask: ExtendedBlockMask, s=None):
        batch_shape = a.shape[:-2]
        n_head = self.n_head
        n_token = a.shape[-2]
        c = self.c

        out = None
        block_mask = extended_block_mask.block_mask if extended_block_mask is not None else None

        q, k, v, bias, g = self.prepare_qkvbg(a, z, s)

        """
        TODO: Implement lines 10-14 of Algorithm 24 from the paper. 
 
        - Apply flex_attention. You will need a "def bias_score_mod(score, b, h, q_idx, kv_idx): " score_mod function
          that applies the bias to the score. How that function looks depends on whether we have atom_level=True
          (e.g. working with a block-sparse bias) or not (e.g. bias is a regular tensor). You can write two different
          functions like
          if self.atom_level:
            def bias_score_mod(...):
              ...
          else:
            def bias_score_mod(...):
              ...

          In case of atom_level=False, you can index directly into bias (which has shape (batch_dim, n_tokens, 
          n_tokens, n_heads)). Otherwise, you need to get the row and column index of the necessary block 
          (e.g. q_idx//block_size, kv_idx//block_size), use that and the batch index to look up the block index
          through bias.forward_index, and then use the block index and the within-block indices (e.g. 
          q_idx%block_size, kv_idx%block_size) to get the bias value. 
          After you build bias_score_mod, you can just call self.flex_attention using q, k, v, block_mask, and 
          the score_mod. You might also need kernel_options={ 'BLOCK_M': 32, 'BLOCK_N': 32 }, otherwise 
            flex_attention will fail on older graphics cards.
        - Finalize: Unflatten the attention result to get the batch_shape back, permute the heads to the back, 
            e.g. (**batch_shape, n_heads, n_tokens, c) -> (**batch_shape, n_tokens, n_heads, c) and collapse the
            n_heads with the c dimension. As described in the paper, apply the gating, the linear out layer,
            and if using conditioning, the conditioned output projection.
        """

        if self.atom_level:
            block_size = bias.block_size
            forward_index = bias.forward_index
            physical = bias.physical

            def bias_score_mod(score, b, h, q_idx, kv_idx):
                bias_val = physical[forward_index[b, q_idx//block_size, kv_idx//block_size], q_idx%block_size, kv_idx%block_size, h]
                return score + bias_val
        else:
            def bias_score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[b, q_idx, kv_idx, h]

        o = self.flex_attention(q, k, v, score_mod=bias_score_mod, block_mask=block_mask, kernel_options={ 'BLOCK_M': 32, 'BLOCK_N': 32 })

        o = o.reshape(batch_shape + (n_head, n_token, c))
        o = torch.einsum('...hjc->...jhc', o)
        o = o.flatten(-2)

        o = g * o
        out = self.linear_out(o)

        if self.use_conditioning:
            out = torch.sigmoid(self.linear_out_adaptive(s)) * out

        """ End of your code """

        return out



class DiffusionTransformer(nn.Module):
    """
    Implements "Algorithm 23: Diffusion Transformer" from the paper.
    """
    def __init__(self, c_a, c_z, n_head, c_s, n_blocks, atom_level=False):
        """
        Initializes the required modules for Algorithm 23.
        Args:
            c_a (int): Number of channels for the token activations.
            c_z (int): Number of channels for the pair condition.
            n_head (int): Number of heads in AttentionPairBias.
            c_s (int): Number of channels for the single condition.
            n_blocks (int): Nuber of blocks for AttentionPairBias and ConditionedTransitionBlock.
            atom_level (bool): Whether or not the transformer operates on atom-level representations. 
              This affects computations within AttentionPairBias. Defaults to False.
        """
        super().__init__()
        """
        TODO: Initialize the required modules:
        att_pair_bias: nn.ModuleList of AttentionPairBias blocks, using c_s, biased_layer_norm_z=False, atom_level=atom_level
        cond_trans: nn.ModuleList of ConditionedTransitionBlock blocks.
        """
        self.att_pair_bias = nn.ModuleList([AttentionPairBias(c_a, c_z, n_head, c_s, biased_layer_norm_z=False, atom_level=atom_level) for _ in range(n_blocks)])
        self.cond_trans = nn.ModuleList([ConditionedTransitionBlock(c_a, c_s) for _ in range(n_blocks)])

        """ End of your code """


    def forward(self, a, s, z, extended_block_mask: ExtendedBlockMask):
        out = None

        """
        TODO: Implement the forward pass of Algorithm 23.
        """

        for att_pair_block, cond_trans_block in zip(self.att_pair_bias, self.cond_trans):
            a = a + att_pair_block(a, z, extended_block_mask, s=s)
            a = a + cond_trans_block(a, s)

        out = a

        """ End of your code """

        return out