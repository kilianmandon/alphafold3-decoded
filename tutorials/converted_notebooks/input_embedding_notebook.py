#!/usr/bin/env python
# coding: utf-8

# # Chapter 2: Input Embedding
# 
# This Notebook will walk you through implementing the input embedding procedure in AlphaFold 3. This corresponds to the lines 1 to 5 from Algorithm 1 in the AlphaFold 3 paper. Most of the tasks involve implementing algorithms described in the Supplementary Information of the AlphaFold 3 paper, which you can find [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf).
# 
# We will first implement the full input embedding pipeline except of the Atom Attention Encoder. That is pretty straight-forward.
# 
# The Atom Attention Encoder uses the full Diffusion Transformer (that's also used in the Diffusion Module, so we won't need to do much work on that in the Diffusion Chapter), and it relies on block-sparse attention and block-sparse tensors. So, after the input embedding pipeline without Atom Attention, we will implement all the core modules in `common/modules.py`, the block-sparse tensor implementation in `common/block_sparse_tensor.py` (that's a bit tricky and optional, you can also copy the code over from the solutions folder), and finally the Atom Attention Encoder. 
# 
# Run the following two cells to set up the environment. Check that the PythonPath in the first cell correctly includes the `tutorials` folder.

# In[ ]:


import sys
import os

os.getcwd(), sys.path

# Manually add the folder:
# sys.path.append('path/to/alphafold3-decoded/tutorials)


# If you have a CUDA-compatible GPU available, we will do all tests on that. If not, the tests should be able to verify most of your code anyway (although you might have to fix some 'tensor-not-moved-to-device' errors later on when switching to a GPU).

# In[ ]:


import torch

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA GPU found, will be used for all tests.')
else:
    device = 'cpu'
    print('No GPU found, tests will be performed on CPU.')


# In[ ]:


import os

# Set so that Atomworks does not raise a warning, we don't need to actually download the mirrors for this notebook.
os.environ["PDB_MIRROR_PATH"] = ""
os.environ["CCD_MIRROR_PATH"] = ""

from feature_extraction.feature_extraction import Batch, tree_map
from feature_extraction.token_features import TokenFeatures
from feature_extraction.msa_features import MSAFeatures
from feature_extraction.reference_features import ReferenceFeatures

import numpy as np
import tensortrace as ttr
from torch import nn
from config import load_config

# get_ipython().run_line_magic('load_ext', 'autoreload')  # removed by prepare_tutorials.py
# get_ipython().run_line_magic('autoreload', '2')  # removed by prepare_tutorials.py

ttr.TensorTrace('data/tensortraces/input_embedding', mode='read', framework='pytorch').start_trace()

ttr.current_trace().atol = 1e-6
ttr.current_trace().rtol = 1e-5

torch.set_grad_enabled(False)

small_config = load_config('data/configs/af3_small_config.yaml')

from feature_extraction.feature_extraction import custom_af3_pipeline
from common.utils import load_alphafold_input

af3_pipeline = custom_af3_pipeline(small_config)
base_batches = []

for inp in ['fold_input_lysozyme', 'fold_input_multimer', 'fold_input_protein_dna_ion', 'fold_input_protein_rna_ion']:
    batch = af3_pipeline(load_alphafold_input(f'data/fold_inputs/{inp}.json'))['batch']
    batch = tree_map(lambda x: x.to(device), batch)
    base_batches.append(batch)

def get_module_shapes(module: nn.Module):
    named_params = dict(module.named_parameters())
    shapes = { k: torch.tensor(v.shape) for k,v in named_params.items()}
    return shapes

def get_param_distributions(module: nn.Module):
    named_params = dict(module.named_parameters())
    dist = {
        k: torch.tensor([v.mean(), v.std()]) for k, v in named_params.items()
    }
    return dist


def default_initialize_parameters(module: nn.Module):
    with torch.no_grad():
        module_params = list(module.parameters())
        param_count = len(module_params)
        for i, v in enumerate(module_params):
            b = np.linspace(0.5, 1.5, param_count)[i]
            param_value = torch.linspace(-b, b, v.numel(), device=device).reshape(v.shape)
            v.copy_(param_value)


# ## Step 1: InputEmbedder without AtomAttentionEncoder
# 
# As a first step, we will implement the InputEmbedder with a mock AtomAttentionEncoder. Go to `input_embedder.py` and implement `__init__`, `relative_encoding`, and `forward`. For now, set `atom_cross_att` to `None` during initialization. The testing code will monkey-patch a zero-generating function for running the tests, so you can still make a call to `atom_cross_att` from your code (which you will need within InputFeatureEmbedder, line 1 in Algorithm 1).
# 
# Run the following cell to check your implementation.

# In[ ]:


from input_embedding.input_embedder import InputEmbedder

global_config = small_config.global_config
c_s = global_config.c_s
c_z = global_config.c_z
c_s_input = global_config.c_s_input
rel_feat_dim = global_config.rel_feat_dim

input_embedder = InputEmbedder(c_s, c_z, c_s_input, rel_feat_dim, small_config.input_embedding_config).to(device)

class AtomCrossAttPatch(nn.Module):
    def forward(self, ref_features: ReferenceFeatures):
        # Patches the missing atom_cross_att function to return zeros for the token activations,
        # and None for the skip connections
        c = small_config.input_embedding_config.atom_attention_config.c_token
        n_token = ref_features.atom_count // 24
        batch_shape = ref_features.mask.shape[:-1]
        return torch.zeros(batch_shape + (n_token, c), device=device), None

input_embedder.atom_cross_att = AtomCrossAttPatch()

# Module shapes
ttr.log_or_compare(get_module_shapes(input_embedder), 'input_embedder_shapes')

# Relative encodings
for batch in base_batches:
    rel_enc, rel_feat= input_embedder.relative_encoding(batch)
    ttr.log_or_compare(rel_enc, 'input_embedder_rel_enc')
    ttr.log_or_compare(rel_feat, 'input_embedder_rel_feat')

# Full forward
for batch in base_batches:
    s_input, s_init, z_init, rel_feat = input_embedder(batch)
    ttr.log_or_compare(s_input, 'input_embedder_s_input')
    ttr.log_or_compare(s_init, 'input_embedder_s_init')
    ttr.log_or_compare(z_init, 'input_embedder_z_init')
    ttr.log_or_compare(rel_feat, 'input_embedder_rel_feat')


# ## Step 2: Common Modules and Diffusion Transformer
# 
# The AtomAttentionEncoder already uses the Diffusion Transformer (the core part of the Diffusion Module) in itself, just with smaller channel dimensions. The Diffusion Transformer implementation in AlphaFold is basically identical to the one in the original [Diffusion Transformer paper](https://arxiv.org/abs/2212.09748), where they suggested using transformers as the noise prediction networks in image diffusion. 
# 
# The basic information flow of a transformer is (tokens_in) -> (tokens_out). Transformers for diffusion are usually a bit special, because you want to allow for conditional input: If we know that the generated picture should be a bird, then the noise prediction process should nudge in that direction. Because of that, for diffusion transformers, the information flow is (tokens_in, conditioning) -> (tokens_out). You will see that pattern everywhere in how we implement the modules (e.g. instead of LayerNorm, we have AdaptiveLayerNorm that also considers the conditioning).
# 
# Start by implementing the modules `AdaptiveLayerNorm`, `ConditionedTransitionBlock`, and `Transition` in `common/modules.py`. The last one, `Transition`, will not be required for the modules we build today, but it's super simple and in doing so we can close off on `common/modules.py` with this chapter.
# 
# After implementing the modules, run the following cell to test your implementation.

# In[ ]:


from common.modules import AdaptiveLayerNorm, ConditionedTransitionBlock, Transition

c_a = 8
c_s = 4
n = 3
n_token = 5

torch.manual_seed(0)

batch_shapes = [(), (2,)]
inputs_act = []
inputs_single = []

for batch_shape in batch_shapes:
    inputs_act.append(torch.randn(batch_shape + (n_token, c_a), device=device))
    inputs_single.append(torch.randn(batch_shape + (n_token, c_s), device=device))

modules = [AdaptiveLayerNorm(c_a, c_s), ConditionedTransitionBlock(c_a, c_s, n), Transition(c_a, n)]
for mod in modules:
    mod.to(device)

for module, name in zip(modules, ['adaptive_layer_norm', 'conditioned_transition', 'transition']):
    for i in range(len(batch_shapes)):
        if name in ['adaptive_layer_norm', 'conditioned_transition']:
            out = module(inputs_act[i], inputs_single[i])
        else:
            out = module(inputs_act[i])

        ttr.log_or_compare(out, f'{name}_{i}')


# These modules (AdaptiveLayerNorm and ConditionedTransitionBlock) are the equivalents of the LayerNorm and FeedForward in regular transformers, just with an included conditioning. However, just like in transformers, they don't allow for cross-talk between tokens, which is single-handedly done by the Attention part. This is implemented in `AttentionPairBias`, which is Algorithm 24 in the paper. 
# 
# Because we want to allow for sparse attention within the AtomAttentionEncoder, we are using Pytorch's FlexAttention module for the actual attention implementation. Aside of that, as described in Algorithm 24, the code will have some if-branches, because the module is allowed to be used both with and without a conditioning input. The docstrings will guide you through how to do the implementation. This cell will only test the case of block_mask being None, in the step afterward we will implement the atom block masks.
# 
# Implement `AttentionPairBias` in `common/modules.py`, then test your implementation by running the following cell.

# In[ ]:


from common.modules import AttentionPairBias

def att_pair_bias_check(n_token, block_mask, batch_shape, test_name, atom_level):
    n_head = 2
    c_z = 16
    c_a = 8
    c_s = 32

    a = torch.randn(batch_shape + (n_token, c_a), device=device)
    s = torch.randn(batch_shape + (n_token, c_s), device=device)
    if not atom_level:
        z = torch.randn(batch_shape + (n_token, n_token, c_z), device=device)
    else:
        z_i = torch.randn(batch_shape + (n_token, 1, c_z), device=device)
        z_j = torch.randn(batch_shape + (1, n_token, c_z), device=device)
        z = BlockSparseTensor.broadcast_up(z_i, block_mask, batch_shape) + \
            BlockSparseTensor.broadcast_up(z_j, block_mask, batch_shape)

    att_pair_bias_no_cond = AttentionPairBias(c_a, c_z, n_head, c_s=None, atom_level=atom_level)
    att_pair_bias_cond = AttentionPairBias(c_a, c_z, n_head, c_s, biased_layer_norm_z=False, atom_level=atom_level)
    att_pair_bias_no_cond.to(device)
    att_pair_bias_cond.to(device)

    mod_list = [att_pair_bias_no_cond, att_pair_bias_cond]
    mod_names = ['att_pair_bias_no_cond', 'att_pair_bias_cond']
    mod_inputs = [(a, z, block_mask), (a, z, block_mask, s)]

    for mod, mod_name, mod_input in zip(mod_list, mod_names, mod_inputs):
        param_dist = get_param_distributions(mod)
        default_initialize_parameters(mod)
        out = mod(*mod_input)

        ttr.log_or_compare(param_dist, f'{mod_name}_param_dist_{test_name}')
        ttr.log_or_compare(out, f'{mod_name}_out_{test_name}')


att_pair_bias_check(16, None, (), 'single', atom_level=False)
att_pair_bias_check(16, None, (2,), 'batch', atom_level=False)


# Next, we want to allow for masking with BlockMasks. We left out the creation of block masks during feature extraction, and we are now making good on that. Go to `feature_extraction/token_features.py` and `feature_extraction/reference_features.py` and implement the `setup_block_mask` functions in both. The docstrings will guide you through the implementation. Afterward, run the following two cells to test your implementation. We cannot test your implementation of `AttentionPairBias` with `atom_level=True` yet, because we need a working `BlockSparseTensor` class for that. We will do that later.

# In[ ]:


# BlockMask tests
from feature_extraction.feature_extraction import collate_batch

torch.manual_seed(0)

batch_single: Batch = base_batches[0]
batch_joined: Batch = collate_batch([batch_single, batch_single])

for b in [batch_single, batch_joined]:
    b.reference_features.setup_block_mask(num_diffusion_samples=2)
    b.token_features.setup_block_mask()


block_masks = sum(
    [
        [b.token_features.block_mask, b.reference_features.block_mask, b.reference_features.block_mask_diffusion] 
        for b in [batch_single, batch_joined]
    ],
    [])
block_mask_names = sum(
    [
        [f'block_mask_{base_name}_tokens', f'block_mask_{base_name}_atoms', f'block_mask_{base_name}_atoms_diffusion']
    for base_name in ['batch_single', 'batch_joined']
    ],
    []
)

for block_mask, block_mask_name in zip(block_masks, block_mask_names):
    ttr.log_or_compare(block_mask.block_mask.kv_num_blocks, f'{block_mask_name}_kv_nums')
    ttr.log_or_compare(block_mask.block_mask.kv_indices, f'{block_mask_name}_kv_indices')
    ttr.log_or_compare(block_mask.block_mask.to_dense().nonzero(), f'{block_mask_name}_non_zero_inds')

print('Blockmask tests passed.')

# AttentionPairBias checks
att_pair_bias_check(batch_single.token_features.token_count, batch_single.token_features.block_mask, (), 'bm_single_token', atom_level=False)
att_pair_bias_check(batch_joined.token_features.token_count, batch_joined.token_features.block_mask, (2,), 'bm_batch_token', atom_level=False)

print('Token-level BlockMask AttentionPairBias tests passed.')


# With `AttentionPairBias` done, the diffusion transformer is fairly straightforward, it's basically just a stack of `ConditionedTransitionBlock` and `AttentionPairBias` blocks. Implement `DiffusionTransformer` in `common/modules.py` and check your implementation by running the following cell.

# In[ ]:


from common.modules import DiffusionTransformer

torch.manual_seed(0)

n_token = 16
n_head = 2
n_blocks = 3
c_z = 16
c_a = 8
c_s = 32

a = torch.randn((n_token, c_a), device=device)
s = torch.randn((n_token, c_s), device=device)
z = torch.randn((n_token, n_token, c_z), device=device)

diffusion_transformer = DiffusionTransformer(c_a, c_z, n_head, c_s, n_blocks, atom_level=False).to(device)

ttr.log_or_compare(get_module_shapes(diffusion_transformer), 'diffusion_transformer_param_shapes')
ttr.log_or_compare(get_param_distributions(diffusion_transformer), 'diffusion_transformer_param_dist')
ttr.log_or_compare(diffusion_transformer(a, s, z, None), 'diffusion_transformer_forward')

print('Diffusion transformer tests passed.')


# ## Optional Step 3: BlockSparseTensor
# 
# We are, theoretically, ready to implement AtomAttentionEncoder, but there is one caveat. The whole reason for implementing block-sparse attention is that, in atom-layout, pair representations are too large to be computationally feasible. However, looking at Algorithm 5 (Atom Attention Encoder), there is the nasty pair representation $p_{lm}$, outside of the attention modules. If we were storing that in dense layout, we would have the same issue as with dense attention.
# 
# For that reason, we need a way to store general pair representations in a block-sparse format efficiently, and that's exactly what `BlockSparseTensor` in `common/block_sparse_tensor.py` does, together with the accompanying class `ExtendedBlockMask` that wraps a `BlockMask` from `flex_attention` and provides utilities for indexing block-sparse tensors based on the sparsity pattern. 
# 
# Concretely, we have:
# - `_build_forward_index` in `ExtendedBlockMask`, a method that creates an (batch_idx, query_block_idx, key_block_idx) -> block_idx lookup table
# - `_build_inverse_indices` in `ExtendedBlockMask`, which provides indices that allow creation of a BST from a dense tensor, used for example in
# -  `broadcast_up` in `BlockSparseTensor`, providing an efficient way to broadcast an input to the full pair representation shape (without realizing the full dense tensor)
# 
# The indexing operations are pretty technical, so please feel free to just take the code from the solutions folder and look through it. If you want to do the implementation yourself, the comments in the python file will guide you through it, and you can run the following cell to test your implementation.

# In[ ]:


from common.block_sparse_tensor import BlockSparseTensor

for batch in [batch_single, batch_joined]:
    batch.reference_features.setup_block_mask(num_diffusion_samples=2)
    batch.token_features.setup_block_mask()

block_mask_single = batch_single.reference_features.block_mask
block_mask_diffusion = batch_single.reference_features.block_mask_diffusion
block_mask_batch = batch_joined.reference_features.block_mask
block_size = block_mask_single.block_mask.BLOCK_SIZE[0]

block_masks = [block_mask_single, block_mask_diffusion, block_mask_batch]
block_mask_names = ['base_mask', 'diffusion_mask', 'batch_mask']

dense_inp = torch.randn(batch_single.reference_features.atom_count, 2, device=device)
dense_inp_batch = torch.randn(2, batch_joined.reference_features.atom_count, 2, device=device)

for block_mask, block_mask_name in zip(block_masks, block_mask_names):
    ttr.log_or_compare(block_mask.forward_index, f'bst_{block_mask_name}_forward_index')

print('Forward index tests passed.')

for block_mask, block_mask_name in zip(block_masks, block_mask_names):
    for idx, idx_name in zip(block_mask.inverse_indices, ['batch_idx', 'q_idx', 'k_idx']):
        ttr.log_or_compare(idx, f'bst_{block_mask_name}_inverse_indices_{idx_name}')

print('Inverse indices tests passed.')


# broadcast_up checks

# query dim, key dim (no batch or channel dim)
inp1 = torch.randn(batch_single.reference_features.atom_count, 1, device=device)
# batch dim, query dim, key dim, channel dim
inp2 = torch.randn(2, 1, batch_joined.reference_features.atom_count, 3, device=device)
ttr.log_or_compare(BlockSparseTensor.broadcast_up(inp1, block_mask_single, ()).physical, 'bst_broadcast_up_single')
ttr.log_or_compare(BlockSparseTensor.broadcast_up(inp2, block_mask_batch, (2,)).physical, 'bst_broadcast_up_batch')

print('broadcast_up tests passed.')


# With a working `BlockSparseTensor` in place, we can finally check if your `AttentionPairBias` implementation works correctly with block-sparse biases!

# In[ ]:


att_pair_bias_check(batch_single.reference_features.atom_count, batch_single.reference_features.block_mask, (), 'bm_single_atoms', atom_level=True)
att_pair_bias_check(batch_joined.reference_features.atom_count, batch_joined.reference_features.block_mask_diffusion, (2, 2), 'bm_batch_atoms', atom_level=True)


# ## Final Step 4:
# 
# Now we are all set to implement AtomAttentionEncoder, and with that, the full Input Embedder! The AtomAttentionEncoder consists of three steps:
# 1) SingleConditioning (Line 1 in Algorithm 5)
# 2) PairConditioning (Line 2-6 in Algorithm 5)
# 3) Everything else
# 
# The following cell tests your implementation of the single conditioning module (`SingleAtomConditioning` in `atom_attention.py`) and the pair conditioning module (`AtomPairConditioning` in `atom_attention.py`).

# In[ ]:


from input_embedding.atom_attention import SingleAtomConditioning, AtomPairConditioning

single_atom_conditioning = SingleAtomConditioning(small_config.input_embedding_config.atom_attention_config).to(device)
atom_pair_conditioning = AtomPairConditioning(small_config.input_embedding_config.atom_attention_config, False).to(device)

# Module shapes
ttr.log_or_compare(get_module_shapes(single_atom_conditioning), 'single_atom_cond_shapes')
ttr.log_or_compare(get_module_shapes(atom_pair_conditioning), 'atom_pair_cond_shapes')

# Forward pass
for batch in [batch_single, batch_joined]:
    single_atom_out = single_atom_conditioning(batch.reference_features)
    atom_pair_out = atom_pair_conditioning(batch.reference_features)
    ttr.log_or_compare(single_atom_out, 'single_atom_out')
    ttr.log_or_compare(atom_pair_out.physical, 'atom_pair_out')

print('SingleAtomConditioning and AtomPairConditioning tests passed.')


# Now, using these two modules, you can implement the `AtomAttentionEncoder` in `atom_attention.py`. Except for the trunk update, this should be straightforward. The trunk update will need a bit of head-wrapping-around-it, mostly because of the indexing of $z$ in line 10 in the algorithm. The indexing there promotes $z$ from a regular tensor to a block-sparse tensor with atom dimensions, and you will need a good understanding of how our `BlockSparseTensor` type works to fully understand it. The docstrings in `atom_attention.py` will help you with the necessary steps.
# 
# After implementing `AtomAttentionEncoder`, you can also fix your `InputEmbedder` in `input_embedding.py`. The following cell tests both modules.

# In[ ]:


from input_embedding.atom_attention import AtomAttentionEncoder
from feature_extraction.feature_extraction import tree_map

c_s = small_config.global_config.c_s
c_z = small_config.global_config.c_z
c_s_input = small_config.global_config.c_s_input
rel_feat_dim = small_config.global_config.rel_feat_dim

torch.manual_seed(0)

atom_att_enc = AtomAttentionEncoder(c_s, c_z, small_config.input_embedding_config.atom_attention_config, use_trunk=False)
atom_att_enc_trunk = AtomAttentionEncoder(c_s, c_z, small_config.input_embedding_config.atom_attention_config, use_trunk=True)
input_embedder = InputEmbedder(c_s, c_z, c_s_input, rel_feat_dim, small_config.input_embedding_config)

modules = [atom_att_enc, atom_att_enc_trunk, input_embedder]
names = ['atom_att_enc', 'atom_att_enc_trunk', 'input_embedder_full']
for mod in modules:
    mod.to(device)

# Parameter shapes check
for mod, name in zip(modules, names):
    ttr.log_or_compare(get_module_shapes(mod), f'{name}_module_shapes')

def expand_diff_sample_dim(x):
    return x[None, ...].broadcast_to((2,) + x.shape)

# Forward check AtomAttentionEncoder
for i, (mod, name) in enumerate(zip(modules[:2], names[:2])):
    for batch in [batch_single, batch_joined]:
        if i==1:
            batch = tree_map(expand_diff_sample_dim, batch, skip_unconvertible_entries=True)
        batch_shape = batch.token_features.mask.shape[:-1]
        n_tokens = batch.token_features.mask.shape[-1]
        n_atoms = batch.reference_features.mask.shape[-1]
        r = torch.randn(batch_shape + (n_atoms, 3), device=device)
        s_trunk = torch.randn(batch_shape + (n_tokens, c_s), device=device)
        z = torch.randn(batch_shape + (n_tokens, n_tokens, c_z), device=device)

        token_act, (single_act, single_cond, pair_cond) = mod(batch.reference_features, r, s_trunk, z)
        out = {'token_act': token_act, 'single_act': single_act, 'single_cond': single_cond, 'pair_cond': pair_cond.physical}
        ttr.log_or_compare(out, f'{name}_forward')


# Forward check InputEmbedder
for batch in [batch_single, batch_joined]:
    s_input, s_init, z_init, rel_feat = input_embedder(batch)
    out = {'s_input': s_input, 's_init': s_init, 'z_init': z_init, 'rel_feat': rel_feat}
    ttr.log_or_compare(out, f'input_embedder_full_forward')

print('AtomAttentionEncoder and InputEmbedder tests passed.')


# To wrap it of, please also implement the `AtomAttentionDecoder` in `atom_attention.py`. While `AtomAttentionEncoder` does the direction atom_layout -> token_layout, `AtomAttentionDecoder` does the reverse direction, but in a way simpler fashion. Basically, it just broadcasts the token features so that each atom index gets a copy of its corresponding token's feature, then runs our diffusion transformer on it. For the broadcasting, we already implemented the method `reference_features.to_atom_layout` in the previous chapter. The `AtomAttentionDecoder` is not required for input embedding, but it's quite simple and we will implement it here so we can close off on `atom_attention.py`. After you finished your implementation, run the following cell to test your code.

# In[ ]:


from input_embedding.atom_attention import AtomAttentionDecoder

atom_att_conf = small_config.input_embedding_config.atom_attention_config
atom_attention_decoder = AtomAttentionDecoder(atom_att_conf).to(device)

ttr.log_or_compare(get_module_shapes(atom_attention_decoder), 'atom_att_dec_module_shapes')

c_token = atom_att_conf.c_token
c_atom = atom_att_conf.c_atom
c_atompair = atom_att_conf.c_atompair


for batch in [batch_single, batch_joined]:
    batch_shape = batch.token_features.mask.shape[:-1]
    n_token = batch.token_features.mask.shape[-1]
    n_atoms = batch.reference_features.mask.shape[-1]
    a = torch.randn(batch_shape + (n_token, c_token), device=device)
    q_skip = torch.randn(batch_shape + (n_atoms, c_atom), device=device)
    c_skip = torch.randn(batch_shape + (n_atoms, c_atom), device=device)
    p_skip = torch.randn(batch_shape + (n_atoms, 1, c_atompair), device=device)
    p_skip = BlockSparseTensor.broadcast_up(p_skip, batch.reference_features.block_mask, batch_shape)

    out = atom_attention_decoder(a, q_skip, c_skip, p_skip, batch.reference_features)
    ttr.log_or_compare(out, 'atom_attention_decoder_forward')

print('AtomAttentionDecoder tests passed.')



# ## Conclusion
# 
# With that, we implemented all the modules for input embedding, and some more. Even though input embedding only contains a small fraction of the model parameters, it already contains a big chunk of the module code we will need for the series.
# 
# In the next chapter, we will implement the Evoformer, the main trunk of AlphaFold, and we will see AlphaFold's huge memory footprint (compared to its relatively small model size), how we can deal with it using activation offloading, and check how compiling affects model throughput. In the later chapters, we will implement AlphaFold's diffusion sampler and try to train it. Stay tuned!
