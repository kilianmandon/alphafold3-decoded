from dataclasses import dataclass, fields
from functools import cached_property
import math
import numpy as np
from atomworks.constants import UNKNOWN_AA, STANDARD_RNA, UNKNOWN_RNA, STANDARD_DNA, UNKNOWN_DNA, STANDARD_AA
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from common.residue_constants import AF3_TOKENS_MAP
import common.utils as utils

Array = np.ndarray | torch.Tensor

def round_to_bucket(v: int) -> int:
    """
    Calculates the smallest bucket size that is at least as big as v.

    Args:
        v (int): The number of tokens for the input.

    Returns:
        int: The smallest bucket size that fits the tokens.
    """
    buckets = np.array([256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072,
                        3584, 4096, 4608, 5120])

    selected_bucket = None

    """
    TODO: Implement the logic to select the smallest bucket value that fulfills b >= v.
    """

    selected_bucket =  buckets[np.argmax(buckets >= v)]

    """ End of your code """

    return selected_bucket

def encode_restype(restype: np.ndarray[np.str_]) -> np.ndarray[int]:
    """
    Encodes residue types into integer token indices.
    """

    restype_encoding = None
    """
    TODO: Map restype through the AF3_TOKENS_MAP dictionary to get the integer encoding. You can do so efficiently using np.vectorize.
    """

    restype_encoding = np.vectorize(lambda x: AF3_TOKENS_MAP.get(x, AF3_TOKENS_MAP[UNKNOWN_AA]))(restype)

    """ End of your code """

    return restype_encoding


@dataclass
class TokenFeatures:
    """
    Token-level features for AlphaFold 3. Each feature has shape (**batch_shape, n_token).
    """

    # Residue index of each token, as stated in the PDB files
    residue_index: Array
    # Token index, e.g. range(n_token)
    token_index: Array
    # unique identifier for each chain, e.g. each pn_unit_iid in the AtomArray
    asym_id: Array
    # unique identifier for each group of identical chains, e.g. each pn_unit_entity in the AtomArray
    entity_id: Array
    # unique identifier for each chain within an entity, e.g. within equal entity_ids
    sym_id: Array
    # Mask to indicate presence of a token (1) vs padding (0)
    mask: Array
    # Encoded residue type of each token
    restype: Array
    # Boolean flags for whether each token is part of a protein, RNA, DNA, or ligand
    is_rna: Array
    is_dna: Array
    is_protein: Array
    is_ligand: Array


    @property
    def token_count(self):
        return self.residue_index.shape[-1]

    @cached_property
    def block_mask(self) -> BlockMask:
        block_mask = None

        """ 
        TODO in Chapter 2 (Input Embedding): Create a block mask which that masks out keys that should not be attended, e.g. where self.mask is 0. For that, implement a function block_mask with signature (b, h, q, k) -> bool and use it in create_block_mask to build the block mask. You an use utils.unify_batch_dimension to unify the mask of shape (**batch_shape, n_token) to shape (batch_size, n_token).
        """

        mask = utils.unify_batch_dimension(self.mask, self.mask.shape[:-1])
        def mask_mod(b, h, q, k):
            return mask[b, k]
        
        batch_size = mask.shape[0]
        block_mask = create_block_mask(mask_mod, batch_size, None, self.token_count, self.token_count, self.mask.device)

        """ End of your code """

        return block_mask
    


class CalculateTokenFeatures(Transform):
    """
    Calculates the token-level features for AlphaFold 3.
    """

    def forward(self, data):
        atom_array = data['atom_array']

        """
        TODO: Build a TokenFeatures object based on atom_array. The four basic steps are:
        1. Get token-level atom array using get_token_starts and indexing with these into atom_array 
        2. Extract the information from the token-level features:
            - Note: token_index, asym_id, entity_id, sym_id all start at 1 in AF3, so be sure to add 1 to the respective features
            - residue_index: directly accessible as atom_arary.res_id
            - token_index: just a range(n_token) + 1
            - asym_id: you can use np.unique(..., return_inverse=True) to get labels 0, ..., n_chain-1 for each unique chain identifier based on atom_array.pn_unit_iid
            - entity_id: same as asym_id but based on atom_array.pn_unit_entity
            - sym_id: iterate over the entity_ids, use the same procedure as for asym_id but on the masked atom array [entity_id == current_entity_id]
            - restype: obtained from token_array.res_name using encode_restype
            - is_rna, is_dna, is_protein: from token_array.res_name, using np.isin and the constants STANARD_RNA, STANDARD_DNA, STANDARD_AA, UNKNOWN_RNA, UNKNOWN_DNA, UNKNOWN_AA
            - is_ligand: inverse of the union of the three previous arrays
            - mask: ones like residue_index
        3. Padding: Calculate the padded token count from the actual token count using round_to_bucket, pad all features using utils.pad_to_shape
        4. Build a TokenFeatures object with all the features
        """

        token_starts = get_token_starts(atom_array)

        token_array = atom_array[token_starts]

        token_index = np.arange(len(token_array))
        res_id = token_array.res_id

        unique_entity_ids, entity_id = np.unique(token_array.pn_unit_entity, return_inverse=True)
        _, asym_id = np.unique(token_array.pn_unit_iid, return_inverse=True)

        sym_id = np.zeros_like(entity_id)

        for entity in unique_entity_ids:
            mask = entity_id == entity
            _, sym_id_subset = np.unique(token_array.pn_unit_iid[mask], return_inverse=True)
            sym_id[mask] = sym_id_subset


        restype = encode_restype(token_array.res_name)

        is_rna = np.isin(token_array.res_name, STANDARD_RNA +(UNKNOWN_RNA,))
        is_dna = np.isin(token_array.res_name, STANDARD_DNA + (UNKNOWN_DNA,))
        is_protein = np.isin(token_array.res_name, STANDARD_AA + (UNKNOWN_AA,))
        is_ligand = ~(is_rna | is_dna | is_protein)


        token_features = {
            'residue_index': res_id,
            'token_index': token_index + 1,
            'asym_id': asym_id + 1,
            'entity_id': entity_id + 1,
            'sym_id': sym_id + 1,
            'mask': np.ones(len(token_array), dtype=bool),
            'restype': restype,

            'is_rna': is_rna,
            'is_dna': is_dna,
            'is_protein': is_protein,
            'is_ligand': is_ligand,
        }

        padded_token_count = round_to_bucket(len(token_array))
        for k, v in token_features.items():
            token_features[k] = utils.pad_to_shape(v, (padded_token_count,))

        token_features = TokenFeatures(**token_features)

        """ End of your code """ 

        data['token_features'] = token_features

        return data
