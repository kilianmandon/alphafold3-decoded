from dataclasses import dataclass, fields
from functools import cached_property
import math

import numpy as np
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from torch.nn import functional as F
import rdkit
import torch
from atomworks.io.tools.rdkit import atom_array_from_rdkit, ccd_code_to_rdkit
from atomworks.io.utils.ccd import get_available_ccd_codes
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.rdkit_utils import sample_rdkit_conformer_for_atom_array, generate_conformers
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

import common.utils as utils


Array = np.ndarray | torch.Tensor

@dataclass
class RefStructFeatures:
    """
    Reference Structure Features for AlphaFold 3. All features have shape (**batch_shape, n_atoms), 
    except for atom_name_chars which has shape (**batch_shape, n_atoms, 4).
    """

    element: Array
    charge: Array
    atom_name_chars: Array
    positions: Array
    mask: Array
    ref_space_uid: Array
    token_index: Array

    @property
    def atom_count(self):
        return self.mask.shape[-1]

    @property
    def unpadded_atom_count(self):
        if isinstance(self.mask, np.ndarray):
            return np.sum(self.mask, axis=-1)            
        else:
            return torch.sum(self.mask, dim=-1)


    @cached_property
    def token_layout_ref_mask(self):
        """
        Computes a mask of shape (**batch_shape, n_tokens, 24) indicating which atoms of which tokens are actually 
        present in the input. This is useful for conversion between atom-layout and token-layout.
        """
        token_count = self.atom_count // 24
        token_index = torch.as_tensor(self.token_index)
        mask = torch.as_tensor(self.mask)

        ref_mask = None

        """
        TODO: Compute the token_layout_ref_mask, a boolean array of shape (**batch_shape, n_tokens, 24) that indicates 
        which atoms atoms are actually present in the tokens. For example, if the first token had 4 atoms, 
        the first 4 positions in the mask should be True for that token. The first step is to compute 
        the number of atoms per token, of shape (**batch_shape, n_tokens). Two options to do so are either 
        using F.one_hot(token_index), masking, and summing, or using scatter_add with token_index. If you choose 
        the latter, you can scatter_add from mask so that a 1 is only added for token_index values that actually 
        belong to valid atoms. 
        After computing the number of atoms per token, you can do a broadcasted comparison 
        with range(24) to get the final mask. 
        """

        token_atom_one_hot = F.one_hot(token_index, num_classes=token_count) * mask[..., None]
        token_atom_counts = token_atom_one_hot.sum(dim=-2)
        ref_mask = torch.arange(24, device=mask.device) < token_atom_counts[..., None]

        """ End of your code """

        if isinstance(self.mask, np.ndarray):
            return ref_mask.numpy()
        else:
            return ref_mask


    def to_token_layout(self, feature):
        """
        Converts a feature from atom_layout (shape (**batch_shape, n_atoms, **feat_dims)) to 
        token_layout (shape (**batch_shape, n_tokens, 24, **feat_dims)). 
        """
        batch_shape = self.mask.shape[:-1]
        token_count = self.atom_count // 24
        feature = torch.as_tensor(feature)
        token_layout_ref_mask = torch.as_tensor(self.token_layout_ref_mask)
        out = None

        """
        TODO: Convert feature of shape (**batch_shape, n_atoms, **feat_dims) 
        to shape (**batch_shape, n_tokens, 24, **feat_dims). With the masks we created, this is fairly simple: 
        Construct an all-zeros output array of the target shape, then just set out = feature, 
        using the token_layout_ref_mask on the left and the regular mask on the right.
        """

        feature_shape = feature.shape[len(batch_shape)+1:]
        out_shape = batch_shape + (token_count, 24) + feature_shape
        out = torch.zeros(out_shape, dtype=feature.dtype, device=feature.device)
        out[token_layout_ref_mask] = feature[self.mask]

        """ End of your code """

        if isinstance(self.mask, np.ndarray):
            return out.numpy()
        else:
            return out


    def patch_atom_dimension(self, feature):
        """
        Broadcasts the feature from shape (**batch_shape, n_tokens, **feat_dims) 
        to shape (**batch_shape, n_tokens, 24, **feat_dims).
        """
        batch_shape = self.element.shape[:-1]
        token_count = self.atom_count // 24
        feature = torch.as_tensor(feature)
        is_numpy = isinstance(feature, np.ndarray)

        """
        TODO: Implement the broadcasting. You can use .expand(broadcasted_shape) for the actual broadcasting, 
        after making the shapes compatible by inserting an atom dimension of size 1.
        """

        unsqueezed_shape = batch_shape + (token_count, 1) + feature.shape[len(batch_shape) + 1:]
        broadcasted_shape = batch_shape + (token_count, 24) + feature.shape[len(batch_shape) + 1:]

        feature = feature.reshape(unsqueezed_shape)
        feature = feature.expand(broadcasted_shape)

        """ End of your code """

        if is_numpy:
            return feature.numpy()
        else:
            return feature

    def to_atom_layout(self, feature, has_atom_dimension=True):
        """
        Converts a feature from token_layout (shape (**batch_shape, n_tokens, 24, **feat_dims)) 
        to atom_layout (shape (**batch_shape, n_atoms, **feat_dims)). Optionally, if has_atom_dimension is False, 
        the input feature has shape (**batch_shape, n_tokens, **feat_dims) and is broadcasted to include the 
        atom dimension before conversion.
        """
        batch_shape = self.element.shape[:-1]
        feature = torch.as_tensor(feature)
        mask = torch.as_tensor(self.mask)
        token_layout_ref_mask = torch.as_tensor(self.token_layout_ref_mask)

        out = None

        """
        TODO: Implement the conversion. You can use patch_atom_dimension to handle has_atom_dimension. 
        Then, you can do the layout conversion just as in to_token_layout using the two boolean masks, but in reverse.
        """

        if not has_atom_dimension:
            feature = self.patch_atom_dimension(feature)

        out_shape = batch_shape + (self.atom_count,) + feature.shape[len(batch_shape)+2:]
        out = torch.zeros(out_shape, dtype=feature.dtype, device=feature.device)
        out[mask] = feature[token_layout_ref_mask]

        """ End of your code """

        if isinstance(self.mask, np.ndarray):
            return out.numpy()
        else:   
            return out

    @cached_property
    def block_mask(self) -> BlockMask:
        """
        Creates a local attention block mask for use with flex_attention. Atoms are split into overlapping blocks 
        of size 128, and attend only other atoms within their block during atom attention. Concretely, 
        the block centers for the block rows are chosen as range(16, n_atoms, step=32), and the right and left bounds 
        of the block are chosen as center +/- 64. If these boundaries exceed range(n_atoms), they are shifted 
        so that they fit. The blocks have height 32.
        The block_mask is cached, so that it isn't recomputed on reevaluation.
        """

        batch_shape = self.mask.shape[:-2]
        batch_size = math.prod(batch_shape) 
        # self.atom_count is a single number, since all inputs in a batch are padded to the same atom count
        n_blocks = self.atom_count // 32
        # unpadded_atom_count has shape (batch_size,)
        unpadded_atom_count = utils.unify_batch_dimension(self.unpadded_atom_count, batch_shape) 

        block_mask = None

        """
        TODO: This is not part of Chapter 1! Implement this in Chapter 2, Input Embedding.  
        For the mask_mod, we want to create two arrays of shape (batch_size, n_blocks), left_bounds and right_bounds, 
        so that we can just check (left_bounds[b, q//32] <= k) & (k < right_bounds[b, q//32]) to check 
        if q attends k (q//32 is the block q belongs to). To construct these bounds, do as follows:
        - Construct centers as range(16, n_atoms, step=32), and expand it to include the batch_size dimension. 
            Note: The fact that this has shape (n_blocks,), n_blocks = n_atoms // 32, is due to the choice 
            of the buckets for n_tokens: Each of the buckets is divisible by 32.
        - Create left and  right bounds by shifting by +/- 64.
        - Calculate the left and right violations of the borders (by clamping left_bounds and the difference 
            of right_bounds and unpadded_atom_count to the respective sides, using torch.clamp), 
            and calculate the total shift required to fix the violation. 
            Take care you do the sign correctly.
        - Add the shift, and detach the bounds and unpadded_atom_count using .detach() so that the gradients 
            don't backprop through them
        - build the mask_mod, by checking the boundary conditions and whether q is smaller than the unpadded atom count
        - Use create_block_mask for to create the actual block mask.
        """

        centers = torch.arange(16, self.atom_count, 32, device=self.mask.device, dtype=torch.int32)
        centers = centers.unsqueeze(0).expand(batch_size, n_blocks)
        left_bounds = centers - 64
        right_bounds = centers + 64

        left_violation = torch.clamp(left_bounds, max=0)
        right_violation = torch.clamp(right_bounds - unpadded_atom_count[:, None], min=0)
        shift = -(left_violation + right_violation)

        left_bounds += shift
        right_bounds += shift

        left_bounds = left_bounds.detach()
        right_bounds = right_bounds.detach()
        unpadded_atom_count = unpadded_atom_count.detach()

        def mask_mod(b, h, q, k):
            return (q < unpadded_atom_count[b]) & (left_bounds[b, q//32] <= k) & (k < right_bounds[b, q//32])
        
        block_mask = create_block_mask(mask_mod, batch_size, None, self.atom_count, self.atom_count, self.mask.device)

        """ End of your code """
        return block_mask


class CalculateRefStructFeatures(Transform):

    @staticmethod
    def calculate_ref_positions(atom_array: AtomArray):
        """
        Calculates the reference positions for all residues in the input atom_array. Individual conformers are sampled
        for each residue type in each distinct chain. Using the same conformers for same residues within a chain. 

        Args:
            atom_array: AtomArray of shape (n_atoms,)

        Returns: 
            positions: Positions array of shape (n_atoms, 3)
        """
        residue_borders = get_residue_starts(atom_array, add_exclusive_stop=True)
        residue_starts, residue_ends = residue_borders[:-1], residue_borders[1:]
        res_names = atom_array.res_name[residue_starts]
        chain_iids = atom_array.chain_iid[residue_starts]
        known_ccd_codes = get_available_ccd_codes() - { 'UNL' }

        positions = None

        """
        TODO: Create the array positions of shape (n_atoms, 3). To do so, go through the following steps:
          - Create dictionaries for cached_conformers and cached_unknown_conformers
          - Initialize an all-zeros output array 'positions'
          - Iterate over the residue names / chain_iids
          - If the res_name is in the known_ccd_codes, either look it up from your cache 
            (e.g. look up if a conformer for the res_name | chain_iid pair was already generated) or create a new 
              conformer with the steps: 
              1. use ccd_code_to_rdkit to create an RDKit Mol object
              2. use generate_conformers on that to add a conformer to the Mol object 
              3. use atom_array_from_rdkit on that updated Mol object to convert it to an atom_array, cache that
          - If the res_name is not in the known_ccd_codes, either look it up from your unknown conformer cache 
            (e.g. look up if a conformer for the res_name | chain_iid pair was already genearted) 
              or create a new conformer with the steps:
              1. extract the slice of the atom_array from the residue start to its end
              2. use sample_rdkit_conformer_for_atom_array to get a conformer based on the atom_array, instead of 
                 a ccd Mol as in the case of known_ccd_codes. This directly returns an atom_array. Cache it.
          - However you retrieved the conformer (which is an atom_array), iterate over the atom_array indices 
              between residue start and residue end. Search for the atom_array.atom_name at the indices 
              within the conformer.atom_name, and plug the conformer.coords of that index into the output positions. 
              If you don't find a matching atom name, print a warning and skip that atom.
        """

        cached_conformers = {}
        cached_unknown_conformers = {}

        positions = np.zeros((len(atom_array), 3))

        for i, (res_name, chain_iid) in enumerate(zip(res_names, chain_iids)):
            if res_name in known_ccd_codes:
                if (res_name, chain_iid) not in cached_conformers:
                    mol = ccd_code_to_rdkit(res_name)
                    # Note: AF3 sanitizes the atom order by sorting atoms based on their name. 
                    # This does not strongly affect the results, since the conformer generation doesn't rely 
                    # on the atom order, aside from its random seed.

                    # annotations = mol._annotations
                    # order = np.argsort(annotations['atom_name'])
                    # mol = rdkit.Chem.RenumberAtoms(mol, order.tolist())
                    # mol._annotations = {
                    #     k: v[order] for k, v in annotations.items()
                    # }
                    # mol = generate_conformers(mol, seed=1, optimize=False, attempts_with_distance_geometry=250, hydrogen_policy='keep')

                    mol = generate_conformers(mol)
                    cached_conformers[(res_name, chain_iid)] = atom_array_from_rdkit(mol, conformer_id=0)
                conformer = cached_conformers[(res_name, chain_iid)]
            else:
                if res_name not in cached_unknown_conformers:
                    res_atom_array = atom_array[residue_starts[i]:residue_ends[i]]
                    cached_unknown_conformers[res_name] = sample_rdkit_conformer_for_atom_array(res_atom_array)

                conformer = cached_unknown_conformers[res_name]

            for j in range(residue_starts[i], residue_ends[i]):
                matching_atom_idx = np.nonzero(conformer.atom_name == atom_array.atom_name[j])[0]
                if len(matching_atom_idx) == 0:
                    print(f'Warning: could not find matching atom for residue {res_name}')
                else:
                    positions[j] = conformer.coord[matching_atom_idx]

        """ End of your code """

        return positions

    def prep_atom_chars(self, atom_names):
        """
        Pads or crops the atom names to length 4 and encodes them with their ascii code. 
        The ascii codes are subtracted by 32, so that a space is encoded as a zero.

        Args:
            atom_names (np.ndarray): String array of shape (n_atoms,)

        Returns:
            np.ndarray: Encoded atom chars of shape (n_atoms, 4)
        """

        encoded = None

        """
        TODO: Implement the padding/cropping and encoding. You can use np.strings.ljust for padding, 
        np.strings.slice for cropping, and np.strings.encode(..., encoding='ascii') for encoding. 
        After encoding, the array will have shape (n_atoms,) with dtype 'S4' (string of length 4). 
        You can use .view(np.uint8) to convert it to bytes (of shape (4*n_atoms,)), then reshape it 
        for the desired output format. Finally, subtract 32 to get the correct range.
        """
        padded = np.strings.ljust(atom_names, width=4)
        cropped = np.strings.slice(padded, 0, 4)
        encoded = np.strings.encode(cropped, encoding='ascii')
        encoded = encoded.view(np.uint8).reshape(-1, 4) - 32

        """ End of your code """

        return encoded


    def forward(self, data: dict):
        atom_array: AtomArray = data['atom_array']
        ref_struct_features = None

        """
        TODO: Implement the feature construction. Concretely,
          - element and charge can be directly obtained from atom_array.atomic_number and atom_array.charge
          - atom_name_chars and positions can be computed using the functions you implemented above
          - mask is just ones of shape (n_atoms,) (before padding)
          - ref_space_uid and token_index can be obtained by rounding range(n_atoms) down to the nearest 
              residue starts / token starts respectively, using utils.round_down_to with return_indices=True 
              (the indices are the ids).
          - In the end, compute the padded atom count as token_count * 24 (token_count from data['token_features']) 
              and pad all features to leading shape (n_padded_atom_count,) using utils.pad_to_shape, and assemble 
              the RefStructFeatures dataclass.
        """

        residue_starts = get_residue_starts(atom_array)
        _, ref_space_uid = utils.round_down_to(np.arange(len(atom_array)), residue_starts, return_indices=True)

        token_starts = get_token_starts(atom_array)
        _, token_index = utils.round_down_to(np.arange(len(atom_array)), token_starts, return_indices=True)

        ref_struct = {
            'element': atom_array.atomic_number,
            'charge': atom_array.charge,
            'atom_name_chars': self.prep_atom_chars(atom_array.atom_name),
            'positions': self.calculate_ref_positions(atom_array).astype(np.float32),
            'mask': np.ones_like(atom_array.atomic_number).astype(bool),
            'ref_space_uid': ref_space_uid,
            'token_index': token_index,
        }

        padded_atom_count = data['token_features'].token_count * 24


        for k, v in ref_struct.items():
            padded_shape = (padded_atom_count,) + v.shape[1:]
            ref_struct[k] = utils.pad_to_shape(v, padded_shape)

        ref_struct_features = RefStructFeatures(**ref_struct)

        """ End of your code """

        data['ref_struct'] = ref_struct_features

        return data