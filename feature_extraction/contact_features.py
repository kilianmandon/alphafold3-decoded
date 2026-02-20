import numpy as np
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts

import common.utils as utils


class CalculateContactMatrix(Transform):
    """
    Calculates the contact matrix of shape (n_token, n_token, 1). The contact matrix is a binary matrix that has ones exactly at indices i<j where either i or j is atomized (i.e. a ligand, or a modified residue), and there is a bond between any atom in token i and any atom in token j.
    """
    def forward(self, data):
        atom_array = data['atom_array']
        # bonds is of shape (n_bonds, 3) and has columns (atom1_idx, atom2_idx, bond_type). The bond_type is not relevant here.
        bonds = atom_array.bonds.as_array()
        contact_matrix = None

        """
        TODO: Construct contact_matrix. Note the specifics: Only entries i<j are set to 1, and only if either token i or j is atomized (or both).

        Hints: 
            - You need the token index each atom of the bond belongs to. To do so, you can use get_token_starts to get the atom indices of the tokens, and round the bond atom indices down using utils.round_down_to (which optionally returns the index of the target it rounded down to).
            - atom_array.atomize is a boolean array of shape (n_atom,) that indicates whether each atom is atomized or not. This can be indexed with the token starts to get a boolean array of shape (n_token,).
            - be sure to set only entries i<j (also not i==j) to 1.
        """

        atom1_idxs, atom2_idxs, _ = bonds.T
        token_starts = get_token_starts(atom_array)
        _, atom1_token_indices = utils.round_down_to(atom1_idxs, token_starts, return_indices=True)
        _, atom2_token_indices = utils.round_down_to(atom2_idxs, token_starts, return_indices=True)

        padded_token_count = data['token_features'].token_count
        is_atomized = utils.pad_to_shape(atom_array[token_starts].atomize, (padded_token_count,))

        atom1_token_indices, atom2_token_indices = np.minimum(atom1_token_indices, atom2_token_indices), np.maximum(atom1_token_indices, atom2_token_indices)
        contact_matrix = np.zeros((padded_token_count, padded_token_count))
        contact_matrix[atom1_token_indices, atom2_token_indices] = 1

        non_poly_poly = is_atomized.reshape(-1, 1) | is_atomized.reshape(1, -1)
        non_diagonal = 1 - np.eye(padded_token_count)

        contact_matrix = contact_matrix * non_poly_poly * non_diagonal

        contact_matrix = contact_matrix[..., None]

        """ End of your code """

        data['contact_matrix'] = contact_matrix.astype(np.float32)

        return data