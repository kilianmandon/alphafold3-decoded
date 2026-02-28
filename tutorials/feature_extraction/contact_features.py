import numpy as np
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts

import common.utils as utils


class CalculateContactMatrix(Transform):
    """
    Calculates the contact matrix of shape (n_tokens n_tokens 1). The contact matrix is a binary matrix that has
    ones exactly at indices i<j where either i or j is atomized (i.e. a ligand, or a modified residue),
    and there is a bond between any atom in token i and any atom in token j.
    """

    def forward(self, data):
        atom_array = data["atom_array"]
        # bonds is of shape (n_bonds, 3) and has columns (atom1_idx, atom2_idx, bond_type).
        # #The bond_type is not relevant here.
        bonds = atom_array.bonds.as_array()
        contact_matrix = None
        padded_token_count = utils.round_to_bucket(len(get_token_starts(atom_array)))

        """
        TODO: Construct contact_matrix. Note the specifics: Only entries i<j are set to 1, and only if either 
        token i or j is atomized (or both).

        Hints: 
            - Given the atom indices of the bonds, 'bonds', you need the token index each of these atoms belongs to. 
              To do so, you can use get_token_starts to get the index of the first atom in each token of the input, 
              and round the bond atom indices down to those using utils.round_down_to 
              (which optionally returns the index of the target it rounded down to).
            - atom_array.atomize is a boolean array of shape (n_atom,) that indicates whether the atoms belong 
                to an atomized token or not. This can be indexed with the token starts to get a boolean array 
                of shape (n_tokens). Note that you might need to pad this array to the padded token count, 
                since len(get_token_starts(atom_array)) is the unpadded token count.
            - be sure to set only entries i<j to 1 (not where i==j).
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        data["contact_matrix"] = contact_matrix.astype(np.float32)

        return data
