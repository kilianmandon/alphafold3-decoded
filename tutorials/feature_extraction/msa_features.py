import copy
from dataclasses import dataclass
import numpy as np
import torch
from atomworks.constants import UNKNOWN_AA
from atomworks.ml.transforms.atom_array import get_chain_instance_starts
from atomworks.ml.transforms.base import Transform, Compose
from atomworks.ml.transforms.msa._msa_constants import MSA_INTEGER_TO_THREE_LETTER
from atomworks.ml.transforms.msa.msa import LoadPolymerMSAs
from atomworks.ml.utils.token import get_token_count, get_token_starts
from atomworks.io.utils.selection import get_residue_starts


from torch.nn import functional as F
import common.utils as utils
from common.residue_constants import AF3_TOKENS_MAP, PROTEIN_TO_ID

Array = np.ndarray | torch.Tensor


@dataclass
class MSAFeatures:
    """Represents the MSA features for an AlphaFold 3 pass."""

    # msa_feat is of shape (max_msa_sequences, n_tokens, msa_feat_dim=34)
    msa_feat: Array
    # msa_mask is of shape (max_msa_sequences, n_tokens)
    msa_mask: Array
    # target_feat is of shape (n_tokens, target_feat_dim=65)
    target_feat: Array


class HotfixDuplicateRowIfSingleMSA(Transform):
    """
    Due to how AlphaFold manages paired and unpaired MSAs, if there is no paired MSA, the first line of
    the unpaired MSA (which is just the sequence itself) is duplicated.
    This transform patches the data in this case by duplicating the first row.
    """

    def __init__(self, max_msa_sequences):
        self.max_msa_sequences = max_msa_sequences

    def forward(self, data):
        msa_hashes = set(
            [
                hash(v["msa"].tobytes())
                for v in data["polymer_msas_by_chain_id"].values()
            ]
        )
        if len(msa_hashes) == 1:
            msa_data = list(data["polymer_msas_by_chain_id"].values())[0]
            for k, v in msa_data.items():
                new_shape = (v.shape[0] + 1,) + v.shape[1:]
                new_val = np.zeros(new_shape, v.dtype)
                new_val[1:] = v
                new_val[0] = v[0]
                if new_val.shape[0] > self.max_msa_sequences:
                    new_val = new_val[: self.max_msa_sequences]

                msa_data[k] = new_val

            # Note: This is relevant for the profile and deletion_mean features, which shouldn't use the duplicated row.
            # This is because duplication happens during the AF3 MSA pairing procedure, and profile and deletion_mean 
            # are calculated on the unpaired MSA
            msa_data["msa_is_padded_mask"][0] = True

            for k in data["polymer_msas_by_chain_id"]:
                data["polymer_msas_by_chain_id"][k] = copy.deepcopy(msa_data)

        return data


class HotfixAF3LigandAsGap(Transform):
    """
    In the MSA encoding, AF3 actually doesn't use the ligand's restype token but the 
    GAP token (this affects the first row of the MSA). This transform mirrors that.
    """

    def forward(self, data):
        ligand_inds = np.nonzero(data["token_features"].is_ligand)[0]
        data["msa_features"]["msa"][0, ligand_inds] = AF3_TOKENS_MAP["<G>"]
        return data


class EncodeMSA(Transform):
    def __init__(self):
        self.lookup_table = None

        """
        TODO: Set self.lookup_table, a numpy array of shape (len(MSA_INTEGER_TO_THREE_LETTER),). 
        Atomworks stores the loaded MSA as an integer array, and their map MSA_INTEGER_TO_THREE_LETTER tells you 
        which integer corresponds to which amino acid / nucleic acid. We want to allow conversion of the custom 
        Atomworks encoding to the AF3 encoding through simple integer indexing, 
        e.g. af3_enc_msa = self.lookup_table[atomworks_enc_msa]. To do so, if MSA_INTEGER_TO_THREE_LETTER 
        has an entry i: GLY for example, we would need lookup_table[i] = AF3_TOKENS_MAP['GLY']. 
        If 'GLY' was not present, the fallback should be AF3_TOKENS_MAP[UNKNOWN_AA]. 
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

    def forward(self, data):
        """
        TODO: For each entry chain_id, msa_data in data['polymer_msas_by_chain_id'], replace ,msa_data['msa'] with its 
        AF3 encoding, using self.lookup_table.
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        return data


class ConcatMSAs(Transform):
    def __init__(self, max_msa_sequences=16384):
        self.max_msa_sequences = max_msa_sequences

    def forward(self, data):
        """
        Populates data['msa_features'] with the following features:
            - msa: Array of shape (max_msa_sequences, n_tokens) containing the concatenated MSA
            - deletion_count: Array of shape (max_msa_sequences, n_tokens), containing the count of deletions
                (insertions in Atomworks terminology) before each position for each MSA sequence.
            - full_msa_mask: Array of shape (max_msa_sequences, n_tokens), with the full block of 
                shape (max_msa_size, unpadded_token_count) set to 1, ignoring individual msa sizes. 
                max_msa_size is the maximum size of the MSAs across all chains.
            - individual_msa_mask: Array of shape (max_msa_sequences, n_tokens), with the valid positions for each 
                MSA sequence set to 1. Used for calculating the profile and deletion_mean features.
        """

        polymer_msas = data["polymer_msas_by_chain_id"]
        unpadded_token_count = data["token_features"].unpadded_token_count

        """
        TODO: Implement MSA concatenation, consisting of the following steps:
            - Set the full block (max_msa_size, unpadded_token_count) to the GAP encoding (AF3_TOKENS_MAP['<G>']) 
                and the corresponding full_msa_mask block to 1. Here, max_msa_size is the maximum size of the MSAs 
                across all chains, or 1 if no MSAs are present.
            - Set the first row of the full MSA to the token_features restype, and update the first row 
                of individual_msa_mask accordingly.
            - Plugging the actual MSAs into the full MSA is a bit tricky, because of the fact that some residues 
                in the MSA might be atomized (in which case there are more tokens in the feature 
                than residues in the MSA). In this case, only the first token of the residue 
                (which corresponds to the get_residue_starts(atom_array) of the atom array) should get the MSA values. 
                For the individual MSAs, you can create an integer index and use that to 
                populate msa, deletion_count, and individual_msa_mask, 
                e.g. msa[:msa_size, residue_starts_for_chain_id] = polymer_msas[chain_id]['msa']. 
                The corresponding Atomworks names are polymer_msas[chain_id]['msa'], polymer_msas[chain_id]['ins'], 
                and ~polymer_msas[chain_id]['msa_is_padded_mask'] respectively (note the negation for the latter, 
                msa_is_padded_mask is the negation of the actual mask).
            - Assemble the dict data['msa_features'] with the keys 'msa', 'deletion_count', 'full_msa_mask',
                'individual_msa_mask', and the corresponding values.
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        return data


class AssembleMSAFeatures(Transform):
    def __init__(
        self, msa_trunc_count, n_recycling_iterations, msa_shuffle_orders=None
    ):
        self.msa_trunc_count = msa_trunc_count
        self.msa_shuffle_orders = msa_shuffle_orders
        self.n_recycling_iterations = n_recycling_iterations

    def calculate_target_feat(self, restype, profile, deletion_mean):
        """
        Calculates the target_feat feature for AlphaFold 3.

        Args:
            restype (Array): Array of shape (n_tokens,) containing an integer encoding of the residue types.
            profile (Array): Array of shape (n_tokens, 32), containing the residue profile over the MSA.
            deletion_mean (Array): Array of shape (n_tokens,), containig the mean deletion count over the MSA.

        Returns:
            Array: target_feat of shape (n_tokens, 65), stacking the one-hot encoded restype, 
                the profile, and deletion_mean.
        """

        target_feat = None

        """
        TODO: Implement the calculation of target_feat, by calculating a one-hot encoding of restype 
        (e.g. using F.one_hot, translating from and to numpy) and concatenating it with profile and deletion_mean.
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        return target_feat

    def sample_msa_features(self, base_msa_feat, base_msa_mask):
        """
        Samples a random subset of msa_trunc_count rows from base_msa_feat and base_msa_mask. 
        The subset is sampled uniformly, except for preferring rows that have at least one valid entry.

        Args:
            base_msa_feat (Array): MSA feature of shape (max_msa_sequences, n_tokens, msa_feat_dim)
            base_msa_mask (Array): MSA mask of shape (max_msa_sequences, n_tokens)

        Returns:
            tuple: (msa_feat, msa_mask) of shapes (msa_trunc_count, n_tokens, msa_feat_dim, n_cycle)
                and (msa_trunc_count, n_tokens, n_cycle) respectively, containing the sampled MSA features and masks.
        """

        msa_feat = None
        msa_mask = None

        """
        TODO: Implement the sampling procedure. To match the (random) test-cases, you have to implement the 
          uniform-order strategy as follows:
          - Calculate a 1/0 mask of which rows have at least one valid entry (e.g. using np.sum and np.clip)
          - Sample a random score using torch.distributions.Gumpel(0, 1) and add a large positive constant (e.g. 1e2) 
              to rows without a valid entry. To match  the test results, sample all scores, 
              e.g. for (n_cycle, max_msa_sequences), at once. 
          - argsort the scores in ascending order to get a shuffle order. Truncate that to msa_trunc_count 
              to get the sampled row indices, and use those to create the sampled msa_feat
              and msa_mask (of shape (n_cycle, msa_trunc_count, n_tokens, [msa_feat_dim]))
          - move the n_cycle dimension to the end, for example with np.moveaxis
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        return msa_feat, msa_mask

    def forward(self, data):
        deletion_count = data["msa_features"]["deletion_count"]
        msa = data["msa_features"]["msa"]
        full_msa_mask = data["msa_features"]["full_msa_mask"]
        individual_msa_mask = data["msa_features"]["individual_msa_mask"]
        restype = data["token_features"].restype

        msa_features = None

        """
        TODO: Build the MSAFeatures object, through the following steps:
            - Calculate profile and deletion_mean, using utils.masked_mean (on deletion_count and on 
                a one-hot encoding of msa, with 32 classes). These are based on individual_mask (e.g. the real mask), 
                not full_msa_mask (the larger block mask)
            - Calculate deletion_value as 2/pi arctan(deletion_count / 3) and has_deletion (for example with np.clip)
            - Build a full_msa_feat by concatenating the one-hot encoded MSA, has_deletion, and deletion_value
            - Use sample_msa_feat and calculate_target_feat to build the final msa_feat, msa_mask, and target_feat
            - Assemble the MSAFeatures dataclass.
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        data["msa_features"] = msa_features

        return data


class CalculateMSAFeatures(Transform):
    def __init__(
        self,
        max_msa_sequences,
        msa_trunc_count,
        n_cycle,
        protein_msa_dirs=None,
        rna_msa_dirs=None,
        msa_shuffle_orders=None,
    ):

        transforms = []

        """
        TODO: Assemble the MSA feature pipeline, consisting of the following steps:
          - LoadPolymerMSAs (Atomworks Transform): Loads MSAs for each polymer chain and stores it in the annotation 
              data['polymer_msas_by_chain_id']. Set protein_msa_dirs, rna_msa_dirs, max_msa_sequences, 
                  and put use_paths_in_chain_info=True (we specify the MSA paths 
                  in the chain info in load_alphafold_input).
          - HotfixDuplicateRowIfSingleMSA (Transform defined above): If there is only a single MSA, 
              it duplicates the first row of that, to match AF3's handling.
          - EncodeMSA (Transform defined above): Replaces the atomworks-custom integer encoding of the MSA 
              with the AF3 encoding.
          - ConcatMSAs (Transform defined above): Concatenates the explicit MSAs and puts their token_features.restype 
              for the remaining features.
          - HotfixAF3LigandAsGap (Transform defined above): For ligands, AF3 actually doesn't use their restype token 
              but the GAP token in the MSA (this affects only the first row). This transform does that substitution.
          - BuildMSAAndTargetFeat (Transform defined above): Samples the MSA features for the recycling iterations, 
              and builds the target_feat.
        """

        # Replace 'pass' with your code
        pass

        """ End of your code """

        self.transforms = Compose(transforms)

    def forward(self, data: dict):
        data = self.transforms(data)

        return data
