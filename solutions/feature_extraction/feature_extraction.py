from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import torch
from atomworks.constants import STANDARD_AA, STANDARD_RNA, STANDARD_DNA

from atomworks.io.utils import ccd
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, Identity, RandomRoute, Transform
from atomworks.ml.transforms.filters import RemoveHydrogens
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3

from common import utils
from config import Config
from feature_extraction.contact_features import CalculateContactMatrix
from feature_extraction.msa_features import CalculateMSAFeatures, MSAFeatures
from feature_extraction.ref_struct_features import (
    CalculateRefStructFeatures,
    RefStructFeatures,
)
from feature_extraction.token_features import CalculateTokenFeatures, TokenFeatures


Array = np.ndarray | torch.Tensor


@dataclass
class Batch:
    """
    Represents the input features for an AlphaFold 3 pass.
    """

    token_features: TokenFeatures
    msa_features: MSAFeatures
    ref_struct: RefStructFeatures
    contact_matrix: Array


def tree_map(fn, x):
    """
    Recursively applies a function to all elements in a nested structure of dataclasses, dicts, and tensors / arrays.

    Args:
        fn: A function that takes a tensor or array and returns a tensor or array.
        x: The input data, which can be a tensor, array, dataclass, or nested structure of dataclasses.
    """
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return fn(x)

    if is_dataclass(x):
        field_dict = {f.name: tree_map(fn, getattr(x, f.name)) for f in fields(x)}
        return type(x)(**field_dict)

    if isinstance(x, dict):
        field_dict = {k: tree_map(fn, v) for k, v in x.items()}
        return field_dict

    raise ValueError(f"Cannot apply tree_map to object of type {type(x)}")


# different name for drop_unconvertibles:
def collate_batch(
    batch: list[dataclass] | list[torch.Tensor] | list[dict],
    drop_unconvertible_entries=False,
) -> dataclass:
    """
    Recursively collates a list of nested structures of dataclasses, dicts, and tensors into a single nestex structure
    where the tensors are stacked along a new batch dimension. The tensors in the batch are padded to the maximum shape
    along each dimension to ensure they can be stacked together.

    Args:
        batch: List of objects to collate.
    """
    first = batch[0]
    if isinstance(first, torch.Tensor):
        max_shape = np.array([b.shape for b in batch]).max(axis=0).tolist()
        padded_batch = [utils.pad_to_shape(b, max_shape) for b in batch]
        return torch.stack(padded_batch)

    if isinstance(first, np.ndarray):
        max_shape = np.array([b.shape for b in batch]).max(axis=0).tolist()
        padded_batch = [utils.pad_to_shape(b, max_shape) for b in batch]
        return np.stack(padded_batch)

    if is_dataclass(first):
        field_dict = {
            f.name: collate_batch(
                [getattr(b, f.name) for b in batch],
                drop_unconvertible_entries=drop_unconvertible_entries,
            )
            for f in fields(first)
        }
        return type(first)(**field_dict)

    if isinstance(first, dict):
        field_dict = {
            k: collate_batch(
                [b[k] for b in batch],
                drop_unconvertible_entries=drop_unconvertible_entries,
            )
            for k in first.keys()
        }
        return field_dict

    if not drop_unconvertible_entries:
        raise ValueError(f"Cannot collate batch of type {type(first)}")
    else:
        return None


class HotfixDropSaccharideO1(Transform):
    """
    Drops all atoms named 'O1' in residues that are classified as D-Saccharides according to the CCD.
    This processing step is performed in AlphaFold 3 and is necessary to match their input features 
    for correct inference with AF3 model parameters.
    """

    def forward(self, data):
        atom_array = data["atom_array"]
        res_names = np.unique(atom_array.res_name)
        saccharide_res_names = [
            res_name
            for res_name in res_names
            if "D-SACCHARIDE" in ccd.get_chem_comp_type(res_name)
        ]
        mask = np.isin(atom_array.res_name, saccharide_res_names) & (
            atom_array.atom_name == "O1"
        )
        data["atom_array"] = atom_array[~mask]

        return data


class BuildBatch(Transform):
    """Assembles the Batch dataclass from the individual features in the data dict."""

    def forward(self, data):
        batch = Batch(
            token_features=data["token_features"],
            msa_features=data["msa_features"],
            ref_struct=data["ref_struct"],
            contact_matrix=data["contact_matrix"],
        )

        batch = tree_map(lambda x: torch.tensor(x), batch)

        return {"batch": batch}


def custom_af3_pipeline(
    config: Config, msa_shuffle_orders=None, is_inference=True
) -> Transform:
    """
    Creates an AlphaFold 3 feature extraction pipeline consisting of the following steps:
    1. Remove hydrogens
    2. Drop 'O1' atoms in D-Saccharides
    3. Atomize by CCD name (atomizing all residues by default, except if they are amino acids, RNA, or DNA)
    4. Build Token Features, Reference Structure Features, MSA Features, and Contact Matrix

    Args:
        config: Config object for the whole AlphaFold model. In particular, the data pipeline needs to be aware of
            the number of recycling iterations, and choices for the msa truncation per iteration.
        msa_shuffle_orders: An optional tensor of shape (n_recycling_iterations, n_msa_sequences) containing
            a deterministic shuffle order for the MSA sequences. If not provided, random shuffling will be performed
            for each recycling iteration. This can ensure reproducibility for testing.
        is_inference: Whether the pipeline is used for inference or training.
            For training, an additional random cropping step is added after atomization.
    """
    max_msa_sequences = config.featurization_config.max_msa_sequences
    msa_trunc_count = config.featurization_config.msa_trunc_count
    n_cycle = config.global_config.n_cycle

    transforms = []

    """
    TODO: Initialize a list of the transforms RemoveHydrogens, HotfixDropSaccharideO1, AddGlobalAtomIdAnnotation, 
    AtomizeByCCDName (atomizing by default, except if they are part of STANDARD_AA, STANDARD_RNA, or STANDARD_DNA), 
    and the four AF3-specific transforms CalculateTokenFeatures, CalculateRefStructFeatures, CalculateMSAFeatures, 
    and CalculateContactMatrix (in this order), which we implement in the other files. 
    Finally, append BuildBatch to assemble the dict into a Batch object. The transforms fulfill the following tasks:
    - RemoveHydrogens: Removes all hydrogen atoms from the input.
    - HotfixDropSaccharideO1: Drops all atoms named 'O1', which is necessary to match AF3's input features.
    - AddGlobalAtomIdAnnotation: Adds range(N) as a global atom id annotation. We don't need that, but ourselves, 
        but the Atomworks Cropping transforms for training require the transform. 
    - AtomizeByCCDName: Adds an 'atomize' flag to the input AtomArray, that is true for atoms belonging to ligands 
        or modified residues. We use this flag in ConcatMSAs and CalculateContactMatrix. 
    - our custom transforms construct the actual features based on the AtomArray.

    Note for when you are doing Chapter Training: Also add a train_transform that is a RandomRoute 
    between CropContiguousLikeAF3 and CropSpatialLikeAF3 with 0.5 probability each, 
    cropping to a crop_size of 384. If is_inference is False, add this after the atomization step. 
    You don't need to do this for the feature extraction part.
    """

    # Note: You will only implement cropping in the very last step, you don't need to implement it here 
    # during the feature extraction part.
    train_transform = RandomRoute(
        [
            CropContiguousLikeAF3(crop_size=384),
            CropSpatialLikeAF3(crop_size=384),
        ],
        probs=[0.5, 0.5],
    )

    transforms = [
        RemoveHydrogens(),
        HotfixDropSaccharideO1(),
        AddGlobalAtomIdAnnotation(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
        ),
        # Cropping
        Identity() if is_inference else train_transform,
        CalculateTokenFeatures(),
        CalculateRefStructFeatures(),
        CalculateMSAFeatures(
            max_msa_sequences,
            msa_trunc_count,
            n_cycle,
            msa_shuffle_orders=msa_shuffle_orders,
        ),
        CalculateContactMatrix(),
        BuildBatch(),
    ]

    """ End of your code """

    return Compose(transforms)
