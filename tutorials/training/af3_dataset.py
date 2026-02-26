import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from atomworks import parse
from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.utils.visualize import view
from atomworks.ml.datasets import FileDataset, StructuralDatasetWrapper, ConcatDatasetWithID, PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.datasets.parsers import PNUnitsDFParser, InterfacesDFParser
from atomworks.ml.pipelines.af3 import build_af3_transform_pipeline
from atomworks.ml.samplers import calculate_af3_example_weights, get_cluster_sizes
from atomworks.ml.transforms.atom_array import AddGlobalAtomIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.crop import CropSpatialLikeAF3
from torch.utils.data import WeightedRandomSampler

from common.utils import load_alphafold_input
from config import Config
from feature_extraction.feature_extraction import collate_batch, custom_af3_pipeline


def simple_loading_fn(raw_data):
    parse_output = parse(raw_data)
    return {'atom_array': parse_output['assemblies']['1'][0]}

def build_af3_dataset(config: Config):
    datasets = [
        # Single PN units
        PandasDataset(
            name='pn_units',
            id_column='example_id',
            data='data/datasets/pdb_metadata/pn_units_df_top1000.parquet',
            loader=create_loader_with_query_pn_units(pn_unit_iid_colnames='q_pn_unit_iid', base_path='data/datasets/pdb_mirror',
                                                     extension='.cif.gz', sharding_pattern='/1:3/', path_colname='pdb_id'),
            filters=[
                "deposition_date < '2022-01-01'",
                "resolution < 5.0 and ~method.str.contains('NMR')",
                "num_polymer_pn_units <= 20",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
                # Train only on D-polypeptides:
                "q_pn_unit_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
                # Exclude ligands from AF3 excluded set:
                "~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))",
            ],
            transform=custom_af3_pipeline(config, is_inference=False),
            save_failed_examples_to_dir=None
        ),
        # Binary interfaces
        PandasDataset(
            name='interfaces',
            id_column='example_id',
            data=Path('data/datasets/pdb_metadata/interfaces_df_top1000.parquet'),
            loader=create_loader_with_query_pn_units(pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"],
                                                     base_path='data/datasets/pdb_mirror', extension='.cif.gz',
                                                     sharding_pattern='/1:3/', path_colname='pdb_id'),
            transform=custom_af3_pipeline(config, is_inference=False),
            filters=[
                "deposition_date < '2022-01-01'",
                "resolution < 5.0 and ~method.str.contains('NMR')",
                "num_polymer_pn_units <= 20",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
                # Train only on D-polypeptide interfaces:
                "pn_unit_1_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
                "pn_unit_2_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
                "~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))",
                "~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))"
            ],
            save_failed_examples_to_dir=None
        )
    ]

    af3_pdb_dataset = ConcatDatasetWithID(datasets)
    return af3_pdb_dataset


def build_sampler(dataset):
    for ds in dataset.datasets:
        cluster_id_to_size_map = get_cluster_sizes(ds.data, cluster_column='cluster')
        ds.data['cluster_size'] = ds.data['cluster'].map(cluster_id_to_size_map)

    alphas = {
        "a_prot": 3,
        # Choosing same as for protein,
        # even though atomworks says peptides were oversampled in AF3
        "a_peptide": 3,
        "a_nuc": 3,
        "a_ligand": 1,
        "a_loi": 0
    }
    beta_chain = 0.5
    beta_interface = 1

    weights_chains = calculate_af3_example_weights(dataset.datasets[0].data, alphas, beta_chain)
    weights_interfaces = calculate_af3_example_weights(dataset.datasets[1].data, alphas, beta_interface)
    weights = np.concatenate([weights_chains.to_numpy(), weights_interfaces.to_numpy()])

    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler



def main():
    config = Config()
    dataset = build_af3_dataset(config)
    sampler = build_sampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4, collate_fn=collate_batch)
    samples = next(iter(loader))
    ...


if __name__ == '__main__':
    main()
