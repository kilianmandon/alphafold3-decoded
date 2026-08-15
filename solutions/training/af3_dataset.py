from pathlib import Path

import numpy as np
import torch
from atomworks import parse
from atomworks.ml.datasets import ConcatDatasetWithID, PandasDataset
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units
from atomworks.ml.samplers import calculate_af3_example_weights, get_cluster_sizes
from torch.utils.data import WeightedRandomSampler

from config import Config
from feature_extraction.feature_extraction import collate_batch, custom_af3_pipeline


def af3_pipeline_none_on_error(config, is_inference=False):
    pipeline = custom_af3_pipeline(config, is_inference=is_inference)
    def apply(*args, **kwargs):
        try:
            return pipeline(*args, **kwargs)
        except Exception as e:
            print('Skipping entry due to error')
            print(e)
            return None
    return apply

def collate_batch_drop_none(batch, config: Config):
    if any(b is None for b in batch):
        none_count = len([b for b in batch if b is None])
        print(f'Dropping {none_count} out of {len(batch)} entries because of errors in feature extraction.')
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
    collated = collate_batch(batch)

    return collated


def build_eval_dataset(config: Config, samples_per_group=8, is_inference=True):
    np.random.seed(23)
    # add_filters = ["deposition_date >= '2022-01-01'"]
    add_filters = ["deposition_date < '2022-01-01'"]
    print('Loading pn unit ds...')
    single_pn_unit_ds = build_single_pn_units_dataset(config, is_inference=is_inference, additional_filters=add_filters)
    
    print('Loading interfaces ds...')
    binary_interfaces_ds = build_binary_interfaces_dataset(config, is_inference=is_inference, additional_filters=add_filters)
    print('Done.')

    single_pn_unit_ds.data = single_pn_unit_ds.data[single_pn_unit_ds.data['total_num_atoms_in_unprocessed_assembly'] < 3000]
    binary_interfaces_ds.data = binary_interfaces_ds.data[binary_interfaces_ds.data['total_num_atoms_in_unprocessed_assembly'] < 3000]

    for ds in [single_pn_unit_ds, binary_interfaces_ds]:
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

    weights_pn_units = calculate_af3_example_weights(single_pn_unit_ds.data, alphas, beta_chain)
    weights_interfaces = calculate_af3_example_weights(binary_interfaces_ds.data, alphas, beta_interface)

    weights_pn_units = weights_pn_units / np.sum(weights_pn_units)
    weights_interfaces = weights_interfaces / np.sum(weights_interfaces)

    pn_units_inds = np.random.choice(np.arange(weights_pn_units.shape[0]), replace=False, p=weights_pn_units, size=samples_per_group)
    interfaces_inds = np.random.choice(np.arange(weights_interfaces.shape[0]), replace=False, p=weights_interfaces, size=samples_per_group)
    single_pn_unit_ds.data = single_pn_unit_ds.data.iloc[pn_units_inds]
    binary_interfaces_ds.data = binary_interfaces_ds.data.iloc[interfaces_inds]

    dataset = ConcatDatasetWithID([single_pn_unit_ds, binary_interfaces_ds])
    return dataset
    

def build_single_pn_units_dataset(config: Config, is_inference, additional_filters=None):
    if additional_filters is None:
        additional_filters = []
    return PandasDataset(
        name='pn_units',
        id_column='example_id',
        data='data/datasets/pdb_metadata/pn_units_df_top1000.parquet',
        loader=create_loader_with_query_pn_units(pn_unit_iid_colnames='q_pn_unit_iid', base_path='data/datasets/pdb_mirror',
                                                    extension='.cif.gz', sharding_pattern='/1:3/', path_colname='pdb_id'),
        filters=(additional_filters + [
            "resolution < 5.0 and ~method.str.contains('NMR')",
            "num_polymer_pn_units <= 20",
            "cluster.notnull()",
            "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
            # Train only on D-polypeptides:
            "q_pn_unit_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            # Exclude ligands from AF3 excluded set:
            "~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))",
        ]),
        transform=af3_pipeline_none_on_error(config, is_inference=is_inference),
        save_failed_examples_to_dir=None
    )

def build_binary_interfaces_dataset(config: Config, is_inference, additional_filters=None):
    if additional_filters is None:
        additional_filters = []

    return PandasDataset(
        name='interfaces',
        id_column='example_id',
        data=Path('data/datasets/pdb_metadata/interfaces_df_top1000.parquet'),
        loader=create_loader_with_query_pn_units(pn_unit_iid_colnames=["pn_unit_1_iid", "pn_unit_2_iid"],
                                                    base_path='data/datasets/pdb_mirror', extension='.cif.gz',
                                                    sharding_pattern='/1:3/', path_colname='pdb_id'),
        transform=af3_pipeline_none_on_error(config, is_inference=is_inference),
        filters=(additional_filters + [
            "resolution < 5.0 and ~method.str.contains('NMR')",
            "num_polymer_pn_units <= 20",
            "cluster.notnull()",
            "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
            # Train only on D-polypeptide interfaces:
            "pn_unit_1_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            "pn_unit_2_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            "~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))",
            "~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('${af3_excluded_ligands_regex}', regex=True))"
        ]),
        save_failed_examples_to_dir=None
    )




    

def build_af3_dataset(config: Config):
    add_filters = ["deposition_date < '2022-01-01'"]
    datasets = [
        build_single_pn_units_dataset(config, is_inference=False, additional_filters=add_filters),
        build_binary_interfaces_dataset(config, is_inference=False, additional_filters=add_filters),
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
    eval_ds = build_eval_dataset(config)


if __name__ == '__main__':
    main()
