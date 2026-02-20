import json
import numpy as np
import torch

from common.block_sparse_tensor import BlockSparseTensor
from atomworks.enums import ChainType
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import components_to_atom_array


Array = np.ndarray | torch.Tensor

def pad_to_shape(data: Array , padded_shape, value=0):
    if isinstance(data, np.ndarray):
        padded = np.full(padded_shape, fill_value=value, dtype=data.dtype, device=data.device)
    else:
        padded = torch.full(padded_shape, fill_value=value, dtype=data.dtype, device=data.device)

    inds = tuple(slice(i) for i in data.shape)
    padded[inds] = data

    return padded


def round_down_to(data, rounding_target, return_indices=False):
    sorting_indices = np.argsort(rounding_target)[::-1]
    target_inds = np.argmax(rounding_target[sorting_indices] <= data[..., None], axis=-1)
    target_inds = sorting_indices[target_inds]

    if return_indices:
        return rounding_target[target_inds], target_inds
    else:
        return rounding_target[target_inds]

def round_up_to(data, rounding_target, return_indices=False):
    data = np.array(data)
    sorting_indices = np.argsort(rounding_target)
    target_inds = np.argmax(rounding_target[sorting_indices] >= data[..., None], axis=-1)
    target_inds = sorting_indices[target_inds]

    if return_indices:
        return rounding_target[target_inds], target_inds
    else:
        return rounding_target[target_inds]


def masked_mean(feat: Array, mask: Array, axis, keepdims=False):
    if isinstance(feat, np.ndarray):
        feat_sum = np.sum(feat * mask, axis=axis, keepdims=keepdims)
        count = np.sum(mask, axis=axis, keepdims=keepdims)
        return feat_sum / np.clip(count, a_min=1e-10, a_max=None)
    else:
        feat_sum = (feat*mask).sum(dim=axis, keepdim=keepdims)
        count = mask.sum(dim=axis, keepdim=keepdims)
        return feat_sum / torch.clip(count, min=1e-10)



def unify_batch_dimension(x: torch.Tensor | BlockSparseTensor, batch_shape):
    if isinstance(x, BlockSparseTensor):
        return x

    if len(batch_shape) == 0:
        return x[None, ...]
    else:
        return x.flatten(end_dim=len(batch_shape)-1)


def load_alphafold_input(path):
    with open(path, 'r') as f:
        data = json.load(f)

    example_id = data['name']
    components = []

    for entry in data['sequences']:
        for _type, component in entry.items():
            if isinstance(component['id'], str):
                component['id'] = [component['id']]

            for i, _id in enumerate(component['id']):
                if _type == 'protein':
                    new_component = {
                        'seq': component['sequence'],
                        'chain_type': ChainType.POLYPEPTIDE_L,
                    }
                    if 'unpairedMsaPath' in component:
                        new_component['msa_path'] = component['unpairedMsaPath']
                    components.append(new_component)
                elif _type == 'rna':
                    new_component ={
                        'seq': component['sequence'],
                        'chain_type': ChainType.RNA,
                    }
                    if 'unpairedMsaPath' in component:
                        new_component['msa_path'] = component['unpairedMsaPath']
                    components.append(new_component)
                elif _type == 'dna':
                    new_component = {
                        'seq': component['sequence'],
                        'chain_type': ChainType.DNA,
                    }
                    if 'unpairedMsaPath' in component:
                        new_component['msa_path'] = component['unpairedMsaPath']
                    components.append(new_component)

                elif _type == 'ligand':
                    components.append({
                        'ccd_code': component['ccdCodes'][i]
                    })

    atom_array, components = components_to_atom_array(components, return_components=True)
    atom_array_data = parse_atom_array(atom_array)
    atom_array = atom_array_data['assemblies']['1'][0]
    chain_info = atom_array_data['chain_info']

    for component in components:
        if hasattr(component, 'msa_path'):
            chain_info[component.chain_id]['msa_path'] = component.msa_path

    return {
        "example_id": example_id,
        "atom_array": atom_array,
        "chain_info": chain_info,
    }