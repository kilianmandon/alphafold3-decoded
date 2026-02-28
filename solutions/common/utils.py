import json
import math
import numpy as np
import torch

from common.block_sparse_tensor import BlockSparseTensor
from atomworks.enums import ChainType
from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.inference import components_to_atom_array


Array = np.ndarray | torch.Tensor

def pad_to_shape(data: Array , padded_shape, value=0):
    """
    Pads the data to the specified shape, using the specified value for padding.
    """

    padded = None
    is_numpy = isinstance(data, np.ndarray)
    data = torch.as_tensor(data)

    """
    TODO: Implement  the padding. In the output, the data should be in the top-left corner 
    of the padded array (not in the center, as with image padding). 
    You can create the padded array using torch.full, and then fill in the value by creating an index of the form
        (slice(0, dim_1), slice(0, dim_2), ...), 
    then setting the data into the padded array using this index. If you are not familiar 
    with this way of indexing, a tuple of slices is exactly what indexing of the form
        arr[:dim_1, :dim_2, ...] 
    creates internally. Using the function slice(...) explicitly is helpful when you want to create 
    indices programmatically, for example if you don't know the number of dimensions in advance. 
    Make sure you copy the dtype and device of the input data when creating the padded array!
    """

    padded = torch.full(padded_shape, fill_value=value, dtype=data.dtype, device=data.device)
    inds = tuple(slice(i) for i in data.shape)
    padded[inds] = data

    """ End of your code """

    if is_numpy:
        return padded.numpy()
    else:
        return padded


def round_down_to(data: np.ndarray, rounding_target: np.ndarray, return_indices=False):
    """
    Rounds the data values down to the closest value in rounding_target, optionally returning 
    the indices of the selected values.

    Args:
        data: numpy array of any shape
        rounding_target: Sorted 1D numpy array of values to round down to
    """

    rounding_results = None
    rounding_inds = None

    """
    TODO: Implement the rounding procedure. You can unsqueeze data and do a broadcasted comparison to find 
    which valus in rounding_target are smaller or equal to the values in data. We want the last value that 
    satisfies this, or the first in the reversed rounding_target. To find the first index that fulfills 
    this condition, you can use np.argmax on the broadcasted dimension, then calculate back to the index 
    in the original array. These indices can be used as an integer index to get the rounded values 
    from rounding_target. 
    """

    rounding_inds = np.argmax(rounding_target[::-1] <= data[..., None], axis=-1)
    rounding_inds = rounding_target.shape[0] - 1 - rounding_inds
    rounding_results = rounding_target[rounding_inds]

    """ End of your code """

    if return_indices:
        return rounding_results, rounding_inds
    else:
        return rounding_results


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




def masked_mean(feat: Array, mask: Array, axis, keepdims=False):
    is_numpy = isinstance(feat, np.ndarray)
    feat = torch.as_tensor(feat)
    mask = torch.as_tensor(mask)

    result = None

    """
    TODO: Implement the masked mean function. This calculates the mean of feat along axis, but not dividing by the shape 
    of the axis, but by the number of valid (non-masked) entries along that axis. If this number is zero, divide by 1e-10 instead.
    """

    feat_sum = (feat*mask).sum(dim=axis, keepdim=keepdims)
    count = mask.sum(dim=axis, keepdim=keepdims)
    result =  feat_sum / torch.clip(count, min=1e-10)

    """ End of your code """

    if is_numpy:
        return result.numpy()
    else:
        return result


def rand_rot(batch_shape: tuple, device: torch.device):
    q = None

    """
    TODO: Sample random rotation matrices of shape (**batch_shape, 3, 3). To do this, you can sample a random normal matrix, 
    do a QR decomposition with torch.linalg.qr, then fix the sign of the determinant: Rotation matrices have determinant 1. 
    You can compute the determinant of q with torch.linalg.det(q). For 3x3 matrices, we have det(-A) = -det(A), so we can flip 
    the sign of q's determinant by flipping the sign of all values in q.
    """

    rand_matrices = torch.randn(batch_shape + (3, 3), device=device)
    q, _ = torch.linalg.qr(rand_matrices)
    q = q * torch.linalg.det(q).sign()[..., None, None]

    """ End of your code """

    return q



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