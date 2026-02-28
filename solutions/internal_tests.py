import time
import torch
from common.utils import load_alphafold_input, rand_rot
from config import Config
from feature_extraction.feature_extraction import Batch, custom_af3_pipeline, tree_map, collate_batch
from diffusion.model import Model
import tensortrace as ttr
import os
# Set so that Atomworks does not raise a warning, we don't need to actually download the mirrors for this notebook.
os.environ["PDB_MIRROR_PATH"] = ""
os.environ["CCD_MIRROR_PATH"] = ""


def reorder_encoding(dim=-1, offset=0):
    token_enc_shift = {
                          i: i for i in range(31)
                      } | {
                          23: 24,
                          24: 23,
                          26: 27,
                          27: 29,
                          28: 28,
                          29: 30,
                          30: 26,
                      }
    def f(tensor):
        new_shape = list(tensor.shape)
        new_shape[dim] -= 1
        new_tensor = torch.zeros(new_shape, device=tensor.device,  dtype=tensor.dtype)
        new_tensor[:, :offset] = tensor[:, :offset]
        new_tensor[:, offset+31:] = tensor[:, offset+32:]
        for i_old, i_new in token_enc_shift.items():
            new_tensor[:, offset+i_old] = tensor[:, offset+i_new]
        return new_tensor
    
    return f

def to_float(tensor):
    return tensor.float()


def main(test_name):
    msa_shuffle_order = torch.stack(ttr.load_all('evoformer/msa_shuffle_order'), dim=0)
    config = Config()
    config.global_config.n_cycle = 2
    config.diffusion_config.denoising_steps = 4

    model = Model(config)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)

    t1 = time.time()
    data = load_alphafold_input(f'data/fold_inputs/fold_input_{test_name}.json')
    transform = custom_af3_pipeline(config, msa_shuffle_orders=msa_shuffle_order)

    data = transform.forward(data)
    batch = data['batch']

    hotfix_roll_parts = []

    if test_name == 'protein_dna_ion':
        hotfix_roll_parts = [slice(301, 326), slice(327, 352)]
    elif test_name == 'protein_rna_ion':
        hotfix_roll_parts = [slice(1, 75)]

    def hotfix_roll(x, shifts=1):
        x = x.clone()
        for part in hotfix_roll_parts:
            x[part] = torch.roll(x[part], shifts=shifts, dims=1)
        return x

    def hotfix_roll_inv(x):
        return hotfix_roll(x, shifts=-1)

    debug_positions = ttr.load('ref_structure/positions')
    debug_positions = hotfix_roll(debug_positions, shifts=-1)

    batch.ref_struct.positions = batch.ref_struct.to_atom_layout(debug_positions)

    print(f'Featurization took {time.time()-t1:.1f} seconds.')



    ttr.compare({
        'mask': batch.ref_struct.mask,
        'charge': batch.ref_struct.charge,
        'element': batch.ref_struct.element,
        'atom_name_chars': batch.ref_struct.atom_name_chars,
        'ref_space_uid': batch.ref_struct.ref_space_uid,
    }, 'ref_structure', use_mask={'mask': False }, input_processing=[lambda x: batch.ref_struct.to_token_layout(x), hotfix_roll])

    token_feats = {
        'asym_id': batch.token_features.asym_id,
        'sym_id': batch.token_features.sym_id,
        'entity_id': batch.token_features.entity_id,
        'is_dna': batch.token_features.is_dna,
        'is_rna': batch.token_features.is_rna,
        'is_protein': batch.token_features.is_protein,
        'residue_index': batch.token_features.residue_index,
        'token_index': batch.token_features.token_index,
        'mask': batch.token_features.mask,
    }
    ttr.compare(token_feats, 'token_features', use_mask={'mask': False})

    ttr.compare(batch.msa_features.target_feat, 'input_embedding/target_feat', input_processing=[reorder_encoding(offset=32), reorder_encoding(offset=0)])
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch: Batch = tree_map(lambda x: x.to(device=device), batch)

    model = model.to(device=device)
    model.eval()

    s_input, s_trunk, z_trunk, rel_enc = model.evoformer(batch)
    ttr.compare(s_trunk, 'evoformer/single')
    ttr.compare(z_trunk, 'evoformer/pair')


    def t2q(tensor):
        return batch.ref_struct.to_atom_layout(tensor)

    def indexing(*args):
        def apply_index(tensor):
            return tensor[*args]
        return apply_index
    
    def to_device(tensor):
        return tensor.to(device)

    def to_float(tensor):
        return tensor.float()


    diffusion_randomness = {
        'init_pos': ttr.load('diffusion/initial_positions', processing=[to_device, indexing(0), hotfix_roll_inv, t2q, to_float]),
        'noise': ttr.load_all('diffusion/noise', processing=[to_device, indexing(0), hotfix_roll_inv, t2q, to_float]),
        'aug_rot': ttr.load_all('diffusion/rand_aug/rot', processing=[indexing(0), to_device, to_float]),
        'aug_trans': ttr.load_all('diffusion/rand_aug/trans', processing=[indexing(0), to_device, to_float]),
    }

    with ttr.Chapter('diffusion'):
        diff_x = model.diffusion_sampler(model.diffusion_module,
                                s_input, s_trunk, z_trunk, rel_enc, 
                                batch, noise_data=diffusion_randomness)

    diff_x = batch.ref_struct.to_token_layout(diff_x)

    ttr.compare(diff_x, 'diffusion/final_positions', processing=[indexing(0), hotfix_roll_inv])

def batch_test():
    test1 = 'lysozyme'
    test2 = 'multimer'

    config = Config()
    config.global_config.n_cycle = 2
    config.diffusion_config.denoising_steps = 4

    model = Model(config)
    params = torch.load('data/params/af3_pytorch.pt')
    model.load_state_dict(params)

    t1 = time.time()
    transform = custom_af3_pipeline(config)

    data1 = load_alphafold_input(f'data/fold_inputs/fold_input_{test1}.json')
    data2 = load_alphafold_input(f'data/fold_inputs/fold_input_{test2}.json')
    torch.cuda.nvtx.range_push('Featurization 1')
    b1 = transform.forward(data1)['batch']
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push('Featurization 2')
    b2 = transform.forward(data2)['batch']
    torch.cuda.nvtx.range_pop()

    def gen_diff_noise(batch: Batch):
        diffusion_randomness = {
            'init_pos': torch.randn(batch.ref_struct.atom_count, 3),
            'noise': torch.randn(config.diffusion_config.denoising_steps, batch.ref_struct.atom_count, 3),
            'aug_rot': rand_rot((config.diffusion_config.denoising_steps,), device=batch.ref_struct.positions.device),
            'aug_trans': torch.randn(config.diffusion_config.denoising_steps, 3),
        }

        return diffusion_randomness

    n1 = gen_diff_noise(b1)
    n2 = gen_diff_noise(b2)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    b1: Batch = tree_map(lambda x: x.to(device=device), b1)
    b2: Batch = tree_map(lambda x: x.to(device=device), b2)
    n1 = tree_map(lambda x: x.to(device=device), n1)
    n2 = tree_map(lambda x: x.to(device=device), n2)

    model = model.to(device=device)
    model.eval()

    def model_fwd(model, data, noise):
        s_input, s_trunk, z_trunk, rel_enc = model.evoformer(data)
        out = model.diffusion_sampler(model.diffusion_module,
                                s_input, s_trunk, z_trunk, rel_enc, 
                                data, noise_data=noise)
        return out

    torch.cuda.memory._record_memory_history(
       max_entries=100000
   )

    b_joined = collate_batch([b1, b2])
    noise_joined = collate_batch([n1, n2])
    # n11 = tree_map(lambda x: x[0], noise_joined)
    noise_joined['noise'] = torch.moveaxis(noise_joined['noise'], 0, 1)
    noise_joined['aug_rot'] = torch.moveaxis(noise_joined['aug_rot'], 0, 1)
    noise_joined['aug_trans'] = torch.moveaxis(noise_joined['aug_trans'], 0, 1)

    out1 = model_fwd(model, b1, n1)
    b11 = tree_map(lambda x: x[0], collate_batch([b1, b2]))
    # out11 = model_fwd(model, b11, n11)

    out2 = model_fwd(model, b2, n2)


    out_joined = model_fwd(model, b_joined, noise_joined)

    try:
        torch.cuda.memory._dump_snapshot(f"mem_profile.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history(enabled=None)

    out_single_joined = collate_batch([out1, out2])
    mask = b_joined.ref_struct.mask

    da = torch.abs(out_joined - out_single_joined)
    da = da * mask.unsqueeze(-1)
    print(da.max())
    ...




if __name__=='__main__':
    # test_name = 'lysozyme'
    # with torch.no_grad(), ttr.TensorTrace(f'data/tensortraces/{test_name}_trace', mode='read', framework='pytorch'):
    #     main(test_name)
    with torch.no_grad():
        batch_test()