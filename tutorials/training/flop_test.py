import os
import types

import numpy as np

from feature_extraction.reference_features import ReferenceFeatures
# Set so that Atomworks does not raise a warning, we don't need to actually download the mirrors for this notebook.
os.environ["PDB_MIRROR_PATH"] = ""
os.environ["CCD_MIRROR_PATH"] = ""

from torch_snapkit import memory_snapshot
from torch._subclasses.fake_tensor import FakeTensorMode

from feature_extraction.feature_extraction import tree_map
from common.modules import AttentionPairBias
from evoformer.evoformer import TriangleAttention
from training.af3_dataset import build_af3_dataset, build_sampler, collate_batch_drop_none
from training.training_module import AF3TrainingModule
from training.debug_flex_attention import debug_flex_attention

import pickle

from torch.utils.flop_counter import FlopCounterMode

from config import Config
from diffusion.model import Model
import torch

def fake_to_token_layout(ref_features, feature):
    batch_shape = ref_features.mask.shape[:-1]
    token_count = ref_features.atom_count // 24
    feature_shape = feature.shape[len(batch_shape)+1:]
    out_shape = batch_shape + (token_count, 24) + feature_shape
    return feature.reshape(out_shape)

def fake_to_atom_layout(ref_features, feature, has_atom_dimension=True):

    batch_shape = ref_features.element.shape[:-1]
    feature = torch.as_tensor(feature)

    if not has_atom_dimension:
        feature = ref_features.patch_atom_dimension(feature)

    out_shape = batch_shape + (ref_features.atom_count,) + feature.shape[len(batch_shape)+2:]
    out = feature.reshape(out_shape)

    if isinstance(ref_features.mask, np.ndarray):
        return out.numpy()
    else:   
        return out


def main():
    config = Config()
    config.global_config.n_cycle = 1
    model = Model(config)

    for module in model.modules():
        if isinstance(module, AttentionPairBias) or isinstance(module, TriangleAttention):
            module.flex_attention = debug_flex_attention


    # train_ds = build_af3_dataset(config)
    # sampler = build_sampler(train_ds)
    # train_dl = torch.utils.data.DataLoader(train_ds, num_workers=15, batch_size=1, sampler=sampler, collate_fn=lambda x: collate_batch_drop_none(x, config))

    # samples = [sample for _, sample in zip(range(10), train_dl)]

    # with open('test_samples.pkl', 'wb') as f:
    #     pickle.dump(samples, f)

    with open('test_samples.pkl', 'rb') as f:
        samples = pickle.load(f)


    device = 'cuda:0'
    model.to(device)
    samples = tree_map(lambda x: x.to(device), samples, skip_unconvertible_entries=True)
    batch_with_labels = samples[0]
    ReferenceFeatures.to_token_layout = fake_to_token_layout
    ReferenceFeatures.to_atom_layout = fake_to_atom_layout
    module =  AF3TrainingModule(model, config, num_devices=1)

    with FlopCounterMode(display=True), FakeTensorMode(allow_non_fake_inputs=True):
        # module._shared_step(samples[0], 0, stage='val')
        module.training_step(batch_with_labels, 0)


    # print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    # print(torch.cuda.memory.max_memory_allocated() / 1e9, "GB peak allocated")
    # print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
    # print(torch.cuda.max_memory_reserved() / 1e9, "GB peak reserved")


if __name__=='__main__':
    # with torch.autograd.detect_anomaly(), memory_snapshot('debug_flop', share=True, share_code='daily-secret-05', log_shapes=True):
    # with memory_snapshot('debug_flop', share=True, share_code='daily-secret-05', log_shapes=True):
    # with torch.no_grad():
    main()
