import os
# Set so that Atomworks does not raise a warning, we don't need to actually download the mirrors for this notebook.
os.environ["PDB_MIRROR_PATH"] = ""
os.environ["CCD_MIRROR_PATH"] = ""

from torch_snapkit import memory_snapshot

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
    module =  AF3TrainingModule(model, config, num_devices=1)

    with FlopCounterMode(display=True):
        module.training_step(samples[0], 0)


if __name__=='__main__':
    with torch.autograd.detect_anomaly(), memory_snapshot('debug_flop', share=True, share_code='daily-secret-05', log_shapes=True):
        main()