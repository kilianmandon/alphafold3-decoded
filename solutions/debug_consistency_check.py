#!/usr/bin/env python
"""
frozen_consistency_check.py -- the previous consistency check re-ran the (stochastic)
feature pipeline, so it was never looking at the same inputs the model memorised. This
version reads `full_train_ds.pkl` directly: the same frozen, already-transformed samples
the training loop and GenerationProbe use, with no transform applied.

It is self-contained on purpose -- no imports from the earlier debug scripts, and the
model config overrides are copied verbatim from torchrun_training.py so the checkpoint
loads without a shape trap.

What it answers:
  1. does a single denoiser call on the frozen sample reproduce probe/generate ~0.6 A?
  2. what happens above sigma=45.7, which the probe never tested and the sampler spends
     its first half in?
  3. does the full 30-step sampler reach the same place as the single call?
  4. where in the trajectory does it diverge, if it does?
  5. side by side: frozen features vs a fresh draw from the live pipeline

Run:
    python -m solutions.frozen_consistency_check --ckpt checkpoints/step_4999.pt
    python -m solutions.frozen_consistency_check --elem 0 --fresh-pickle eval_ds.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from common import utils
from config import Config
from diffusion.model import Model
from feature_extraction.feature_extraction import collate_batch, tree_map
from training.af3_dataset import af3_pipeline_none_on_error, collate_batch_drop_none
from training.training_module import weighted_align


# ======================================================================================
# config -- copied from torchrun_training.py so the checkpoint shapes match
# ======================================================================================


def build_config(n_cycle=1):
    config = Config()
    config.global_config.n_cycle = n_cycle

    config.global_config.c_s = 64
    config.global_config.c_z = 32
    config.global_config.c_m = 32

    config.evoformer_config.msa_module_config.n_blocks = 1
    config.evoformer_config.template_module_config.n_blocks = 1
    config.evoformer_config.pairformer_config.n_blocks = 8
    config.evoformer_config.pairformer_config.n_head_pairstack = 2
    config.evoformer_config.msa_module_config.n_head_pairstack = 2
    config.evoformer_config.pairformer_config.n_head_att_pair_bias = 2
    config.diffusion_config.n_head_diffusion_transformer = 2

    config.diffusion_config.n_block_diffusion_transformer = 4
    config.diffusion_config.atom_attention_config.c_token = 32
    config.diffusion_config.atom_attention_config.n_block_atom_transformer = 1

    config.training_config.batch_size = 4
    config.training_config.micro_batch_size = 2
    return config


def load_model(config, ckpt, device):
    model = Model(config)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt}")
    if missing:
        print(f"[ckpt] !! {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[ckpt] !! {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    if not missing and not unexpected:
        print("[ckpt] state dict matched exactly")
    print(f"[ckpt] {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model.to(device).eval()


# ======================================================================================
# metrics
# ======================================================================================


def aligned_rmsd(x_pred, x_gt, mask):
    w = mask.float()
    if w.sum() < 3:
        return float("nan")
    with torch.autocast(device_type="cuda", enabled=False):
        gt_al = weighted_align(x_gt.float(), x_pred.float(), w)
    d2 = ((x_pred.float() - gt_al) ** 2).sum(-1)
    return torch.sqrt((d2 * w).sum() / w.sum()).item()


def rg(x, mask):
    w = mask.float()[..., None]
    if w.sum() < 1:
        return float("nan")
    c = (x * w).sum(-2, keepdim=True) / w.sum(-2, keepdim=True)
    return torch.sqrt(
        (((x - c) ** 2).sum(-1) * mask.float()).sum() / mask.float().sum()
    ).item()


def lddt_ca(x_pred, x_gt, mask, cutoff=15.0):
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() < 4:
        return float("nan")
    p, g = x_pred[idx].float(), x_gt[idx].float()
    dp, dg = torch.cdist(p, p), torch.cdist(g, g)
    n = idx.numel()
    pair = (dg < cutoff) & ~torch.eye(n, dtype=torch.bool, device=p.device)
    if pair.sum() == 0:
        return float("nan")
    d = (dp - dg).abs()
    return (sum((d < t).float() for t in (0.5, 1.0, 2.0, 4.0)) / 4.0)[pair].mean().item()


def table(rows, headers):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(x) for h, x in zip(headers, w))
    return "\n".join([line, "-" * len(line)] + ["  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows])


def fmt(v, n=2):
    return "  -  " if v is None or v != v else f"{v:.{n}f}"


# ======================================================================================
# a prepared sample, built from ALREADY-TRANSFORMED items (no pipeline call)
# ======================================================================================


class FrozenSample:
    def __init__(self, model, items, config, device, elem=0, bf16_trunk=True):
        """`items` is a list of transformed dataset entries, exactly as they sit in
        full_train_ds.pkl. They are collated here the same way the training DataLoader
        collates them, so the batch the model sees is identical."""
        self.device, self.elem = device, elem
        raw = collate_batch_drop_none(items, config)
        if raw is None:
            raise RuntimeError("collate dropped every item (transform errors?)")
        bwl = tree_map(lambda x: x.to(device), raw, skip_unconvertible_entries=True)
        self.bwl = bwl
        batch = bwl["batch"]
        batch.reference_features.setup_block_mask(num_diffusion_samples=1)
        batch.token_features.setup_block_mask()
        batch.reference_features.materialize()
        self.batch = batch

        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16_trunk \
            else torch.amp.autocast(device_type="cuda", enabled=False)
        with torch.no_grad(), ctx:
            trunk = model.evoformer(batch)
        self.trunk = tuple(t.detach().float() for t in trunk)

        # ground truth exactly as evaluation() builds it
        arrs = bwl["atom_array"]
        x_gt = [torch.tensor(np.asarray(a.coord), device=device, dtype=torch.float32) for a in arrs]
        x_gt = utils.pad_to_shape(collate_batch(x_gt), batch.reference_features.positions.shape)
        gt_mask = ~(x_gt.isnan().any(-1))
        ca = [torch.tensor([n == "CA" for n in a.atom_name], device=device) for a in arrs]
        ca = utils.pad_to_shape(collate_batch(ca), batch.reference_features.mask.shape)
        self.x_gt = torch.nan_to_num(x_gt)
        self.mask = gt_mask & batch.reference_features.mask.bool()
        self.ca_mask = self.mask & ca

        self.n_tokens = batch.token_features.token_count
        self.n_ca = int(self.ca_mask[elem].sum().item())
        self.pdb = getattr(arrs[elem], "pdb_id", None) or items[elem].get("pdb_id", "?") \
            if isinstance(items[elem], dict) else "?"

    @property
    def gt(self):
        return self.x_gt[self.elem]

    @property
    def ca(self):
        return self.ca_mask[self.elem]

    def metrics(self, x):
        p = x[self.elem]
        return {
            "rmsd_ca": aligned_rmsd(p, self.gt, self.ca),
            "lddt_ca": lddt_ca(p, self.gt, self.ca),
            "rg": rg(p, self.ca),
            "rg_gt": rg(self.gt, self.ca),
        }

    def centroid_rmsd(self):
        c = (self.gt * self.ca[..., None]).sum(0) / self.ca.sum()
        return aligned_rmsd(c.expand_as(self.gt), self.gt, self.ca)


# ======================================================================================
# denoiser / sampler
# ======================================================================================


def amp(bf16):
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 \
        else torch.amp.autocast(device_type="cuda", enabled=False)


@torch.no_grad()
def call_denoiser(model, s, x, sigma, bf16):
    amount = torch.full(x.shape[:-2], float(sigma), device=s.device)
    with amp(bf16):
        return model.diffusion_module.forward(x, amount, *s.trunk, s.batch).float()


def centered_noise(s, sigma):
    x = sigma * torch.randn_like(s.x_gt)
    return x - utils.masked_mean(
        x, s.batch.reference_features.mask[..., None], axis=-2, keepdims=True
    )


def noised_gt(model, s, sigma):
    x = model.diffusion_sampler.center_random_aug(s.x_gt, s.batch.reference_features)
    return x + sigma * torch.randn_like(x)


def noise_levels(steps, sigma_data, s_max, s_min, rho, device):
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float64)
    return (sigma_data * (s_max ** (1 / rho) + t * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho).float()


@torch.no_grad()
def run_sampler(model, s, config, *, bf16=False, steps=None, s_max=None, step_scale=None,
                gamma_0=None, noise_scale=None, record=False):
    dc = config.diffusion_config
    steps = steps or dc.denoising_steps
    s_max = s_max if s_max is not None else dc.s_max
    step_scale = step_scale if step_scale is not None else dc.step_scale
    gamma_0 = gamma_0 if gamma_0 is not None else dc.gamma_0
    noise_scale = noise_scale if noise_scale is not None else dc.noise_scale

    dev = s.device
    ref = s.batch.reference_features
    shape = s.trunk[1].shape[:-2] + (ref.atom_count, 3)
    c = noise_levels(steps, dc.sigma_data, s_max, dc.s_min, dc.rho, dev)
    x = c[0] * torch.randn(shape, device=dev)
    traj = []
    for i in range(steps):
        c_prev, c_cur = c[i], c[i + 1]
        x = model.diffusion_sampler.center_random_aug(x, ref)
        gamma = gamma_0 if c_cur.item() > dc.gamma_min else 0.0
        t_hat = c_prev * (gamma + 1)
        x_noisy = x + noise_scale * torch.sqrt(
            torch.clamp(t_hat**2 - c_prev**2, min=0)
        ) * torch.randn(shape, device=dev)
        d = call_denoiser(model, s, x_noisy, t_hat.item(), bf16)
        if record:
            traj.append({
                "step": i, "sigma": c_prev.item(),
                "rmsd_x": aligned_rmsd(x_noisy[s.elem], s.gt, s.ca),
                "rmsd_D": aligned_rmsd(d[s.elem], s.gt, s.ca),
                "rg_D": rg(d[s.elem], s.ca),
            })
        x = x_noisy + step_scale * (c_cur - t_hat) * (x_noisy - d) / t_hat
    return x, traj


# ======================================================================================
# tests
# ======================================================================================

PROBE_SIGMAS = (4.8, 16.0, 45.7)
HIGH_SIGMAS = (160.0, 640.0, 2560.0)


def test_probe_repro(model, s, args, tag):
    """Reproduce the wandb probe numbers, then extend past where the probe stopped."""
    print("\n" + "=" * 96)
    print(f"TEST 1 -- reproduce probe/denoise and probe/generate  [{tag}]")
    print("=" * 96)
    base = s.centroid_rmsd()
    rows = []
    for sig in PROBE_SIGMAS + HIGH_SIGMAS:
        torch.manual_seed(args.seed)
        d_gt = call_denoiser(model, s, noised_gt(model, s, sig), sig, args.bf16)
        torch.manual_seed(args.seed)
        d_pn = call_denoiser(model, s, centered_noise(s, sig), sig, args.bf16)
        sigma_data = 16.0
        rows.append([
            f"{sig:g}",
            fmt(sigma_data**2 / (sigma_data**2 + sig**2), 3),
            fmt(aligned_rmsd(d_gt[s.elem], s.gt, s.ca)),
            fmt(aligned_rmsd(d_pn[s.elem], s.gt, s.ca)),
            fmt(lddt_ca(d_pn[s.elem], s.gt, s.ca), 3),
            fmt(rg(d_pn[s.elem], s.ca) / rg(s.gt, s.ca)),
        ])
        torch.cuda.empty_cache()
    print(table(rows, ["sigma", "d_skip", "denoise", "generate", "lDDT(gen)", "Rg/gt(gen)"]))
    print(f"\ncentroid baseline = {base:.2f} A,  Rg_gt = {rg(s.gt, s.ca):.1f} A")
    print("(d_skip is how much of the INPUT the preconditioner forces into the output --")
    print(" at sigma=4.8 it is ~0.92, so 'generate' cannot look good there by construction.)")
    return base


def test_sampler(model, s, config, args, tag):
    print("\n" + "=" * 96)
    print(f"TEST 2 -- full sampler vs single call  [{tag}]")
    print("=" * 96)
    rows = []
    for label, kw in [
        ("AF3 default", {}),
        ("step_scale=1.0", dict(step_scale=1.0)),
        ("pure Euler (g=0, ss=1)", dict(gamma_0=0.0, noise_scale=0.0, step_scale=1.0)),
        ("s_max=20", dict(s_max=20.0)),
        ("120 steps", dict(steps=120)),
    ]:
        torch.manual_seed(args.seed)
        x, _ = run_sampler(model, s, config, bf16=args.bf16, **kw)
        m = s.metrics(x)
        rows.append([label, fmt(m["rmsd_ca"]), fmt(m["lddt_ca"], 3),
                     fmt(m["rg"] / m["rg_gt"])])
        print("  ".join(rows[-1]))
        torch.cuda.empty_cache()
    print()
    print(table(rows, ["sampler config", "RMSD_CA", "lDDT_CA", "Rg/gt"]))


def test_trajectory(model, s, config, args, tag):
    print("\n" + "=" * 96)
    print(f"TEST 3 -- per-step trace  [{tag}]")
    print("=" * 96)
    torch.manual_seed(args.seed)
    _, traj = run_sampler(model, s, config, bf16=args.bf16, record=True)
    print(table(
        [[t["step"], f"{t['sigma']:.4g}", fmt(t["rmsd_x"], 1), fmt(t["rmsd_D"]), fmt(t["rg_D"], 1)]
         for t in traj],
        ["step", "sigma", "RMSD(x)", "RMSD(D)", "Rg(D)"],
    ))
    print(f"\nGT Rg = {rg(s.gt, s.ca):.1f} A")


# ======================================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/step_1999.pt")
    ap.add_argument("--frozen", default="full_train_ds.pkl")
    ap.add_argument("--fresh-pickle", default="eval_ds.pkl",
                    help="dataset object with a live transform, for the A/B comparison")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="match training micro_batch_size so the collated batch is identical")
    ap.add_argument("--elem", type=int, default=0, help="which element of the batch to score")
    ap.add_argument("--start", type=int, default=0, help="index of the first frozen item")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cycle", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-fresh", action="store_true", help="skip the live-pipeline comparison")
    args = ap.parse_args()

    config = build_config(args.n_cycle)
    torch.cuda.set_device(args.device)
    model = load_model(config, args.ckpt, args.device)

    with open(args.frozen, "rb") as f:
        frozen = pickle.load(f)
    print(f"[data] {args.frozen}: {len(frozen)} frozen samples")
    items = [frozen[args.start + i] for i in range(args.batch_size)]

    s = FrozenSample(model, items, config, args.device, elem=args.elem)
    print(f"[data] scoring element {args.elem}: {s.n_tokens} tokens, {s.n_ca} CA")

    test_probe_repro(model, s, args, "FROZEN")
    test_sampler(model, s, config, args, "FROZEN")
    test_trajectory(model, s, config, args, "FROZEN")

    if not args.no_fresh and Path(args.fresh_pickle).exists():
        print("\n" + "#" * 96)
        print("# same structures, but re-drawn through the live (stochastic) pipeline")
        print("#" * 96)
        with open(args.fresh_pickle, "rb") as f:
            ds = pickle.load(f)
        ds.transform = af3_pipeline_none_on_error(config, is_inference=True)
        fresh_items = [ds[args.start + i] for i in range(args.batch_size)]
        try:
            s2 = FrozenSample(model, fresh_items, config, args.device, elem=args.elem)
            test_probe_repro(model, s2, args, "FRESH DRAW")
            test_sampler(model, s2, config, args, "FRESH DRAW")
        except Exception as e:
            print(f"[fresh comparison failed: {e}]")
        print(
            "\nThe gap between the FROZEN and FRESH DRAW tables is the cost of the pipeline\n"
            "stochasticity: it is what separates probe/generate from eval/loss."
        )


if __name__ == "__main__":
    main()