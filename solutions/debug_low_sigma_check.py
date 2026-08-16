#!/usr/bin/env python
"""
lowsigma_check.py -- the frozen run localised the failure precisely.

    step 22   t_hat = 5.55   RMSD(D) =  1.74   Rg(D) = 22.8
    step 23   t_hat = 3.26   RMSD(D) = 10.16   Rg(D) = 17.4

23 steps of near-perfect denoising (0.76-1.02 A from sigma=2560 down), then a cliff inside
a factor of 1.7 in sigma, and no recovery because d_skip > 0.99 below there pins the output
to the input. Your probe sampled 4.8 / 16 / 45.7, which straddles the cliff without
resolving it.

This script answers three things:

  A. where exactly is the cliff, and does it depend on WHAT the input is (in-distribution
     noised GT) or only on the noise LEVEL?
  B. is it a noise-level *mismatch* problem? step_scale=1.5 leaves the state cleaner than
     its nominal sigma, so the network is routinely told a sigma larger than the noise it
     is actually looking at. Training never shows it that combination.
  C. what does the sampler score if you simply stop early or floor sigma?

Run:
    python -m solutions.lowsigma_check --ckpt checkpoints/step_1999.pt
"""

import argparse
import pickle

import numpy as np
import torch

# the frozen-sample machinery, whatever you named that file
_src = None
for _name in ("debug_consistency_check", "frozen_consistency_check", "consistency_check"):
    try:
        _src = __import__(_name)
        break
    except ImportError:
        continue
if _src is None:  # pragma: no cover
    raise ImportError("could not find the frozen consistency check module")

FrozenSample = _src.FrozenSample
build_config = _src.build_config
call_denoiser = _src.call_denoiser
centered_noise = _src.centered_noise
noised_gt = _src.noised_gt
noise_levels = _src.noise_levels
aligned_rmsd = _src.aligned_rmsd
lddt_ca = _src.lddt_ca
rg = _src.rg
table = _src.table
fmt = _src.fmt

from common import utils
from diffusion.model import Model


def load_model(config, ckpt, device):
    """Same as before but with the state-dict prefix stripping fixed: `.replace` would
    rewrite 'diffusion_module.' to 'diffusion_.' on every key."""
    model = Model(config)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    def strip(k):
        for p in ("module.", "_orig_mod."):
            while k.startswith(p):
                k = k[len(p):]
        return k

    sd = {strip(k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    return model.to(device).eval()


SIGMA_SCAN = [0.5, 0.8, 1.2, 1.8, 2.5, 3.26, 4.5, 5.55, 7.0, 9.0, 12.0, 16.0]


# ======================================================================================
# A. dense scan across the cliff, with three kinds of input
# ======================================================================================


@torch.no_grad()
def test_cliff(model, s, args):
    """Three inputs at each sigma:
        gt    randaug(x_gt) + sigma*noise           -- exactly the training distribution
        self  randaug(D_high) + sigma*noise         -- what the sampler actually feeds it:
                                                       a noised version of its OWN output
        noise sigma*noise                           -- pure generation
    If `gt` survives the cliff but `self` does not, the model cannot denoise its own
    near-miss. If both die at the same sigma, it is a plain low-sigma competence hole."""
    print("\n" + "=" * 100)
    print("TEST A -- dense sigma scan across the cliff, three input types")
    print("=" * 100)

    # the model's own high-sigma answer, i.e. what the trajectory is carrying at step ~22
    torch.manual_seed(args.seed)
    d_high = call_denoiser(model, s, centered_noise(s, 45.7), 45.7, args.bf16)
    print(f"reference D(sigma=45.7) is {aligned_rmsd(d_high[s.elem], s.gt, s.ca):.2f} A from GT")

    rows = []
    for sig in SIGMA_SCAN:
        vals = []
        for kind in ("gt", "self", "noise"):
            torch.manual_seed(args.seed)
            if kind == "gt":
                x = noised_gt(model, s, sig)
            elif kind == "self":
                base = model.diffusion_sampler.center_random_aug(
                    d_high, s.batch.reference_features
                )
                x = base + sig * torch.randn_like(base)
            else:
                x = centered_noise(s, sig)
            d = call_denoiser(model, s, x, sig, args.bf16)
            vals.append((aligned_rmsd(d[s.elem], s.gt, s.ca),
                         aligned_rmsd(x[s.elem], s.gt, s.ca),
                         rg(d[s.elem], s.ca) / rg(s.gt, s.ca)))
            torch.cuda.empty_cache()
        rows.append([f"{sig:g}", fmt(256 / (256 + sig**2), 3)]
                    + [fmt(v[0]) for v in vals]
                    + [fmt(vals[0][1]), fmt(vals[0][2])])
    print(table(rows, ["sigma", "d_skip", "D(gt in)", "D(self in)", "D(noise in)",
                       "input RMSD", "Rg/gt"]))
    print("\nA column that is fine at 5.55 and broken at 3.26 reproduces the trajectory cliff.")


# ======================================================================================
# B. noise-level mismatch
# ======================================================================================


@torch.no_grad()
def test_mismatch(model, s, args):
    """The sampler tells the network t_hat while the state carries a different amount of
    actual noise. With step_scale=1.5 the residual after a step is

        |1 + 1.5*(c_next/t_hat - 1)| * t_hat

    which is far below c_next, so every call sees a state cleaner than advertised. Here we
    build inputs with `true` noise and label them `told`, and see how far apart they can be
    before the network breaks."""
    print("\n" + "=" * 100)
    print("TEST B -- noise level mismatch: input noised at `true`, network told `told`")
    print("=" * 100)
    told_vals = [1.8, 3.26, 5.55, 9.0, 16.0]
    ratios = [0.25, 0.5, 0.83, 1.0, 1.5, 2.0]
    rows = []
    for told in told_vals:
        row = [f"{told:g}"]
        for r in ratios:
            true = told * r
            torch.manual_seed(args.seed)
            x = noised_gt(model, s, true)
            d = call_denoiser(model, s, x, told, args.bf16)
            row.append(fmt(aligned_rmsd(d[s.elem], s.gt, s.ca)))
            torch.cuda.empty_cache()
        rows.append(row)
    print(table(rows, ["told sigma"] + [f"true/told={r:g}" for r in ratios]))
    print("\nThe diagonal (ratio 1.0) is the training distribution. Degradation towards the")
    print("left is intolerance to a state that is cleaner than its nominal sigma -- which is")
    print("exactly what step_scale > 1 manufactures at every step.")


# ======================================================================================
# C. what rescues the sample
# ======================================================================================


@torch.no_grad()
def sampler_with_stop(model, s, config, *, stop_step=None, sigma_floor=None,
                      step_scale=None, bf16=False, steps=None, take_denoised=True):
    """Standard AF3 sampler with two extra exits: stop after `stop_step` iterations, or
    stop once c_prev drops below `sigma_floor`. Returns the last denoiser output rather
    than the last state when `take_denoised` (the state carries the churn noise)."""
    dc = config.diffusion_config
    steps = steps or dc.denoising_steps
    step_scale = step_scale if step_scale is not None else dc.step_scale
    dev = s.device
    ref = s.batch.reference_features
    shape = s.trunk[1].shape[:-2] + (ref.atom_count, 3)
    c = noise_levels(steps, dc.sigma_data, dc.s_max, dc.s_min, dc.rho, dev)
    x = c[0] * torch.randn(shape, device=dev)
    last_d = x
    for i in range(steps):
        if stop_step is not None and i >= stop_step:
            break
        if sigma_floor is not None and c[i].item() < sigma_floor:
            break
        c_prev, c_cur = c[i], c[i + 1]
        x = model.diffusion_sampler.center_random_aug(x, ref)
        gamma = dc.gamma_0 if c_cur.item() > dc.gamma_min else 0.0
        t_hat = c_prev * (gamma + 1)
        x_noisy = x + dc.noise_scale * torch.sqrt(
            torch.clamp(t_hat**2 - c_prev**2, min=0)
        ) * torch.randn(shape, device=dev)
        last_d = call_denoiser(model, s, x_noisy, t_hat.item(), bf16)
        x = x_noisy + step_scale * (c_cur - t_hat) * (x_noisy - last_d) / t_hat
    return last_d if take_denoised else x


def test_rescue(model, s, config, args):
    print("\n" + "=" * 100)
    print("TEST C -- early stopping and sigma floors")
    print("=" * 100)
    rows = []
    for label, kw in (
        [("full 30 steps (state)", dict(take_denoised=False))]
        + [(f"stop after step {k}", dict(stop_step=k)) for k in (18, 20, 22, 23, 24, 26)]
        + [(f"sigma floor {f:g}", dict(sigma_floor=f)) for f in (8.0, 5.0, 3.0, 2.0)]
        + [("step_scale=1.0, floor 3", dict(step_scale=1.0, sigma_floor=3.0))]
    ):
        torch.manual_seed(args.seed)
        x = sampler_with_stop(model, s, config, bf16=args.bf16, **kw)
        p = x[s.elem]
        rows.append([label, fmt(aligned_rmsd(p, s.gt, s.ca)),
                     fmt(lddt_ca(p, s.gt, s.ca), 3),
                     fmt(rg(p, s.ca) / rg(s.gt, s.ca))])
        print("  ".join(rows[-1]))
        torch.cuda.empty_cache()
    print()
    print(table(rows, ["config", "RMSD_CA", "lDDT_CA", "Rg/gt"]))
    print("\nReturning the last DENOISER OUTPUT rather than the last state also matters:")
    print("the state still carries churn noise, the denoised prediction does not.")


def test_precision(model, s, args):
    """bf16 has an 8-bit mantissa; at low sigma the output is d_skip*x_noisy plus a small
    correction, so the correction competes with the rounding error on ~24 A coordinates."""
    print("\n" + "=" * 100)
    print("TEST D -- bf16 vs fp32 at low sigma")
    print("=" * 100)
    rows = []
    for sig in (1.8, 3.26, 5.55, 16.0):
        vals = []
        for bf16 in (True, False):
            torch.manual_seed(args.seed)
            x = noised_gt(model, s, sig)
            d = call_denoiser(model, s, x, sig, bf16)
            vals.append(aligned_rmsd(d[s.elem], s.gt, s.ca))
            torch.cuda.empty_cache()
        rows.append([f"{sig:g}", fmt(vals[0]), fmt(vals[1])])
    print(table(rows, ["sigma", "bf16", "fp32"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/step_1999.pt")
    ap.add_argument("--frozen", default="full_train_ds.pkl")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--elem", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cycle", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--tests", nargs="+", default=["cliff", "mismatch", "rescue", "precision"])
    args = ap.parse_args()

    config = build_config(args.n_cycle)
    torch.cuda.set_device(args.device)
    model = load_model(config, args.ckpt, args.device)

    with open(args.frozen, "rb") as f:
        frozen = pickle.load(f)
    print(f"[data] {args.frozen}: {len(frozen)} frozen samples")
    items = [frozen[args.start + i] for i in range(args.batch_size)]
    s = FrozenSample(model, items, config, args.device, elem=args.elem)

    if "cliff" in args.tests:
        test_cliff(model, s, args)
    if "mismatch" in args.tests:
        test_mismatch(model, s, args)
    if "rescue" in args.tests:
        test_rescue(model, s, config, args)
    if "precision" in args.tests:
        test_precision(model, s, args)


if __name__ == "__main__":
    main()