import numpy as np
import torch
from common import utils


class GenerationProbe:
    """Cheap per-step diagnostic separating denoising from generation.
 
    For a fixed held-in structure it reports, at each sigma:
        denoise   RMSD of D(noised ground truth)   -- what train/loss already measures
        generate  RMSD of D(pure noise)            -- what the sampler actually depends on
        spread    RMSD between two independent draws at the same sigma  -> variance
        centroid  RMSD of predicting the centroid everywhere            -> the floor
 
    `generate` sitting at `centroid` means the model cannot synthesise a fold from the
    conditioning, no matter how good `denoise` looks. `spread` is the variance term in
    single^2 = bias^2 + var; watch it fall as training progresses.
 
    Cost: one evoformer pass plus 2*len(sigmas) diffusion calls. Run it every ~25 steps.
    Uses a single fixed batch so torch.compile does not see new shapes.
    """
 
    def __init__(self, model, batch_with_labels, config, device, sigmas=(4.8, 16.0, 45.7)):
        self.config = config
        self.device = device
        self.sigmas = sigmas
        from feature_extraction.feature_extraction import collate_batch, tree_map
 
        self.bwl = tree_map(lambda x: x.to(device), batch_with_labels,
                            skip_unconvertible_entries=True)
        batch = self.bwl["batch"]
        batch.reference_features.setup_block_mask(1)
        batch.token_features.setup_block_mask()
        batch.reference_features.materialize()
        self.batch = batch
 
        atom_array = self.bwl["atom_array"][0]
        x_gt = torch.tensor(np.asarray(atom_array.coord), device=device, dtype=torch.float32)
        x_gt = utils.pad_to_shape(collate_batch([x_gt]), batch.reference_features.positions.shape)
        m = ~(x_gt.isnan().any(-1)) & batch.reference_features.mask.bool()
        ca = torch.tensor([n == "CA" for n in atom_array.atom_name], device=device)
        ca = utils.pad_to_shape(collate_batch([ca]), batch.reference_features.mask.shape)
        self.x_gt = torch.nan_to_num(x_gt)
        self.mask = m
        self.ca = m & ca
        self.centroid_rmsd = self._rmsd(
            ((self.x_gt[0] * self.ca[0][..., None]).sum(0) / self.ca[0].sum()).expand_as(
                self.x_gt[0]
            )
        )
 
    def _rmsd(self, x_pred):
        from training.training_module import weighted_align
 
        w = self.ca[0].float()
        with torch.autocast(device_type="cuda", enabled=False):
            gt_al = weighted_align(self.x_gt[0].float(), x_pred.float(), w)
        d2 = ((x_pred.float() - gt_al) ** 2).sum(-1)
        return torch.sqrt((d2 * w).sum() / w.sum()).item()
 
    @torch.no_grad()
    def __call__(self, model, bf16=True):
        was_training = model.training
        model.eval()
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else \
            torch.amp.autocast(device_type="cuda", enabled=False)
        out = {"probe/centroid_rmsd": self.centroid_rmsd}
        try:
            with ctx:
                trunk = model.evoformer(self.batch)
            trunk = tuple(t.float() for t in trunk)
            ref = self.batch.reference_features
            for sig in self.sigmas:
                amount = torch.full(self.x_gt.shape[:-2], float(sig), device=self.device)
                x_aug = model.diffusion_sampler.center_random_aug(self.x_gt, ref)
                d_gt = self._call(model, x_aug + sig * torch.randn_like(x_aug), amount, trunk, ctx)
 
                pn = [sig * torch.randn_like(self.x_gt) for _ in range(2)]
                pn = [p - utils.masked_mean(p, ref.mask[..., None], axis=-2, keepdims=True)
                      for p in pn]
                d_a = self._call(model, pn[0], amount, trunk, ctx)
                d_b = self._call(model, pn[1], amount, trunk, ctx)
 
                from training.training_module import weighted_align
 
                w = self.ca[0].float()
                with torch.autocast(device_type="cuda", enabled=False):
                    b_al = weighted_align(d_b[0].float(), d_a[0].float(), w)
                spread = torch.sqrt(
                    (((d_a[0] - b_al) ** 2).sum(-1) * w).sum() / w.sum()
                ).item()
 
                out[f"probe/denoise_s{sig:g}"] = self._rmsd(d_gt[0])
                out[f"probe/generate_s{sig:g}"] = self._rmsd(d_a[0])
                out[f"probe/spread_s{sig:g}"] = spread
                out[f"probe/gen_gap_s{sig:g}"] = (
                    self.centroid_rmsd - self._rmsd(d_a[0])
                )  # >0 means better than a blob
        finally:
            model.train(was_training)
        return out
 
    def _call(self, model, x, amount, trunk, ctx):
        with ctx:
            return model.diffusion_module.forward(x, amount, *trunk, self.batch).float()