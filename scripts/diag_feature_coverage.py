"""H23: how far outside its fitting range does the width model extrapolate?

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/diag_feature_coverage.py --ckpt runs/u0/best.pt \
        --sigma-source het --out runs/feat_coverage_u0_het.json

H22 measured that the two residual features are real signal, are exactly
scale-invariant (so they cannot be the amplitude effect), and move exactly one
family: Darcy from 6/10 evaluation shards in band to 0/10, pushed through the
**top** of the band at 1.18-2.71x width. The failure is one of scale, not of
information -- the response is correctly signed and too large.

A linear model in standardized features extrapolates without bound. If the
evaluation shards' features lie outside the range the development shards
covered, a coefficient that is right *inside* the fitting range produces an
arbitrarily large width *outside* it. This script measures that distance, per
family, and fits nothing:

* `frac_outside`  -- fraction of evaluation samples beyond the development
  min-max for that feature and family;
* `max_z_beyond`  -- how far the furthest evaluation sample lies beyond the
  development range, in units of the development standard deviation. This is
  the same standardization `QuantileScale` applies, so the number is directly
  the size of the extrapolation the width model performs.

`a_spec9` (log input amplitude) is carried through as a **within-run control**.
It is the coefficient that carries every arm measured so far, at 6.3x the
larger residual coefficient. If it extrapolates just as far on Darcy without
blowing the band, then extrapolation distance alone does not explain H22 and
prediction 1 being true would not establish the cause. The control is the point
of the script, not a decoration.

No coverage is computed here and no quantile is calibrated, so nothing in this
file can flatter a clause. It costs one forward pass per shard plus one operator
apply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit import devshift as D  # noqa: E402
from uqkit.features import spectral_features  # noqa: E402
from uqkit.ood import consistency_score  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402

COVARIATE_KINDS = {"input_shift", "graded_rough"}
#: the amplitude feature, carried as the within-run control. Index 9 of channel
#: 0 in `spectral_features` is log10(std of channel 0); `tests/` pin that.
AMP_IDX = 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-n", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default="het",
                    choices=["het", "cqr", "const"])
    args = ap.parse_args()

    D.assert_disjoint()
    dev_specs = D.register()
    man = json.loads(Path(args.root, "manifest.json").read_text())
    model, ck = load_model(args.ckpt, args.device)
    if not ck["args"].get("uq"):
        raise SystemExit(f"{args.ckpt} is not a UQ checkpoint")
    stats = ck["stats"]
    sims: dict = {}

    def feats(a, mean, parent):
        """(n, 3): log_consist, log_resid, a_spec9. None where no cheap apply."""
        if parent not in sims:
            sims[parent] = PDE2DSimulator(parent, device=a.device)
        r = sims[parent].residual(a, mean)
        if r is None:
            return None
        f = sims[parent].rhs(a)
        eps = 1e-12
        c = consistency_score(r, r + f, f)
        rn = r.flatten(1).norm(dim=1) / f.flatten(1).norm(dim=1).clamp_min(eps)
        amp = spectral_features(a)[:, AMP_IDX]
        return torch.stack([torch.log10(c.clamp_min(eps)),
                            torch.log10(rn.clamp_min(eps)),
                            amp], dim=-1).double().cpu().numpy()

    def run(blob):
        m, _s, _t, a = predict_shard_single(model, blob, stats, args.device,
                                            sigma_source=args.sigma_source)
        return feats(a, m, PARENT.get(blob["task"], blob["task"]))

    # ---- development rows, per family: the range h is fitted over -----------
    dev: dict = {}
    for i, (task, parent, _mech, _over) in enumerate(dev_specs):
        a, u, _ = D.generate(task, args.dev_n, N=64, device=args.device,
                             index=i)
        z = run({"a": a.cpu(), "u": u.cpu(), "task": task, "N": 64,
                 "split": "dev", "parent": parent})
        if z is not None:
            dev.setdefault(parent, []).append(z)
        del a, u
        torch.cuda.empty_cache()
    # the in-distribution calibration half is part of the fitting rows too
    for t in man["in_tasks"]:
        z = run(load_shard(args.root, t, "cal"))
        if z is not None:
            dev.setdefault(t, []).append(z)
    dev = {k: np.concatenate(v) for k, v in dev.items()}

    # ---- evaluation shards --------------------------------------------------
    NAMES = ["log_consist", "log_resid", "a_spec9(control)"]
    res = {"ckpt": args.ckpt, "seed": ck["args"].get("seed"),
           "sigma_source": args.sigma_source, "dev_n": args.dev_n,
           "features": NAMES,
           "note": ("distances only -- no quantile is calibrated and no "
                    "coverage is computed here, so nothing in this file can "
                    "flatter a clause. a_spec9 is the within-run control."),
           "families": sorted(dev), "by_shard": {}, "by_family": {}}

    per_fam: dict = {}
    for kind, tasks in man["ood_suite"].items():
        if kind not in COVARIATE_KINDS:
            continue
        for task in tasks:
            parent = PARENT.get(task, task)
            if parent not in dev:
                continue
            z = run(load_shard(args.root, task, "ood", 64))
            lo, hi = dev[parent].min(0), dev[parent].max(0)
            sd = dev[parent].std(0, ddof=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            beyond = np.maximum(lo - z, z - hi)          # >0 iff outside
            rec = {}
            for j, nm in enumerate(NAMES):
                rec[nm] = {
                    "frac_outside": float((beyond[:, j] > 0).mean()),
                    "max_z_beyond": float(max(beyond[:, j].max(), 0.0) / sd[j]),
                    "median_z_beyond": float(
                        max(float(np.median(beyond[:, j])), 0.0) / sd[j]),
                }
            res["by_shard"][f"{kind}/{task}/N64"] = {
                "parent": parent, "kind": kind, **rec}
            per_fam.setdefault(parent, []).append(rec)
            torch.cuda.empty_cache()

    for fam, recs in per_fam.items():
        res["by_family"][fam] = {
            nm: {"max_z_beyond": max(r[nm]["max_z_beyond"] for r in recs),
                 "mean_frac_outside": float(
                     np.mean([r[nm]["frac_outside"] for r in recs])),
                 "n_shards": len(recs)}
            for nm in NAMES}
        line = "  ".join(
            f"{nm}: max_z {res['by_family'][fam][nm]['max_z_beyond']:7.2f} "
            f"outside {res['by_family'][fam][nm]['mean_frac_outside']:.3f}"
            for nm in NAMES)
        print(f"[{fam:10s}] {line}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
