"""H23: is the residual's negative width coefficient physical, or suppression?

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/resid_corr.py --ckpt runs/u0/best.pt --out runs/rcorr_u0.json

H22 measured that `log_resid` is the second most important of 44 width features
on half the seeds, and that **every** residual coefficient on **every** seed is
negative: a larger equation violation asks for a *narrower* interval. Taken at
face value that is backwards -- a bigger residual should mean a bigger error and
a wider band.

It cannot be taken at face value, because those are partial coefficients in a
multivariate fit over correlated features, where a negative partial coefficient
is routinely a suppression effect. This script measures the quantity that
separates the two readings, and it involves no fitting at all:

    the UNIVARIATE rank correlation, within each shard, between
    log ||r||/||f||  and  log S,   S = max|mu - u| / sigma-tilde

* **positive here, negative in the fit** -> suppression. The sign flip is
  collinearity with the amplitude and sigma features and says nothing physical.
* **negative here too** -> on this surrogate sigma-tilde really does grow faster
  than the error as the equation violation grows, which is a statement about
  the heteroscedastic head and worth having.

The decomposition is also reported directly, because it is the mechanism
either way: rank correlations of `log_resid` against the numerator
(`log max|mu - u|`) and against the denominator (`log sigma-tilde`)
separately. If the residual tracks the error but sigma-tilde tracks it *harder*,
both facts show up as positive correlations with the second one larger, and no
inference is needed.

Ground truth is used here, deliberately and only here: this is a diagnostic
about why a fitted model behaves as it does, not a detector. Nothing it
produces is available at deployment and nothing it produces enters a clause.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import _floor, get_score  # noqa: E402
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.ood import consistency_score  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402

COVARIATE_KINDS = {"input_shift", "graded_rough"}


def _ranks(v):
    """Ranks with ties averaged -- the tie handling that a previous version of
    this repo's Spearman got wrong and that inflated a reported correlation."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    r = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 8:
        return None
    a, b = _ranks(x), _ranks(y)
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if d == 0 else float(a @ b / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default="het")
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    model, ck = load_model(args.ckpt, args.device)
    stats = ck["stats"]
    fn, _ = get_score(args.score)
    sims: dict = {}

    def sim_for(t):
        if t not in sims:
            sims[t] = PDE2DSimulator(t, device=args.device)
        return sims[t]

    def predict(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        return predict_shard_single(model, blob, stats, args.device,
                                    sigma_source=args.sigma_source)

    # the sigma floor is frozen on the pooled calibration split, exactly as
    # every other score in this repo freezes it
    cal_sig = []
    for t in in_tasks:
        _m, s, _tr, _a = predict(t, "cal")
        cal_sig.append(s.flatten())
    MED = float(torch.cat(cal_sig).median())
    del cal_sig

    families = [t for t in in_tasks
                if sim_for(t).residual(torch.zeros(1, 2, 64, 64,
                                                   device=args.device),
                                       torch.zeros(1, 1, 64, 64,
                                                   device=args.device))
                is not None]

    specs = [(k, t, 64) for k, ts in man["ood_suite"].items()
             if k in COVARIATE_KINDS for t in ts
             if PARENT.get(t, t) in families]

    res = {"ckpt": args.ckpt, "seed": ck["args"].get("seed"),
           "score": args.score, "sigma_source": args.sigma_source,
           "families": families, "sigma_floor_median": MED,
           "uses_ground_truth": True,
           "note": ("univariate rank correlations, no fitting. Separates a "
                    "suppression effect in H22's multivariate fit from a real "
                    "statement about the sigma head."),
           "shards": {}}

    for kind, task, N in specs:
        parent = PARENT.get(task, task)
        m, s, tr, a = predict(task, "ood", N)
        sim = sim_for(parent)
        r = sim.residual(a, m)
        f = sim.rhs(a)
        eps = 1e-12
        lr = torch.log10((r.flatten(1).norm(dim=1)
                          / f.flatten(1).norm(dim=1).clamp_min(eps))
                         .clamp_min(eps)).cpu().numpy()
        lc = torch.log10(consistency_score(r, r + f, f).clamp_min(eps)
                         ).cpu().numpy()
        S = fn(m, s, tr, med=MED).cpu().numpy()
        # the decomposition: S is a ratio, so correlate with each part
        num = (m - tr).abs().flatten(1).max(dim=1).values
        den = _floor(s, 0.05, MED).flatten(1).max(dim=1).values
        res["shards"][f"{kind}/{task}/N{N}"] = {
            "kind": kind, "parent": parent,
            "rel_l2_mean": float(rel_l2(m, tr).mean()),
            "spearman_logresid_vs_logS": spearman(lr, np.log10(S + eps)),
            "spearman_logconsist_vs_logS": spearman(lc, np.log10(S + eps)),
            "spearman_logresid_vs_numerator": spearman(
                lr, np.log10(num.cpu().numpy() + eps)),
            "spearman_logresid_vs_denominator": spearman(
                lr, np.log10(den.cpu().numpy() + eps)),
            "n": int(len(S)),
        }
        del m, s, tr, a, r, f
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    v = [c["spearman_logresid_vs_logS"] for c in res["shards"].values()
         if c["spearman_logresid_vs_logS"] is not None]
    print(f"{len(v)} shards; median rho(log_resid, log S) = "
          f"{float(np.median(v)):+.3f}; positive on "
          f"{sum(1 for x in v if x > 0)}/{len(v)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
