"""H25: the interval's shape from sigma, its magnitude from the PDE residual.

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/eval_resid_scale.py --ckpt runs/u0/best.pt \
        --sigma-source het --out runs/rscale_u0_het.json

H23 measured that the residual tracks the surrogate's **error** at rank
correlation +0.374 while tracking **sigma** at only +0.107 -- numerator above
denominator on 8 of 8 seeds. So the residual sees error the heteroscedastic
head does not. H14 used it as a gate (dead) and H22/H24 used it as a feature of
a fitted width model (an exact null). This is the third role: the residual as
the interval's magnitude.

    sigma' = sigma * (relresid / median_cal(relresid))
    relresid = ||L(mu) - f|| / ||f||          one apply, no solve

`median_cal(relresid)` is frozen on the CALIBRATION split, exactly as the sigma
floor already is, so in distribution the modulation is ~1 and the
in-distribution pass should be undisturbed. Everything downstream -- the score,
the per-family split conformal, the band -- is the machinery already in
`uqkit.conformal`, unchanged.

**Nothing here is fitted.** No h, no coefficient, no penalty, no threshold.
That is what separates it from H22/H24: there is no knob to turn toward the
band, so if it works it works for the stated reason.

Two things are forced and are recorded per run rather than assumed:

* `PDE2DSimulator.residual` is None for the two time-stepped families, so this
  is the same 24 covariate shards H22 used, and the baseline reported beside it
  is the plain per-family conformal on **those same 24** -- never the 32-shard
  headline.
* the cost is one operator apply per sample. This script measures coverage, not
  time; clause 2 for this arm stays `[not measured]` until `bench_fair.py`
  times it with the apply inside the captured graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import GroupConformal, get_score  # noqa: E402
from uqkit.metrics import binom_ci, rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402

COVARIATE_KINDS = {"input_shift", "graded_rough"}
BAND = (0.88, 0.92)


def cov_entry(covered):
    c = np.asarray(covered, dtype=bool)
    k, n = int(c.sum()), int(len(c))
    if n == 0:
        return {"coverage": None, "n": 0, "ci95": [None, None]}
    lo, hi = binom_ci(k, n)
    return {"coverage": k / n, "n": n, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default="het",
                    choices=["het", "cqr", "const"])
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    model, ck = load_model(args.ckpt, args.device)
    if not ck["args"].get("uq"):
        raise SystemExit(f"{args.ckpt} is not a UQ checkpoint")
    stats = ck["stats"]
    fn, _ = get_score(args.score)
    sims: dict = {}

    # Which families have a cheap apply is a property of the problem. Ask the
    # simulator rather than hardcoding, so the restriction cannot drift.
    probe_a = torch.zeros(1, 2, 64, 64)
    probe_u = torch.zeros(1, 1, 64, 64)
    in_tasks = [t for t in man["in_tasks"]
                if PDE2DSimulator(t, device="cpu").residual(probe_a, probe_u)
                is not None]
    dropped = [t for t in man["in_tasks"] if t not in in_tasks]
    print(f"families with a cheap apply: {in_tasks}; excluded {dropped}",
          flush=True)

    def relresid(a, mean, parent):
        if parent not in sims:
            sims[parent] = PDE2DSimulator(parent, device=a.device)
        r = sims[parent].residual(a, mean)
        if r is None:
            raise SystemExit(f"{parent} has no cheap apply; it must be "
                             f"excluded, not fed a placeholder")
        f = sims[parent].rhs(a)
        return (r.flatten(1).norm(dim=1)
                / f.flatten(1).norm(dim=1).clamp_min(1e-12))

    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        m, s, t, a = predict_shard_single(model, blob, stats, args.device,
                                          sigma_source=args.sigma_source)
        parent = PARENT.get(task, task)
        return {"mean": m, "sigma": s, "truth": t,
                "rr": relresid(a, m, parent), "parent": parent, "task": task}

    cal = {t: gather(t, "cal") for t in in_tasks}
    # frozen on calibration, per family, like the sigma floor
    RR_MED = {t: float(cal[t]["rr"].median()) for t in in_tasks}
    MED = float(torch.cat([cal[t]["sigma"].flatten()
                           for t in in_tasks]).median())

    def modulate(d):
        """sigma * (relresid / median_cal(relresid)), broadcast over the field."""
        w = (d["rr"] / RR_MED[d["parent"]]).view(-1, 1, 1, 1)
        return d["sigma"] * w

    res = {"alpha": args.alpha, "score": args.score, "ckpt": args.ckpt,
           "seed": ck["args"].get("seed"),
           "sigma_source": args.sigma_source,
           "families": in_tasks, "excluded_families": dropped,
           "relresid_median_cal": RR_MED, "sigma_floor_median": MED,
           "band": list(BAND), "fitted_parameters": 0,
           "note": ("sigma' = sigma * relresid/median_cal(relresid). Nothing "
                    "is fitted: the only constant is a calibration median, "
                    "frozen like the sigma floor. Cost is one operator apply "
                    "per sample; clause 2 for this arm is [not measured]."),
           "in_dist": {}, "shards": {}}

    # ---- calibrate both arms on the same calibration split ------------------
    def scores(d, mod):
        sig = modulate(d) if mod else d["sigma"]
        return fn(d["mean"], sig, d["truth"], med=MED).cpu().numpy()

    conf = {}
    for mod, tag in ((False, "base"), (True, "resid")):
        s = np.concatenate([scores(cal[t], mod) for t in in_tasks])
        g = np.concatenate([[t] * len(cal[t]["rr"]) for t in in_tasks])
        conf[tag] = GroupConformal(args.alpha).fit(s, g)
        res[f"q_{tag}"] = {k: float(v) for k, v in conf[tag].q.items()}

    for t in in_tasks:
        d = gather(t, "test")
        rec = {}
        for mod, tag in ((False, "base"), (True, "resid")):
            s = scores(d, mod)
            rec[tag] = cov_entry(conf[tag].covered(s, np.array([t] * len(s))))
        rec["modulation_median"] = float(
            (d["rr"] / RR_MED[t]).median())
        res["in_dist"][t] = rec
        print(f"  in-dist {t:10s} base {rec['base']['coverage']:.4f}  "
              f"resid {rec['resid']['coverage']:.4f}  "
              f"modulation {rec['modulation_median']:.3f}", flush=True)
        del d
        torch.cuda.empty_cache()

    # ---- the 24 covariate shards -------------------------------------------
    n_band = {"base": 0, "resid": 0}
    for kind, tasks in man["ood_suite"].items():
        if kind not in COVARIATE_KINDS:
            continue
        for task in tasks:
            parent = PARENT.get(task, task)
            if parent not in in_tasks:
                continue
            d = gather(task, "ood", 64)
            rec = {"kind": kind, "parent": parent,
                   "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean()),
                   "modulation_median": float((d["rr"] / RR_MED[parent]).median())}
            for mod, tag in ((False, "base"), (True, "resid")):
                s = scores(d, mod)
                e = cov_entry(conf[tag].covered(s, np.array([parent] * len(s))))
                rec[tag] = e
                if e["coverage"] is not None and \
                        BAND[0] <= e["coverage"] <= BAND[1]:
                    n_band[tag] += 1
                    rec[f"{tag}_in_band"] = True
            res["shards"][f"{kind}/{task}/N64"] = rec
            del d
            torch.cuda.empty_cache()

    res["n_shards"] = len(res["shards"])
    res["in_band"] = n_band
    print(f"[{res['n_shards']} shards] base {n_band['base']}  "
          f"resid {n_band['resid']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
