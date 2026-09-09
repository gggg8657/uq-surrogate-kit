"""Conformal coverage, in distribution and under every shift in the suite.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_conformal.py \
        --ckpts runs/m0/best.pt ... --out runs/conformal.json

Protocol, fixed before the run (see `critique_log.md`, turn 1):

* alpha = 0.1, so the target is 90% and the KPI band is [88, 92].
* Calibration is the `cal` split of the five trained families, disjoint from
  train and test by seed. Nothing on a shifted shard is ever used to fit a
  quantile -- weighted conformal uses only the shifted *inputs*, which is what a
  deployment has.
* Four scores are reported, not one, because "90% coverage" of a field means
  different things (see `uqkit/conformal.py`). `field_max` is the headline: the
  whole field inside the band.
* Every coverage carries a Wilson 95% interval. On a 512-sample OOD shard that
  interval is about +/-2.6pp, which is wider than the KPI band -- so a single
  shard cannot by itself decide a clause, and the pooled numbers say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import (GroupConformal, LikelihoodRatioProbe,  # noqa: E402
                             SplitConformal, WeightedConformal, get_score)
from uqkit.features import spectral_features  # noqa: E402
from uqkit.metrics import binom_ci, pearson, rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402

SCORE_NAMES = ["field_max", "norm_ratio", "rel_l2", "pixel"]


def cov_entry(covered):
    k, n = int(np.sum(covered)), int(len(covered))
    lo, hi = binom_ci(k, n)
    return {"coverage": k / n, "n": n, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/conformal.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]

    # ---- gather predictions -------------------------------------------------
    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        m, s, t, a = predict_shard(models, blob, stats, args.device)
        return {"mean": m, "sigma": s, "truth": t, "a": a,
                "task": task, "parent": PARENT.get(task, task), "N": N}

    cal = {t: gather(t, "cal") for t in in_tasks}
    test = {t: gather(t, "test") for t in in_tasks}

    ood_specs = []
    for kind, tasks in man["ood_suite"].items():
        ood_specs += [(kind, t, 64) for t in tasks]
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                ood_specs.append((f"resolution_{N}", t, N))

    res = {"alpha": args.alpha, "n_members": len(models),
           "ckpts": args.ckpts, "scores": {}, "error": {}, "spread": {}}

    # per-family error and spread-error correlation: context for every coverage
    for t in in_tasks:
        d = test[t]
        err = rel_l2(d["mean"], d["truth"])
        spr = d["sigma"].flatten(1).norm(dim=1) / d["mean"].flatten(1).norm(dim=1)
        res["error"][t] = {"rel_l2_mean": float(err.mean()),
                           "rel_l2_p90": float(err.quantile(0.9)),
                           "spread_error_pearson": pearson(spr, err)}

    # ---- per score ----------------------------------------------------------
    for sname in SCORE_NAMES:
        fn, per_sample = get_score(sname)
        out = {"per_sample": per_sample, "in_dist": {}, "ood": {}}

        cal_scores = {t: fn(cal[t]["mean"], cal[t]["sigma"], cal[t]["truth"])
                      for t in in_tasks}
        pooled_cal = torch.cat([cal_scores[t] for t in in_tasks])
        pooled_grp = np.concatenate([[t] * len(cal_scores[t]) for t in in_tasks])

        split = SplitConformal(args.alpha).fit(pooled_cal)
        group = GroupConformal(args.alpha).fit(pooled_cal, pooled_grp)
        out["q_split"] = split.q
        out["q_group"] = {k: float(v) for k, v in group.q.items()}

        # in distribution
        te_scores = {t: fn(test[t]["mean"], test[t]["sigma"], test[t]["truth"])
                     for t in in_tasks}
        pooled_te = torch.cat([te_scores[t] for t in in_tasks])
        pooled_te_grp = np.concatenate([[t] * len(te_scores[t]) for t in in_tasks])
        out["in_dist"]["pooled_split"] = cov_entry(split.covered(pooled_te))
        out["in_dist"]["pooled_group"] = cov_entry(
            group.covered(pooled_te, pooled_te_grp))
        out["in_dist"]["per_family_split"] = {
            t: cov_entry(split.covered(te_scores[t])) for t in in_tasks}
        out["in_dist"]["per_family_group"] = {
            t: cov_entry(group.covered(te_scores[t],
                                       np.array([t] * len(te_scores[t]))))
            for t in in_tasks}

        # out of distribution, one shard at a time
        if per_sample:
            cal_feat = {t: spectral_features(cal[t]["a"]).cpu() for t in in_tasks}
        for kind, task, N in ood_specs:
            d = gather(task, "ood", N)
            parent = d["parent"]
            s = fn(d["mean"], d["sigma"], d["truth"])
            e = cov_entry(split.covered(s))
            e_g = cov_entry(group.covered(s, np.array([parent] * len(s))))
            rec = {"kind": kind, "parent": parent, "N": N,
                   "split": e, "group": e_g,
                   "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean())}
            # weighted conformal: calibrate against this shard's *inputs* only
            if per_sample and N == 64:
                f_te = spectral_features(d["a"]).cpu()
                probe = LikelihoodRatioProbe().fit(cal_feat[parent], f_te)
                w_cal = probe.weights(cal_feat[parent])
                w_te = probe.weights(f_te)
                wc = WeightedConformal(args.alpha).fit(cal_scores[parent],
                                                       w_cal, w_te)
                rec["weighted"] = cov_entry(wc.covered(s))
                rec["weighted_diag"] = {
                    "probe_auc": probe.auc, "ess": wc.ess,
                    "n_cal": wc.n_cal, "q": wc.q,
                    "q_unweighted": float(group.q.get(parent, group.q_pooled)),
                    "w_max": float(w_cal.max()),
                    "w_median": float(np.median(w_cal))}
            out["ood"][f"{kind}/{task}/N{N}"] = rec
            del d
            torch.cuda.empty_cache()
        res["scores"][sname] = out
        print(f"[{sname}] in-dist pooled split "
              f"{100*out['in_dist']['pooled_split']['coverage']:.1f}%  "
              f"group {100*out['in_dist']['pooled_group']['coverage']:.1f}%",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
