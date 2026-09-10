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
* Every coverage carries a 95% interval. For the per-sample scores that is a
  Wilson interval; on a 512-sample OOD shard it is about +/-2.6pp, which is
  wider than the KPI band, so a single shard cannot by itself decide a clause.
  For the `pixel` score it is a **cluster bootstrap over fields**: 512 fields
  contribute 2.1M pixel outcomes, but they are not 2.1M independent
  observations, and a Wilson interval on the pixel count is about 64x too
  narrow. `pixel` coverage is reported as descriptive for that reason -- the
  exchangeability the conformal guarantee needs holds over fields, not pixels.
* The sigma floor is frozen from the **calibration** split and passed to every
  later score. Letting each shard recompute it from its own median lets a
  shifted shard widen its own band; measured at ~0.2pp in distribution and
  always in the direction of apparent coverage.
* Weighted conformal assumes p(y|x) is unchanged. Under an *operator* shift it
  is not, so those shards are marked `assumption_violated` and their weighted
  coverage is reported as a diagnostic, never as a repair.
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
from uqkit.metrics import (binom_ci, cluster_bootstrap_ci,  # noqa: E402
                           pearson, rel_l2)
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

SCORE_NAMES = ["field_max", "norm_ratio", "rel_l2", "pixel"]
# weighted conformal's covariate-shift assumption does not hold when the
# governing equation itself changed
OPERATOR_SHIFT_KINDS = {"unseen_operator", "param_oor"}


def cov_entry(covered, n_fields=None):
    """Coverage with a 95% interval; clustered by field when scored per pixel."""
    c = np.asarray(covered)
    k, n = int(c.sum()), int(len(c))
    if n_fields is not None and n_fields != n:
        lo, hi = cluster_bootstrap_ci(c.reshape(n_fields, -1))
        kind = "cluster_bootstrap_over_fields"
    else:
        lo, hi = binom_ci(k, n)
        kind = "wilson"
    return {"coverage": k / n, "n": n, "ci95": [lo, hi], "ci_kind": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/conformal.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default=None,
                    choices=["het", "cqr", "const"],
                    help="single-network sigma head (FNO2dUQ). When set, "
                         "--ckpts must be exactly one UQ checkpoint and the "
                         "interval comes from ONE forward pass.")
    args = ap.parse_args()
    single = args.sigma_source is not None
    if single and len(args.ckpts) != 1:
        ap.error("--sigma-source is a single-network mode; pass one checkpoint")

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    if single:
        model, ck = load_model(args.ckpts[0], args.device)
        if not ck["args"].get("uq"):
            raise SystemExit(f"{args.ckpts[0]} is not a UQ checkpoint")
        models = [model]
    else:
        models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]

    # ---- gather predictions -------------------------------------------------
    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        if single:
            m, s, t, a = predict_shard_single(
                models[0], blob, stats, args.device,
                sigma_source=args.sigma_source)
        else:
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
           "ckpts": args.ckpts, "scores": {}, "error": {}, "spread": {},
           "sigma_source": args.sigma_source or "ensemble_spread",
           "forward_passes_per_interval": 1 if single else len(models),
           "seed": ck["args"].get("seed"),
           "detach_uq": ck["args"].get("detach_uq")}

    # per-family error and spread-error correlation: context for every coverage
    for t in in_tasks:
        d = test[t]
        err = rel_l2(d["mean"], d["truth"])
        spr = d["sigma"].flatten(1).norm(dim=1) / d["mean"].flatten(1).norm(dim=1)
        res["error"][t] = {"rel_l2_mean": float(err.mean()),
                           "rel_l2_p90": float(err.quantile(0.9)),
                           "spread_error_pearson": pearson(spr, err)}

    # ---- per score ----------------------------------------------------------
    # The sigma floor, frozen here on the pooled calibration split and passed
    # explicitly to every score call below. Without `med=`, `_floor` falls back
    # to the median of whatever tensor it is handed, so each shifted shard would
    # set its own band width -- always in the direction of apparent coverage.
    MED = float(torch.cat([cal[t]["sigma"].flatten() for t in in_tasks]).median())
    res["sigma_floor_median"] = MED
    res["sigma_floor_frac"] = 0.05

    for sname in SCORE_NAMES:
        fn, per_sample = get_score(sname)
        out = {"per_sample": per_sample, "in_dist": {}, "ood": {}}

        cal_scores = {t: fn(cal[t]["mean"], cal[t]["sigma"], cal[t]["truth"],
                            med=MED)
                      for t in in_tasks}
        pooled_cal = torch.cat([cal_scores[t] for t in in_tasks])
        pooled_grp = np.concatenate([[t] * len(cal_scores[t]) for t in in_tasks])

        split = SplitConformal(args.alpha).fit(pooled_cal)
        group = GroupConformal(args.alpha).fit(pooled_cal, pooled_grp)
        out["q_split"] = split.q
        out["q_group"] = {k: float(v) for k, v in group.q.items()}
        # Mean half-width of the band this quantile produces, per family, in
        # the field's own units and relative to the field's own norm. Coverage
        # alone cannot distinguish a sigma that knows which samples are hard
        # from a constant one, because the conformal quantile rescales either
        # to ~90%; a wider band at equal coverage is a worse interval, and
        # `sharpness_rel` is the number that says so.
        out["sharpness"] = {}
        for t in in_tasks:
            from uqkit.conformal import _floor
            half = split.q * _floor(test[t]["sigma"], res["sigma_floor_frac"],
                                    MED)
            out["sharpness"][t] = {
                "mean_half_width": float(half.mean()),
                "sharpness_rel": float(
                    half.flatten(1).norm(dim=1).mean()
                    / test[t]["truth"].flatten(1).norm(dim=1).mean())}

        # in distribution
        te_scores = {t: fn(test[t]["mean"], test[t]["sigma"], test[t]["truth"],
                           med=MED)
                     for t in in_tasks}
        pooled_te = torch.cat([te_scores[t] for t in in_tasks])
        pooled_te_grp = np.concatenate([[t] * len(te_scores[t]) for t in in_tasks])
        nf_pool = (sum(len(test[t]["mean"]) for t in in_tasks)
                   if not per_sample else None)
        nf = {t: (len(test[t]["mean"]) if not per_sample else None)
              for t in in_tasks}
        out["in_dist"]["pooled_split"] = cov_entry(split.covered(pooled_te),
                                                   nf_pool)
        out["in_dist"]["pooled_group"] = cov_entry(
            group.covered(pooled_te, pooled_te_grp), nf_pool)
        out["in_dist"]["per_family_split"] = {
            t: cov_entry(split.covered(te_scores[t]), nf[t]) for t in in_tasks}
        out["in_dist"]["per_family_group"] = {
            t: cov_entry(group.covered(te_scores[t],
                                       np.array([t] * len(te_scores[t]))), nf[t])
            for t in in_tasks}

        # out of distribution, one shard at a time
        if per_sample:
            cal_feat = {t: spectral_features(cal[t]["a"]).cpu() for t in in_tasks}
        for kind, task, N in ood_specs:
            d = gather(task, "ood", N)
            parent = d["parent"]
            s = fn(d["mean"], d["sigma"], d["truth"], med=MED)
            n_f = None if per_sample else len(d["mean"])
            e = cov_entry(split.covered(s), n_f)
            e_g = cov_entry(group.covered(s, np.array([parent] * len(s))), n_f)
            rec = {"kind": kind, "parent": parent, "N": N,
                   "split": e, "group": e_g,
                   "assumption_violated": kind in OPERATOR_SHIFT_KINDS,
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
                    "n_cal": wc.n_cal, "q_median": wc.q,
                    "n_infinite_quantiles": wc.n_inf,
                    "test_w_max_over_median": wc.test_w_ratio,
                    "q_unweighted": float(group.q.get(parent, group.q_pooled)),
                    "w_max": float(w_cal.max()),
                    "w_median": float(np.median(w_cal)),
                    "note": ("covariate-shift validity requires p(y|x) "
                             "unchanged; this shard changes the operator"
                             if kind in OPERATOR_SHIFT_KINDS else
                             "covariate shift only")}
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
