"""Aggregate H30: what clause 1 under covariate shift costs in labels.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h30.py

Reads runs/label_budget_u*_het.json (8 checkpoints) and emits
runs/h30_labels.json.

THE READING THAT MATTERS IS `frac_draws_in_band`. `coverage` in those files is
an average over `repeats` independent labelled draws, and split conformal is
exactly valid marginally, so that average sits at 1-alpha whenever the quantile
is finite -- "the mean is in band" is nearly a tautology and an earlier version
of this line reported 49/49 shards in band at k=9 on the strength of it. A
deployment gets ONE draw, whose coverage is distribution-free Beta(k+1-l, l)
with l = floor((k+1)*alpha). This aggregator prints the measured per-draw
fraction beside that analytic law, because the agreement between them is the
result: the price is set by the calibrator, not by this surrogate.

Every number here consumes labels from the shard it certifies, so none of it is
evidence for the unlabelled KPI clause. It is a price list.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
BAND = (0.88, 0.92)
ALPHA = 0.1


def beta_pred(k, alpha=ALPHA):
    """P(single split-conformal draw at k labels lands in band). Analytic."""
    l = int(np.floor((k + 1) * alpha))
    if l < 1:
        return None                      # quantile is +inf; coverage is 1
    a, b = k + 1 - l, l
    return float(beta.cdf(BAND[1], a, b) - beta.cdf(BAND[0], a, b))


def main():
    runs = [json.loads(f.read_text())
            for f in sorted((ROOT / "runs").glob("label_budget_u*_het.json"))]
    if not runs:
        raise SystemExit("no H30 runs")
    ks = runs[0]["ks"]
    out = {"n_seeds": len(runs), "seeds": [r["ckpts"][0] for r in runs],
           "alpha": runs[0]["alpha"], "band": list(BAND), "ks": ks,
           "repeats": runs[0]["repeats"],
           "clause_eligible": False,
           "reading_warning": runs[0].get("reading_warning"),
           "oracle_warning": runs[0].get("oracle_warning"),
           "by_k": {}, "arm_c": {}}

    def ood(r):
        return [v for v in r["shards"].values() if v["kind"] != "in_dist"]

    n_ood = len(ood(runs[0]))
    out["n_ood_shards"] = n_ood

    for k in ks:
        key = f"k{k}"
        a_frac, b_frac, a_mean_inband = [], [], []
        for r in runs:
            vs = [v["curve"][key] for v in ood(r) if key in v["curve"]]
            if not vs:
                continue
            a_frac.append(st.median([v["frac_draws_in_band"] for v in vs]))
            b_frac.append(st.median([v["scale_frac_draws_in_band"]
                                     for v in vs]))
            a_mean_inband.append(sum(1 for v in vs if v["in_band_of_mean"]))
        if not a_frac:
            continue
        out["by_k"][key] = {
            "k": k,
            "armA_frac_draws_in_band_median": st.median(a_frac),
            "armA_frac_draws_in_band_per_seed": a_frac,
            "armA_beta_prediction": beta_pred(k),
            "armB_frac_draws_in_band_median": st.median(b_frac),
            "armB_frac_draws_in_band_per_seed": b_frac,
            "armA_shards_whose_MEAN_is_in_band_median": st.median(a_mean_inband),
        }

    # arm C: shape preservation, the ceiling of the cheap scalar estimator
    c_in, c_never = [], []
    per_shard = {}
    for r in runs:
        vs = ood(r)
        c_in.append(sum(1 for v in vs if v["scale_limit_in_band"]))
        c_never.append(sum(1 for v in vs if v["scale_k_min_in_band"] is None))
    for name in sorted(v for v in runs[0]["shards"] if
                       runs[0]["shards"][v]["kind"] != "in_dist"):
        per_shard[name] = {
            "arm_c_coverage_median": st.median(
                [r["shards"][name]["scale_limit_coverage"] for r in runs]),
            "median_ratio_median": st.median(
                [r["shards"][name]["scale_median_ratio"] for r in runs]),
            "arm_c_in_band_on_n_seeds": sum(
                1 for r in runs if r["shards"][name]["scale_limit_in_band"]),
        }
    out["arm_c"] = {
        "in_band_per_seed": c_in, "in_band_median": st.median(c_in),
        "armB_never_in_band_per_seed": c_never,
        "armB_never_in_band_median": st.median(c_never),
        "per_shard": per_shard,
        "note": "arm C is arm B's k->infinity limit, scored on the samples "
                "that formed it. Out of band here means the shifted score "
                "distribution is NOT the calibration one times a constant, so "
                "no label budget reaches the band by a median-ratio scale."}

    # the extrapolated price, analytic and therefore labelled as such
    out["price_extrapolated_analytic"] = {
        f"k{k}": beta_pred(k) for k in (128, 256, 512, 1024, 2048)}
    out["price_note"] = ("beta_pred is the distribution-free law for split "
                         "conformal, NOT a measurement on this repo's data. "
                         "It is quoted because the measured armA fractions "
                         "track it; the measured grid stops at the largest k "
                         "each shard can afford.")

    p = ROOT / "runs" / "h30_labels.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"H30 aggregate: {out['n_seeds']} checkpoints, {n_ood} OOD shards, "
          f"repeats={out['repeats']}\n")
    print("P(ONE k-label draw lands in 90+/-2%), median over shards:")
    print(f"{'k':>5s} {'arm A':>8s} {'Beta law':>9s} {'arm B':>8s}   "
          f"{'[arm A shards whose MEAN is in band]':>36s}")
    for key, d in out["by_k"].items():
        bp = d["armA_beta_prediction"]
        print(f"{d['k']:5d} {d['armA_frac_draws_in_band_median']:8.3f} "
              f"{(f'{bp:.3f}' if bp is not None else '   +inf'):>9s} "
              f"{d['armB_frac_draws_in_band_median']:8.3f}   "
              f"{d['armA_shards_whose_MEAN_is_in_band_median']:36.0f}")
    print(f"\narm C (shape preservation) in band: median "
          f"{out['arm_c']['in_band_median']:.0f}/{n_ood}  "
          f"per seed {c_in}")
    print(f"arm B never in band at any k: median "
          f"{out['arm_c']['armB_never_in_band_median']:.0f}/{n_ood}  "
          f"per seed {c_never}")
    print("\nanalytic extrapolation of the price (NOT measured here):")
    for k, v in out["price_extrapolated_analytic"].items():
        print(f"  {k:>6s} -> P(in band) = {v:.3f}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
