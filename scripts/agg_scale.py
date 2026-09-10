"""Aggregate the H15 (learned width) and H16/H17 (width tolerance) runs.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_scale.py \
        --out runs/scale.json

Writes one JSON that `scripts/report.py` reads, so the disciplines these
experiments registered are enforced once rather than at each call site:

* the `insample_leak` fold never enters a headline -- it is the in-sample
  ceiling of the feature set and model class, and it is carried purely to
  separate "cannot fit" from "cannot generalize";
* every coverage travels with the interval width that produced it, and a shard
  that enters the band on a >3x wider interval is counted separately;
* the equivariant arm (H17) is paired with its baseline seed by seed and tested
  with an exact sign-flip, and `darcy_amp2` -- the registered control that must
  NOT improve -- is reported on its own rather than pooled away.
"""
from __future__ import annotations

import argparse
import glob as _glob
import itertools as _it
import json
import statistics as st
from pathlib import Path

BAND = (0.88, 0.92)
LEAK = "insample_leak"


def sign_flip(v):
    v = [float(x) for x in v]
    obs = sum(v)
    hits = sum(1 for sg in _it.product([1, -1], repeat=len(v))
               if abs(sum(a * b for a, b in zip(sg, v))) >= abs(obs))
    return hits / 2 ** len(v)


def _load(pat):
    return [json.loads(Path(p).read_text()) for p in sorted(_glob.glob(pat))]


def agg_h15(runs):
    folds = sorted(set().union(*[set(r["folds"]) for r in runs]))
    out = {"n_seeds": len(runs), "seeds": [r.get("seed") for r in runs],
           "leak_fold": LEAK,
           "leak_note": ("h fitted ON the evaluation shards: an in-sample "
                         "ceiling for this feature set and model class, never "
                         "a result"),
           "dev": runs[0]["dev"], "band": list(BAND),
           "width_flag": runs[0]["width_flag"],
           "n_cal_fit": runs[0]["n_cal_fit"], "n_cal_q": runs[0]["n_cal_q"],
           "by_fold": {}}
    for f in folds:
        per = [r["folds"][f] for r in runs if f in r["folds"]]
        if len(per) != len(runs):
            continue
        out["by_fold"][f] = {
            "leak": per[0]["leak"],
            "held_out_mechanism": per[0]["held_out_mechanism"],
            "in_band_per_seed": [p["in_band_all_shards"] for p in per],
            "in_band_deployable_per_seed": [
                p["in_band_at_deployable_width_all_shards"] for p in per],
            "in_band_median": st.median([p["in_band_all_shards"]
                                         for p in per]),
            "in_dist_coverage_median": {
                t: st.median([p["in_dist"][t]["scaled"]["coverage"]
                              for p in per])
                for t in per[0]["in_dist"]},
            "in_dist_coverage_pooled_q_median": {
                t: st.median([p["in_dist"][t]["scaled_pooled_q"]["coverage"]
                              for p in per])
                for t in per[0]["in_dist"]},
            "top_features": [c["feature"] for c in per[0]["coef_top"][:6]],
        }
    lomo = [r["leave_one_mechanism_out"] for r in runs]
    ung = {}
    for name in lomo[0]["shards"]:
        cells = [l["shards"][name] for l in lomo]
        ung[name] = {
            "mechanism": cells[0]["mechanism"],
            "parent": cells[0]["parent"],
            "rel_l2_mean": st.median([c["rel_l2_mean"] for c in cells]),
            "scaled_median": st.median([c["scaled"]["coverage"]
                                        for c in cells]),
            "ungated_median": st.median([c["ungated"]["coverage"]
                                         for c in cells]),
            "width_ratio_median": st.median([c["width_ratio_median"]
                                             for c in cells]),
            "in_band_seeds": sum(1 for c in cells if c["in_band"]),
            "in_band_deployable_seeds": sum(
                1 for c in cells if c["in_band_at_deployable_width"]),
            "width_inflated_seeds": sum(1 for c in cells
                                        if c["width_inflated"]),
        }
    lo_seed = [l["in_band"] for l in lomo]
    lo_dep = [l["in_band_at_deployable_width"] for l in lomo]
    leak_seed = out["by_fold"][LEAK]["in_band_per_seed"]
    # the ungated baseline: measured in the same runs, per seed
    base_seed = [sum(1 for v in l["shards"].values()
                     if BAND[0] <= v["ungated"]["coverage"] <= BAND[1])
                 for l in lomo]
    under = [c["underprediction_factor__uses_truth"]
             for l in lomo for c in l["shards"].values()]
    out["lomo"] = {
        "note": lomo[0]["note"],
        "n_shards": lomo[0]["n_shards"],
        "in_band_per_seed": lo_seed,
        "in_band_deployable_per_seed": lo_dep,
        "in_band_median": st.median(lo_seed),
        "in_band_range": [min(lo_seed), max(lo_seed)],
        "ungated_baseline_per_seed": base_seed,
        "vs_ungated": {
            "diff_per_seed": [a - b for a, b in zip(lo_seed, base_seed)],
            "exact_sign_flip_p": sign_flip(
                [a - b for a, b in zip(lo_seed, base_seed)])
            if any(a != b for a, b in zip(lo_seed, base_seed)) else 1.0},
        "vs_insample_ceiling": {
            "leak_per_seed": leak_seed,
            "diff_per_seed": [a - b for a, b in zip(lo_seed, leak_seed)],
            "exact_sign_flip_p": sign_flip(
                [a - b for a, b in zip(lo_seed, leak_seed)])
            if any(a != b for a, b in zip(lo_seed, leak_seed)) else 1.0},
        "underprediction_factor__uses_truth": {
            "min": min(under), "max": max(under),
            "median": st.median(under)},
        "shards": ung,
    }
    return out


def in_dist_rel_l2(pat="runs/conf_u*_het.json"):
    """Per-family in-distribution rel-L2 for the same checkpoints.

    Read from the conformal runs rather than recomputed, so the reference the
    H17 accuracy claim is compared against comes from a run on disk and is
    cited as such.
    """
    rs = _load(pat)
    if not rs:
        return None
    return {t: st.median([r["error"][t]["rel_l2_mean"] for r in rs])
            for t in rs[0]["error"]}


def agg_wtol(base, eq):
    """H16 tolerance and the H17 equivariant arm, paired seed by seed."""
    if not base:
        return None
    scores = sorted(base[0]["by_score"])
    out = {"n_seeds": len(base), "seeds": [r.get("seed") for r in base],
           "band": list(BAND), "n_seeds_eq": len(eq),
           "note": base[0]["note"], "by_score": {}}
    for sn in scores:
        b = [r["by_score"][sn] for r in base]
        names = sorted(b[0]["shards"])
        rec = {
            "tol_rel_median_over_shards": st.median(
                [r["tol_rel_median_over_shards"] for r in b]),
            "tol_rel_min_over_shards": min(r["tol_rel_min_over_shards"]
                                           for r in b),
            "tol_rel_max_over_shards": max(r["tol_rel_max_over_shards"]
                                           for r in b),
            "tol_rel_median_in_dist": st.median(
                [r["tol_rel_median_in_dist"] for r in b]),
            "in_band_per_seed": [r["in_band_shards"] for r in b],
            "in_band_in_dist_per_seed": [r["in_band_in_dist"] for r in b],
            # the framing check: in-band should hold exactly when the deployed
            # width sits inside the tolerance. Every disagreement is a hole in
            # the explanation, so the count is reported rather than described.
            "framing_disagreements_per_seed": [
                len(r["framing_disagreements"]) for r in b],
            "required_factor": {},
        }
        for n in names:
            cells = [r["shards"][n] for r in b]
            rec["required_factor"][n] = {
                "base_median": st.median(
                    [1.0 / c["deployed_width_over_ideal"] for c in cells]),
                "rel_l2_base_median": st.median([c["rel_l2_mean"]
                                                 for c in cells]),
                "tol_rel_median": st.median([c["tol_rel"] for c in cells]),
                "coverage_ungated_median": st.median(
                    [c["coverage_ungated"] for c in cells]),
                "in_band_seeds": sum(1 for c in cells if c["in_band"]),
            }
        if eq:
            # Pairing is by seed, not by position. A missing arm would
            # otherwise silently compare seed i of one arm against seed j of
            # the other and the sign-flip test would be meaningless.
            sb = [r.get("seed") for r in base]
            se = [r.get("seed") for r in eq]
            if sb != se:
                raise SystemExit(
                    f"width-tolerance arms cover different seeds "
                    f"(base {sb}, equivariant {se}); refusing to pair them")
            e = [r["by_score"][sn] for r in eq]
            rec["equivariant"] = {
                "in_band_per_seed": [r["in_band_shards"] for r in e],
                "in_band_in_dist_per_seed": [r["in_band_in_dist"] for r in e],
                "tol_rel_median_over_shards": st.median(
                    [r["tol_rel_median_over_shards"] for r in e]),
                "framing_disagreements_per_seed": [
                    len(r["framing_disagreements"]) for r in e],
            }
            diffs = [a - c for a, c in zip(rec["equivariant"]
                                           ["in_band_per_seed"],
                                           rec["in_band_per_seed"])]
            rec["equivariant"]["vs_base"] = {
                "diff_per_seed": diffs,
                "exact_sign_flip_p": (sign_flip(diffs)
                                      if any(d != 0 for d in diffs) else 1.0)}
            for n in names:
                cells = [r["shards"][n] for r in e]
                f = rec["required_factor"][n]
                f["eq_median"] = st.median(
                    [1.0 / c["deployed_width_over_ideal"] for c in cells])
                f["eq_coverage_ungated_median"] = st.median(
                    [c["coverage_ungated"] for c in cells])
                f["eq_in_band_seeds"] = sum(1 for c in cells if c["in_band"])
                f["rel_l2_eq_median"] = st.median([c["rel_l2_mean"]
                                                   for c in cells])
                f["rel_l2_ratio"] = (f["rel_l2_eq_median"]
                                     / f["rel_l2_base_median"]
                                     if f["rel_l2_base_median"] else None)
                f["factor_change"] = (f["eq_median"] / f["base_median"]
                                      if f["base_median"] else None)
            # H17 turned out to be an ACCURACY result as well as a width one,
            # and that was free: the shard rel-L2 is already recorded in both
            # arms, so no rerun was needed to notice it.
            idl = in_dist_rel_l2()
            if idl:
                rec["equivariant"]["in_dist_rel_l2_same_ckpts"] = idl
                acc = {}
                for n in names:
                    f = rec["required_factor"][n]
                    ref = idl.get(b[0]["shards"][n]["parent"])
                    acc[n] = {
                        "rel_l2_base": f["rel_l2_base_median"],
                        "rel_l2_eq": f["rel_l2_eq_median"],
                        "ratio": f["rel_l2_ratio"],
                        "in_dist_reference": ref,
                        # equal to the in-distribution error is the strong
                        # claim scale-equivariance predicts; 1.15x of it is
                        # the loose reading, and both are recorded
                        "eq_over_in_dist": (f["rel_l2_eq_median"] / ref
                                            if ref else None)}
                rec["equivariant"]["accuracy"] = acc
                rec["equivariant"]["max_abs_rel_l2_change_off_amp"] = max(
                    abs((acc[n]["ratio"] or 1.0) - 1.0)
                    for n in names if "amp2" not in n)
            # the required dynamic range is the headline of H16/H17
            rec["equivariant"]["required_range_base"] = max(
                f["base_median"] for f in rec["required_factor"].values())
            rec["equivariant"]["required_range_eq"] = max(
                f["eq_median"] for f in rec["required_factor"].values())
            # the registered control: darcy_amp2 must NOT improve
            ctl = [n for n in names if "darcy_amp2" in n]
            if ctl:
                f = rec["required_factor"][ctl[0]]
                rec["equivariant"]["control_darcy_amp2"] = {
                    "shard": ctl[0], "base": f["base_median"],
                    "eq": f["eq_median"], "change": f["factor_change"],
                    "must_not_improve": True}
        out["by_score"][sn] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h15-glob", default="runs/scale_u*_het.json")
    ap.add_argument("--wtol-base-glob", default="runs/wtol_base_u*_het.json")
    ap.add_argument("--wtol-eq-glob", default="runs/wtol_eq_u*_het.json")
    ap.add_argument("--h18-glob", default="runs/scaleq_u*_het.json",
                    help="H18: the same width model fitted on top of the H17 "
                         "equivariant predictor. Paired to the H15 arm by "
                         "checkpoint, so the sign-flip test is exact.")
    ap.add_argument("--out", default="runs/scale.json")
    args = ap.parse_args()

    res = {}
    h15 = _load(args.h15_glob)
    if h15:
        res["h15"] = agg_h15(h15)
        L = res["h15"]["lomo"]
        print(f"[H15] LOMO in band {L['in_band_per_seed']} "
              f"(median {L['in_band_median']}/{L['n_shards']}), "
              f"ungated baseline {L['ungated_baseline_per_seed']}, "
              f"p={L['vs_ungated']['exact_sign_flip_p']:.4f}")
        print(f"[H15] in-sample ceiling {L['vs_insample_ceiling']['leak_per_seed']}"
              f", LOMO vs ceiling p="
              f"{L['vs_insample_ceiling']['exact_sign_flip_p']:.4f}")
    h18 = _load(args.h18_glob)
    if h18 and h15:
        res["h18"] = agg_h15(h18)
        a = res["h18"]["lomo"]["in_band_per_seed"]
        b = res["h15"]["lomo"]["in_band_per_seed"]
        if len(a) == len(b):
            d = [x - y for x, y in zip(a, b)]
            res["h18"]["vs_h15"] = {
                "h18_per_seed": a, "h15_per_seed": b, "diff_per_seed": d,
                "exact_sign_flip_p": (sign_flip(d) if any(d) else 1.0),
                "note": ("paired by checkpoint: both arms use runs/u{k} and "
                         "an identical dev suite, so the only difference is "
                         "the equivariant wrapper"),
            }
            print(f"[H18] LOMO {a} vs H15 {b}, diff {d}, "
                  f"p={res['h18']['vs_h15']['exact_sign_flip_p']:.4f}")
            print(f"[H18] in-sample ceiling "
                  f"{res['h18']['lomo']['vs_insample_ceiling']['leak_per_seed']}")
    base, eq = _load(args.wtol_base_glob), _load(args.wtol_eq_glob)
    w = agg_wtol(base, eq)
    if w:
        res["wtol"] = w
        for sn, r in w["by_score"].items():
            line = (f"[H16/{sn}] tolerance median "
                    f"{r['tol_rel_median_over_shards']*100:.2f}%, in band "
                    f"{r['in_band_per_seed']}, framing disagreements "
                    f"{r['framing_disagreements_per_seed']}")
            if "equivariant" in r:
                e = r["equivariant"]
                line += (f" | H17 eq in band {e['in_band_per_seed']}, "
                         f"range {e['required_range_base']:.1f}x -> "
                         f"{e['required_range_eq']:.1f}x, "
                         f"p={e['vs_base']['exact_sign_flip_p']:.4f}")
            print(line)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
