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


def width_accuracy(arms, tol):
    """How far each arm's emitted width is from the width the band requires.

    H16 states the clause without any uncertainty method in it: the widths
    giving coverage in [0.88, 0.92] span [Q0.88(S), Q0.92(S)], so there is a
    required relative tolerance on the width. Every arm records, per shard,
    `underprediction_factor__uses_truth` = Q0.90(S) / median(q*h) -- the factor
    by which the emitted width misses the width that would have given exactly
    0.90. This puts every method on that one axis.

    Two caveats travel with the table and are not optional:

    * the arms do not all evaluate the same shards -- the residual arm runs on
      the 24 with a cheap operator apply -- so the `within_tol` counts are
      comparable to each arm's own shard count and not across arms;
    * `tol` is a median over shards while in-band is decided per shard, so
      `within_tol` is expected to track the in-band count and not to equal it.
      That it tracks at all is the check that H16's framing is predictive
      rather than merely descriptive, and the two columns are printed together
      so a reader can see the correspondence break if it ever does.
    """
    out = {"required_tol_rel": tol, "note": width_accuracy.__doc__.strip(),
           "by_arm": {}}
    for name, runs in arms.items():
        if not runs:
            continue
        shards = sorted(runs[0]["leave_one_mechanism_out"]["shards"])
        errs = {}
        for n in shards:
            f = st.median([r["leave_one_mechanism_out"]["shards"][n]
                           ["underprediction_factor__uses_truth"]
                           for r in runs])
            errs[n] = abs(f - 1.0)
        med = st.median(errs.values())
        out["by_arm"][name] = {
            "n_shards": len(shards),
            "median_abs_width_error": med,
            "x_required_tolerance": med / tol if tol else None,
            "within_tol": sum(1 for e in errs.values() if e <= tol),
            "in_band_median": st.median(
                [r["leave_one_mechanism_out"]["in_band"] for r in runs]),
            "per_shard_abs_width_error": errs,
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
    ap.add_argument("--h19-glob", default="runs/scalena_u*_het.json",
                    help="H19: the H15 arm with the two input-amplitude "
                         "features ablated. Paired against BOTH the H15 and "
                         "the H18 arms by checkpoint, because the question is "
                         "which of the two it resembles.")
    ap.add_argument("--h20-glob", default="runs/scalepf_u*_het.json",
                    help="H20: one width model PER FAMILY instead of one "
                         "pooled fit with family one-hots. Paired against H15 "
                         "(the arm it modifies) and against H19 (the "
                         "amplitude ablation), because the diagnostic "
                         "question is whether H15's Darcy-ladder gain "
                         "survives a fit that never sees another family's "
                         "amplitude rows.")
    ap.add_argument("--h22-glob", default="runs/scalerf_u*_het.json",
                    help="H22: two residual features added to z. Forces a "
                         "restriction to families with a cheap operator "
                         "apply, so it is compared BOTH against the deployed "
                         "arm restricted to the same shards and against a "
                         "fit-restricted control.")
    ap.add_argument("--h22ctl-glob", default="runs/scalerc_u*_het.json",
                    help="H22 control: same families, same fitting set, "
                         "WITHOUT the residual features. H22 minus this is "
                         "exactly the two residual columns.")
    ap.add_argument("--h22-res-glob", default="runs/scalerf_u*_het.json",
                    help="H22: the width model with two PDE-residual features "
                         "added. Restricted to the three families with a cheap "
                         "operator apply -- 24 of the 32 covariate shards.")
    ap.add_argument("--h22-ctl-glob", default="runs/h22_ctl_u*_het.json",
                    help="H22 CONTROL: the same 3-family restriction WITHOUT "
                         "the residual features. The residual arm must be "
                         "compared against this and not against the 5-family "
                         "H15 arm, or the features and the fitting set change "
                         "together.")
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
    h19 = _load(args.h19_glob)
    if h19 and h15:
        res["h19"] = agg_h15(h19)
        a = res["h19"]["lomo"]["in_band_per_seed"]
        cmp = {}
        for name, other in (("h15", res.get("h15")), ("h18", res.get("h18"))):
            if not other:
                continue
            b = other["lomo"]["in_band_per_seed"]
            if len(a) != len(b):
                continue
            d = [x - y for x, y in zip(a, b)]
            cmp[name] = {"other_per_seed": b, "diff_per_seed": d,
                         "exact_sign_flip_p": (sign_flip(d) if any(d)
                                               else 1.0)}
        res["h19"]["vs"] = {
            "h19_per_seed": a, "comparisons": cmp,
            "dropped_features": h19[0].get("dropped_features"),
            "n_features": h19[0].get("n_features"),
            "note": ("H19 answers which explanation of H18 holds: resembling "
                     "H18 means amplitude was the only large learnable signal "
                     "and H17 removes it exactly; resembling H15 means the "
                     "equivariant fit merely lost dynamic range in its "
                     "target."),
        }
        print(f"[H19] LOMO {a} (dropped {res['h19']['vs']['dropped_features']},"
              f" {res['h19']['vs']['n_features']} features)")
        for k, v in cmp.items():
            print(f"[H19] vs {k.upper()} {v['other_per_seed']}: diff "
                  f"{v['diff_per_seed']}, p={v['exact_sign_flip_p']:.4f}")
    h20 = _load(args.h20_glob)
    if h20 and h15:
        res["h20"] = agg_h15(h20)
        a = res["h20"]["lomo"]["in_band_per_seed"]
        cmp = {}
        for name in ("h15", "h19"):
            other = res.get(name)
            if not other:
                continue
            b = other["lomo"]["in_band_per_seed"]
            if len(a) != len(b):
                continue
            d = [x - y for x, y in zip(a, b)]
            cmp[name] = {"other_per_seed": b, "diff_per_seed": d,
                         "exact_sign_flip_p": (sign_flip(d) if any(d)
                                               else 1.0)}
        # the registered diagnostic: does the Darcy ladder keep the coverage
        # the pooled fit gave it, once the fit cannot see another family?
        LAD = [n for n in res["h20"]["lomo"]["shards"]
               if res["h20"]["lomo"]["shards"][n]["parent"] == "darcy"
               and res["h20"]["lomo"]["shards"][n]["mechanism"] == "alpha"]
        res["h20"]["vs"] = {
            "h20_per_seed": a, "comparisons": cmp,
            "per_family_h": h20[0].get("per_family_h"),
            "darcy_alpha_shards": {
                n: {"ungated": res["h20"]["lomo"]["shards"][n]
                    ["ungated_median"],
                    "h20": res["h20"]["lomo"]["shards"][n]["scaled_median"],
                    "h15": (res["h15"]["lomo"]["shards"][n]["scaled_median"]
                            if n in res["h15"]["lomo"]["shards"] else None),
                    "in_band_seeds": res["h20"]["lomo"]["shards"][n]
                    ["in_band_seeds"]}
                for n in sorted(LAD)},
            "note": ("H20 tests whether H15's Darcy gain was a Darcy "
                     "difficulty model or spillover from the loss mass of "
                     "other families' amplitude rows. A per-family fit cannot "
                     "see another family, so if the ladder holds it was the "
                     "former."),
        }
        print(f"[H20] LOMO {a} (per_family_h="
              f"{res['h20']['vs']['per_family_h']}), median "
              f"{res['h20']['lomo']['in_band_median']}/"
              f"{res['h20']['lomo']['n_shards']}")
        for k, v in cmp.items():
            print(f"[H20] vs {k.upper()} {v['other_per_seed']}: diff "
                  f"{v['diff_per_seed']}, p={v['exact_sign_flip_p']:.4f}")
    h22 = _load(args.h22_glob)
    if h22:
        # Control (a) of the two H22 registered: the DEPLOYED 5-family arm
        # scored on the same shards the residual arm could run on. It answers
        # "does the residual arm beat what is shipped today, where it can run",
        # and it deliberately does NOT attribute the difference -- the fitting
        # set moves with the features. Control (b), which does attribute it, is
        # the fit-restricted arm and lives under `h22` below. Both were
        # registered; they are different questions and they get different keys.
        res["h22_vs_deployed"] = agg_h15(h22)
        names = sorted(h22[0]["leave_one_mechanism_out"]["shards"])
        a = res["h22_vs_deployed"]["lomo"]["in_band_per_seed"]
        cmp = {}

        def _restricted(runs_, label):
            """In-band count per seed over ONLY the shards H22 could evaluate.

            Taken from the H22 runs themselves rather than hardcoded, so the
            subset cannot drift away from what the residual arm actually ran
            on. A 24-shard arm and a 32-shard arm are not comparable, and this
            is the only place that restriction is applied.
            """
            out_ = []
            for r in runs_:
                sh = r["leave_one_mechanism_out"]["shards"]
                if not set(names) <= set(sh):
                    return None
                out_.append(sum(1 for n in names
                                if BAND[0] <= sh[n]["scaled"]["coverage"]
                                <= BAND[1]))
            return out_

        for label, runs_ in (("h15_eval_restricted", h15),
                             ("h22_control_fit_restricted",
                              _load(args.h22ctl_glob))):
            if not runs_ or len(runs_) != len(h22):
                continue
            b = _restricted(runs_, label)
            if b is None:
                continue
            d = [x - y for x, y in zip(a, b)]
            cmp[label] = {"other_per_seed": b, "diff_per_seed": d,
                          "exact_sign_flip_p": (sign_flip(d) if any(d)
                                                else 1.0)}
        # the ungated arm on the same 24 shards, from the H22 runs themselves
        ung24 = [sum(1 for n in names
                     if BAND[0] <= r["leave_one_mechanism_out"]["shards"][n]
                     ["ungated"]["coverage"] <= BAND[1]) for r in h22]
        coef = {}
        for r in h22:
            for c in r["folds"]["all"]["coef_top"]:
                coef.setdefault(c["feature"], []).append(c["coef"])
        res["h22_vs_deployed"]["vs"] = {
            "h22_per_seed": a,
            "n_shards_evaluated": len(names),
            "families": h22[0].get("families"),
            "ungated_on_same_shards_per_seed": ung24,
            "comparisons": cmp,
            "residual_coef_median": {
                k: st.median(v) for k, v in coef.items()
                if k in ("log_resid", "log_consist")},
            "residual_coef_seeds_in_top12": {
                k: len(v) for k, v in coef.items()
                if k in ("log_resid", "log_consist")},
            "largest_coef": max(
                ((k, st.median(v)) for k, v in coef.items()),
                key=lambda kv: abs(kv[1]), default=(None, None)),
            "note": ("H22 adds two residual features AND is forced onto the "
                     "families with a cheap apply. `h15_eval_restricted` is "
                     "the deployed arm scored on the same shards; "
                     "`h22_control_fit_restricted` holds the fitting set "
                     "fixed and removes only the two columns, so it is the "
                     "one that attributes a gain."),
        }
        print(f"[H22-vs-deployed] LOMO {a} over {len(names)} shards "
              f"({res['h22_vs_deployed']['vs']['families']}); ungated same shards "
              f"{ung24}")
        for k, v in cmp.items():
            print(f"[H22-vs-deployed] vs {k} {v['other_per_seed']}: diff "
                  f"{v['diff_per_seed']}, p={v['exact_sign_flip_p']:.4f}")
        rc = res["h22_vs_deployed"]["vs"]["residual_coef_median"]
        if rc:
            print(f"[H22-vs-deployed] residual coefs {({k: round(v, 4) for k, v in rc.items()})}"
                  f" vs largest {res['h22_vs_deployed']['vs']['largest_coef']}")
    res22, ctl22 = _load(args.h22_res_glob), _load(args.h22_ctl_glob)
    if res22 and ctl22 and len(res22) == len(ctl22):
        R, C = agg_h15(res22), agg_h15(ctl22)
        a = R["lomo"]["in_band_per_seed"]
        b = C["lomo"]["in_band_per_seed"]
        d = [x - y for x, y in zip(a, b)]
        # prediction 1: does either residual feature reach the top coefficients?
        coefs = [c for r in res22 for c in r["folds"]["all"]["coef_top"]]
        rank = {}
        for r in res22:
            for i, c in enumerate(r["folds"]["all"]["coef_top"]):
                if c["feature"] in ("log_consist", "log_resid"):
                    rank.setdefault(c["feature"], []).append(
                        {"rank": i + 1, "coef": c["coef"]})
        # prediction 3 leak test: are the gains only on the amplitude shards?
        per_shard = {}
        for n, v in R["lomo"]["shards"].items():
            cv = C["lomo"]["shards"].get(n)
            if cv is None:
                continue
            per_shard[n] = {"mechanism": v["mechanism"], "parent": v["parent"],
                            "ungated": v["ungated_median"],
                            "ctl": cv["scaled_median"],
                            "res": v["scaled_median"],
                            "delta": v["scaled_median"] - cv["scaled_median"]}
        res["h22"] = {
            "residual_arm": R, "control_arm": C,
            "n_shards": R["lomo"]["n_shards"],
            "families": res22[0].get("families"),
            "note": ("both arms are restricted to the families with a cheap "
                     "operator apply, so they differ in exactly the two "
                     "residual columns. Compared against the 5-family H15 arm "
                     "instead, the features and the fitting set would change "
                     "together."),
            "vs_control": {
                "residual_per_seed": a, "control_per_seed": b,
                "diff_per_seed": d,
                "exact_sign_flip_p": (sign_flip(d) if any(d) else 1.0)},
            "residual_feature_rank": rank,
            "per_shard": per_shard,
        }
        print(f"[H22] residual {a} vs control {b} (both /{R['lomo']['n_shards']}"
              f", 3 families), diff {d}, "
              f"p={res['h22']['vs_control']['exact_sign_flip_p']:.4f}")
        print(f"[H22] residual features in top-12 coefficients: "
              f"{ {k: len(v) for k, v in rank.items()} } of {len(res22)} seeds")
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
    # every arm on H16's single axis: how far the emitted width is from the
    # width the band requires
    if res.get("wtol"):
        tol = res["wtol"]["by_score"]["field_max"]["tol_rel_median_over_shards"]
        arms = {"h15": h15, "h18": _load(args.h18_glob),
                "h19": _load(args.h19_glob), "h20": _load(args.h20_glob),
                "h22": _load(args.h22_glob),
                "h22_control": _load(args.h22ctl_glob)}
        res["width_accuracy"] = width_accuracy(
            {k: v for k, v in arms.items() if v}, tol)
        wa = res["width_accuracy"]
        print(f"[WIDTH] required tolerance {wa['required_tol_rel']*100:.2f}%")
        for k, v in wa["by_arm"].items():
            print(f"[WIDTH] {k:12s} median |width error| "
                  f"{v['median_abs_width_error']*100:6.1f}%  "
                  f"= {v['x_required_tolerance']:5.1f}x required   "
                  f"within tol {v['within_tol']:2d}/{v['n_shards']}   "
                  f"in band {v['in_band_median']:.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
