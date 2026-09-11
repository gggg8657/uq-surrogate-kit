"""Clause 3 across seeds, and the threshold sensitivity of its conditional pass.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_clause3.py \
        --out runs/clause3.json

The published clause-3 numbers -- 47/49 strict and 33/33 conditional -- were
measured on **one checkpoint** (`runs/consistency_uq.json`, `runs/u0`). The
brief's seed-count rule applies to a passing clause exactly as it does to a
failing one, so this aggregates the 8-seed base arm and reports two things the
single draw could not:

* whether the strict count reproduces, or whether 47/49 was a lucky draw;
* whether the **conditional** reading is stable, which is a question about its
  *denominator*. That reading admits a shard when its error degrades past a
  threshold. Shards sitting near the threshold move in and out from seed to
  seed, so the denominator is itself a random variable and a ratio quoted from
  one draw can be the most favourable of eight.

Both thresholds this repo has quoted are reported side by side rather than one
being chosen: `--thresholds 1.06 1.10`. Picking whichever threshold produces
the better ratio, after seeing them, is the move the brief forbids -- so both
are printed with their populations and the reader picks.

The equivariant arm is aggregated the same way when its files exist, so the
H21 comparison lands in this file rather than in a second one.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics as st
from pathlib import Path


def _load(pat):
    return [json.loads(Path(p).read_text()) for p in sorted(_glob.glob(pat))]


def _degradation(cell):
    """Error under shift relative to the same checkpoint in distribution."""
    return cell["rel_l2_mean"] / max(cell["rel_l2_in_dist"], 1e-12)


def arm(runs, det, thresholds):
    if not runs:
        return None
    out = {"n_seeds": len(runs), "seeds": [r.get("seed") for r in runs],
           "detector": det, "equivariant": bool(runs[0].get("equivariant")),
           "strict": {}, "conditional": {}, "unstable_shards": {}}

    strict = [r["summary"][det]["n_ge_0p9"] for r in runs]
    total = runs[0]["summary"][det]["n_total_shards"]
    mins = [r["summary"][det]["min"] for r in runs]
    out["strict"] = {
        "n_ge_0p9_per_seed": strict, "n_total_shards": total,
        "median": st.median(strict), "range": [min(strict), max(strict)],
        "identical_on_every_seed": len(set(strict)) == 1,
        "min_auroc_per_seed": mins,
        "min_auroc_range": [min(mins), max(mins)],
    }

    for th in thresholds:
        pairs, offenders = [], {}
        for r in runs:
            n = tot = 0
            for k, v in r["shards"].items():
                if _degradation(v) <= th:
                    continue
                tot += 1
                if v["auroc"][det]["auroc"] >= 0.9:
                    n += 1
                else:
                    offenders.setdefault(k, []).append(
                        {"seed": r.get("seed"),
                         "degradation": _degradation(v),
                         "auroc": v["auroc"][det]["auroc"]})
            pairs.append([n, tot])
        dens = [p[1] for p in pairs]
        out["conditional"][f"{th:g}"] = {
            "pairs_per_seed": pairs,
            "denominator_range": [min(dens), max(dens)],
            "denominator_is_stable": len(set(dens)) == 1,
            "seeds_with_a_failure_inside": sum(1 for a, b in pairs if a < b),
            "clean_on_every_seed": all(a == b for a, b in pairs),
            "shards_failing_inside": {
                k: {"n_seeds_inside": len(v),
                    "degradation_range": [min(x["degradation"] for x in v),
                                          max(x["degradation"] for x in v)],
                    "auroc_range": [min(x["auroc"] for x in v),
                                    max(x["auroc"] for x in v)]}
                for k, v in offenders.items()},
        }

    # every shard that is below 0.9 on at least one seed, with its spread
    per = {}
    for r in runs:
        for k, v in r["shards"].items():
            per.setdefault(k, []).append(v["auroc"][det]["auroc"])
    for k, vs in per.items():
        if min(vs) < 0.9:
            out["unstable_shards"][k] = {
                "n_seeds_below_0p9": sum(1 for v in vs if v < 0.9),
                "auroc_range": [min(vs), max(vs)],
            }

    # the degradation of every shard that crosses any threshold on some seed
    # but not on others -- the boundary population, which is what makes a
    # conditional ratio move
    deg = {}
    for r in runs:
        for k, v in r["shards"].items():
            deg.setdefault(k, []).append(_degradation(v))
    out["boundary_shards"] = {
        k: {"degradation_range": [min(v), max(v)],
            "crosses_per_threshold": {f"{th:g}": sum(1 for d in v if d > th)
                                      for th in thresholds}}
        for k, v in deg.items()
        if any(0 < sum(1 for d in v if d > th) < len(v) for th in thresholds)
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-glob", default="runs/cons3_base_u*_het.json")
    ap.add_argument("--eq-glob", default="runs/cons3_eq_u*_het.json")
    ap.add_argument("--detector", default="combo",
                    help="the detector the published headline uses")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[1.06, 1.10],
                    help="degradation thresholds for the conditional reading. "
                         "BOTH values this repo has quoted are reported; "
                         "choosing one after seeing which reads better is the "
                         "move the brief forbids.")
    ap.add_argument("--out", default="runs/clause3.json")
    args = ap.parse_args()

    base, eq = _load(args.base_glob), _load(args.eq_glob)
    res = {"detector": args.detector, "thresholds": args.thresholds,
           "published_single_checkpoint": {
               "source": "runs/consistency_uq.json",
               "note": "47/49 strict and 33/33 conditional, one checkpoint"},
           "base": arm(base, args.detector, args.thresholds),
           "equivariant": arm(eq, args.detector, args.thresholds)}
    if res["base"] and res["equivariant"]:
        a = res["base"]["strict"]["n_ge_0p9_per_seed"]
        b = res["equivariant"]["strict"]["n_ge_0p9_per_seed"]
        if len(a) == len(b):
            import itertools as _it
            d = [x - y for x, y in zip(b, a)]
            obs = sum(d)
            hits = sum(1 for sg in _it.product([1, -1], repeat=len(d))
                       if abs(sum(p * q for p, q in zip(sg, d))) >= abs(obs))
            res["eq_vs_base_strict"] = {
                "base_per_seed": a, "eq_per_seed": b, "diff_per_seed": d,
                "exact_sign_flip_p": (hits / 2 ** len(d) if any(d) else 1.0)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    for nm in ("base", "equivariant"):
        a = res[nm]
        if not a:
            print(f"[{nm}] no runs yet")
            continue
        s = a["strict"]
        print(f"[{nm}] strict {s['n_ge_0p9_per_seed']}/{s['n_total_shards']} "
              f"— identical on every seed: {s['identical_on_every_seed']}")
        for th, c in a["conditional"].items():
            print(f"[{nm}] conditional >{th}x: {c['pairs_per_seed']} "
                  f"| denominator stable: {c['denominator_is_stable']} "
                  f"| clean on every seed: {c['clean_on_every_seed']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
