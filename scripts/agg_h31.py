"""H31: aggregate the one-sided conservative reading over seeds.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h31.py \
        --out runs/h31_onesided.json

Reads `runs/oseq_u*_het.json` (equivariant path) and `runs/osbase_u*_het.json`
(base path), each written by `scripts/eval_scale_target.py --onesided`.

Two readings of clause 1, side by side, never one instead of the other:

* **R2** -- the clause. Coverage in [0.88, 0.92]. Two-sided.
* **R1** -- the conventional conformal reading. Coverage >= 0.90 with a FINITE
  interval, reported only ever beside the realized median width multiplier.

`R1` is reported at every constant in the ladder. The `c_dev_p*` constants are
percentiles of the DEVELOPMENT shards' s*, so a coverage number at one of them
is a method result; the `c_fixed_*` constants are a geometric ladder for
reading the frontier and are labelled `dev_chosen: false`.

The overpayment column is the point of the exercise: `c_dev_p100 / max_eval_s*`
is how much wider the interval has to be when the constant is chosen without
seeing the test shifts, against the oracle constant that would just suffice.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BAND = (0.88, 0.92)
TARGET = 0.90
ARMS = {"eq": "runs/oseq_u*_het.json", "base": "runs/osbase_u*_het.json"}


def spread(v):
    v = [float(x) for x in v]
    return {"min": min(v), "median": float(np.median(v)), "max": max(v),
            "n_seeds": len(v), "per_seed": v}


def sign_flip_p(diff):
    """Exact paired sign-flip test, two-sided, over all 2^n sign assignments."""
    d = np.asarray([x for x in diff], dtype=float)
    n = len(d)
    obs = abs(d.mean())
    hits = 0
    for m in range(1 << n):
        s = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(n)])
        if abs((d * s).mean()) >= obs - 1e-15:
            hits += 1
    return hits / (1 << n)


def arm_summary(files):
    seeds = [json.load(open(f)) for f in files]
    for d in seeds:
        if not d.get("onesided", {}).get("enabled"):
            raise SystemExit("a file in this arm was not written with "
                             "--onesided; the arm is not aggregatable")
    names = sorted(seeds[0]["shards"])
    for d in seeds:
        if sorted(d["shards"]) != names:
            raise SystemExit("shard sets differ across seeds in one arm")
    ladder = sorted(seeds[0]["onesided"]["c_ladder"])
    out = {"n_seeds": len(seeds), "n_shards": len(names),
           "files": [Path(f).name for f in files],
           "shard_names": names, "c": {}}

    # ---- the dev suite and the constants it chose --------------------------
    dev_stats = {}
    for key, fn in (("dev_s_star_max", np.max), ("dev_s_star_median",
                    np.median), ("dev_s_star_min", np.min)):
        dev_stats[key] = spread([
            fn([r["s_star"] for r in d["onesided"]["dev_shards"].values()
                if r["s_star"] is not None]) for d in seeds])
    argmax = []
    for d in seeds:
        rs = {k: r["s_star"] for k, r in d["onesided"]["dev_shards"].items()
              if r["s_star"] is not None}
        argmax.append(max(rs, key=rs.get))
    dev_stats["dev_argmax_shard"] = argmax
    dev_stats["dev_n_shards"] = spread([len(d["onesided"]["dev_shards"])
                                        for d in seeds])
    dev_stats["dev_n_per_shard"] = seeds[0]["onesided"]["dev_n_per_shard"]
    dev_stats["dev_seed_base"] = seeds[0]["onesided"]["dev_seed_base"]
    out["dev"] = dev_stats

    # ---- oracle reference: the smallest constant that would just suffice ---
    max_eval = [max(d["shards"][n]["s_star"] for n in names) for d in seeds]
    out["oracle_max_eval_s_star"] = spread(max_eval)
    out["overpayment_dev_p100_over_oracle"] = spread([
        d["onesided"]["c_ladder"]["c_dev_p100"] / m
        for d, m in zip(seeds, max_eval)])

    # ---- the ladder -------------------------------------------------------
    for cname in ladder:
        r1, r2, r1ci, wmed, wmax, indist_r2, indist_cov = [], [], [], [], [], [], []
        for d in seeds:
            cells = [d["shards"][n]["onesided"][cname] for n in names]
            cov = np.array([c["coverage"] for c in cells])
            hi = np.array([c["ci95"][1] for c in cells])
            w = np.array([c["width_mult_median"] for c in cells])
            if not np.isfinite(w).all():
                raise SystemExit(f"{cname}: a width multiplier is not finite; "
                                 "an infinite interval is abstention, not "
                                 "coverage, and must not be counted")
            r1.append(int((cov >= TARGET).sum()))
            r1ci.append(int((hi >= TARGET).sum()))
            r2.append(int(((cov >= BAND[0]) & (cov <= BAND[1])).sum()))
            wmed.append(float(np.median(w)))
            wmax.append(float(w.max()))
            icov = {t: d["in_dist"][t]["onesided"][cname]["coverage"]
                    for t in d["in_dist"]}
            indist_cov.append(icov)
            indist_r2.append(int(sum(1 for v in icov.values()
                                     if BAND[0] <= v <= BAND[1])))
        out["c"][cname] = {
            "c": spread([d["onesided"]["c_ladder"][cname] for d in seeds]),
            "dev_chosen": cname.startswith("c_dev_"),
            "R1_one_sided_ge_0p90": spread(r1),
            "R1_ci_upper_ge_0p90": spread(r1ci),
            "R2_two_sided_in_band": spread(r2),
            "width_mult_median_over_shards": spread(wmed),
            "width_mult_max_over_shards": spread(wmax),
            "in_dist_R2_in_band_of_3": spread(indist_r2),
            "in_dist_coverage": {t: spread([ic[t] for ic in indist_cov])
                                 for t in indist_cov[0]},
            "all_seeds_R1_full": bool(min(r1) == len(names)),
        }

    # ---- internal consistency: R1 at c must equal #{s* <= c} --------------
    bad = []
    for cname in ladder:
        for d in seeds:
            c = d["onesided"]["c_ladder"][cname]
            by_sstar = sum(1 for n in names
                           if d["shards"][n]["s_star"] is not None
                           and d["shards"][n]["s_star"] <= c)
            cells = [d["shards"][n]["onesided"][cname] for n in names]
            by_cov = sum(1 for x in cells if x["coverage"] >= TARGET)
            if abs(by_sstar - by_cov) > 1:
                bad.append({"c": cname, "seed": d.get("seed"),
                            "by_sstar": by_sstar, "by_coverage": by_cov})
    out["consistency_sstar_vs_coverage"] = {
        "note": ("coverage is monotone increasing in the scale, so "
                 "'coverage >= 0.90 at c' must agree with 's* <= c' to within "
                 "one shard (bisection tolerance at the band edge). A "
                 "disagreement larger than that means one of the two "
                 "measurements is wrong."),
        "disagreements": bad, "ok": not bad}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/h31_onesided.json")
    args = ap.parse_args()

    res = {"hypothesis": "H31",
           "band": list(BAND), "target": TARGET,
           "readings": {
               "R2": "coverage in [0.88, 0.92]; two-sided; THE CLAUSE",
               "R1": ("coverage >= 0.90 with a finite interval; the "
                      "conventional conformal reading; admissible only with "
                      "the width multiplier beside it")},
           "arms": {}}
    for arm, pat in ARMS.items():
        files = sorted(glob.glob(pat))
        if not files:
            res["arms"][arm] = {"status": "[not measured]", "glob": pat}
            continue
        res["arms"][arm] = arm_summary(files)

    # ---- paired comparison between the two paths at each shared c ---------
    a, b = res["arms"].get("eq"), res["arms"].get("base")
    res["eq_vs_base"] = {}
    if isinstance(a, dict) and "c" in a and isinstance(b, dict) and "c" in b:
        for cname in sorted(set(a["c"]) & set(b["c"])):
            if cname.startswith("c_dev_"):
                continue        # c differs between the arms; not paired
            d = [x - y for x, y in zip(
                a["c"][cname]["R1_one_sided_ge_0p90"]["per_seed"],
                b["c"][cname]["R1_one_sided_ge_0p90"]["per_seed"])]
            res["eq_vs_base"][cname] = {
                "mean_R1_diff_eq_minus_base": float(np.mean(d)),
                "per_seed_diff": d,
                "sign_flip_p_two_sided": sign_flip_p(d) if len(d) <= 20 else None}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    for arm in res["arms"]:
        A = res["arms"][arm]
        if "c" not in A:
            print(f"{arm}: {A['status']}")
            continue
        print(f"--- {arm}: {A['n_seeds']} seeds, {A['n_shards']} shards, "
              f"dev max s* {A['dev']['dev_s_star_max']['median']:.4g} "
              f"(argmax {A['dev']['dev_argmax_shard'][0]}), "
              f"oracle max eval s* {A['oracle_max_eval_s_star']['median']:.4g}, "
              f"overpayment x{A['overpayment_dev_p100_over_oracle']['median']:.3g}")
        print(f"  {'c name':14s} {'c':>9s} {'R1/24':>10s} {'R2/24':>8s} "
              f"{'w_med':>9s} {'w_max':>9s} {'in-dist R2/3':>13s}")
        for cn, r in A["c"].items():
            print(f"  {cn:14s} {r['c']['median']:9.4g} "
                  f"{r['R1_one_sided_ge_0p90']['min']:3d}/"
                  f"{int(r['R1_one_sided_ge_0p90']['median']):<2d}/"
                  f"{r['R1_one_sided_ge_0p90']['max']:<3d} "
                  f"{int(r['R2_two_sided_in_band']['median']):8d} "
                  f"{r['width_mult_median_over_shards']['median']:9.4g} "
                  f"{r['width_mult_max_over_shards']['median']:9.4g} "
                  f"{int(r['in_dist_R2_in_band_of_3']['median']):13d}")


if __name__ == "__main__":
    main()
