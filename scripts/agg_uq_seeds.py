"""Aggregate `runs/conf_u*_*.json` into one 8-seed verdict for clause 1.

Every number here is read out of a per-seed run JSON; nothing is recomputed
from a model and nothing is typed. The output is what `report.py` reads.

The comparison that matters is not "did `het` reach 90%". Split conformal
rescales any sigma by a single calibrated quantile, so a **constant** sigma also
reaches ~90% marginal coverage while producing an interval of the same width
everywhere. `const` is carried through the whole pipeline for exactly that
reason, and the columns that separate a useful interval from a rescaled
constant are `sharpness_rel` (narrower is better at equal coverage) and the
spread-error correlation.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

BAND = (0.88, 0.92)


def spread(xs):
    xs = sorted(xs)
    return {"mean": st.fmean(xs), "min": xs[0], "max": xs[-1],
            "median": st.median(xs), "n": len(xs),
            "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
            "range": xs[-1] - xs[0], "values": xs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/uq_seeds.json")
    ap.add_argument("--score", default="field_max")
    args = ap.parse_args()
    root = Path(args.runs)

    by_src: dict[str, list] = {}
    for p in sorted(root.glob("conf_u*_*.json")):
        seed_src = p.stem[len("conf_u"):]
        seed, src = seed_src.split("_", 1)
        r = json.loads(p.read_text())
        h = r["scores"][args.score]
        rec = {"seed": int(seed), "file": p.name,
               "n_members": r["n_members"],
               "forward_passes_per_interval":
                   r.get("forward_passes_per_interval"),
               "sigma_source": r.get("sigma_source"),
               "coverage_in_dist": h["in_dist"]["pooled_group"]["coverage"],
               "coverage_ci95": h["in_dist"]["pooled_group"]["ci95"],
               "n": h["in_dist"]["pooled_group"]["n"]}
        sh = h.get("sharpness") or {}
        if sh:
            rec["sharpness_rel"] = st.fmean(v["sharpness_rel"]
                                            for v in sh.values())
            rec["sharpness_per_family"] = {k: v["sharpness_rel"]
                                           for k, v in sh.items()}
        # coverage under shift, only on shards where the method's assumption
        # holds -- the operator-shift shards are reported by eval_conformal.py
        # but are not evidence for or against a covariate-shift method
        oods = [v for v in h["ood"].values()
                if "weighted" in v and not v.get("assumption_violated")]
        for kind in ("split", "weighted"):
            n_band = sum(1 for v in oods
                         if BAND[0] <= v[kind]["coverage"] <= BAND[1])
            n_over = sum(1 for v in oods if v[kind]["coverage"] > BAND[1])
            rec[f"ood_{kind}_in_band"] = n_band
            rec[f"ood_{kind}_over"] = n_over
        rec["ood_n_shards"] = len(oods)
        by_src.setdefault(src, []).append(rec)

    out = {"score": args.score, "band": list(BAND), "arms": {},
           "note": ("`const` is a control, not a candidate: split conformal "
                    "rescales any sigma to ~90% marginal coverage, so `const` "
                    "landing in the band is the null result that says how much "
                    "of a coverage number is the head and how much is the "
                    "rescaling. Compare `sharpness_rel` between arms, not "
                    "coverage.")}
    for src, recs in sorted(by_src.items()):
        cov = [r["coverage_in_dist"] for r in recs]
        arm = {"n_seeds": len(recs), "seeds": sorted(r["seed"] for r in recs),
               "coverage_in_dist": spread(cov),
               "n_seeds_in_band": sum(1 for c in cov
                                      if BAND[0] <= c <= BAND[1]),
               "forward_passes_per_interval":
                   sorted({r["forward_passes_per_interval"] for r in recs}),
               "per_seed": recs}
        if all("sharpness_rel" in r for r in recs):
            arm["sharpness_rel"] = spread([r["sharpness_rel"] for r in recs])
        for kind in ("split", "weighted"):
            arm[f"ood_{kind}_in_band"] = spread(
                [r[f"ood_{kind}_in_band"] for r in recs])
            arm[f"ood_{kind}_over"] = spread(
                [r[f"ood_{kind}_over"] for r in recs])
        arm["ood_n_shards"] = sorted({r["ood_n_shards"] for r in recs})
        out["arms"][src] = arm

    Path(args.out).write_text(json.dumps(out, indent=2))
    for src, a in out["arms"].items():
        c = a["coverage_in_dist"]
        sh = a.get("sharpness_rel")
        print(f"{src:6s} n={a['n_seeds']}  coverage {c['mean']:.4f} "
              f"[{c['min']:.4f}, {c['max']:.4f}] range {c['range']:.4f}  "
              f"{a['n_seeds_in_band']}/{a['n_seeds']} seeds in band"
              + (f"  sharpness_rel {sh['mean']:.4f}" if sh else ""))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
