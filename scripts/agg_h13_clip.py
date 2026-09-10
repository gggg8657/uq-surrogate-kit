"""Aggregate the H13 clip sweep across seeds into one JSON the report can read.

The question H13 asks is whether the weighted-conformal abstention that fails
clause 1 under shift is a property of the shifts or of the `clip` constant. So
the aggregate has to carry three things per clip, never coverage alone:

  * **abstention rate** -- an infinite quantile covers everything and certifies
    nothing, so a 100% coverage cell next to a high abstention rate is not a
    repair. Coverage is reported both raw and *conditional on a finite
    quantile*, because those answer different questions.
  * **width** -- `q_over_unweighted`, the median finite weighted quantile
    divided by the unweighted one. Covering by being wide is the other way to
    fake this clause.
  * **spread across seeds** -- 8 seeds, and the in-band count is reported per
    seed as well as pooled, because a clause verdict that holds on the mean and
    on 4/8 seeds is not a verdict.

Operator-shift shards (`assumption_violated`) are excluded from every count:
they change p(y|x), so no reweighting of x is even the right tool. They are
still listed, separately, as a diagnostic.

    python scripts/agg_h13_clip.py --out runs/h13_clip.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAND = (0.88, 0.92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="h13_clip_u*_het.json")
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--out", default="runs/h13_clip.json")
    args = ap.parse_args()

    files = sorted((ROOT / "runs").glob(args.glob))
    if not files:
        raise SystemExit(f"no runs matching {args.glob}")

    per_seed = {}
    clips, bound = None, None
    for f in files:
        d = json.loads(f.read_text())
        ood = d["scores"][args.score]["ood"]
        seed = d.get("seed", f.stem)
        rows = {}
        for name, rec in ood.items():
            by = rec.get("weighted_by_clip")
            if not by:
                continue
            if clips is None:
                clips = sorted(by, key=float)
            rows[name] = {
                "violated": bool(rec.get("assumption_violated")),
                "kind": rec.get("kind"), "rel_l2": rec.get("rel_l2_mean"),
                "group": rec["group"]["coverage"],
                "by_clip": {c: {
                    "coverage": by[c]["coverage"]["coverage"],
                    "abstention": by[c]["abstention_rate"],
                    "ess": by[c]["ess"],
                    "q_over_unweighted": by[c]["q_over_unweighted"],
                    "below_bound": by[c]["clip_below_bound"],
                } for c in by},
            }
            bound = by[clips[0]]["no_inf_bound"]
        per_seed[str(seed)] = rows

    eligible = [n for n, r in next(iter(per_seed.values())).items()
                if not r["violated"]]
    violated = [n for n, r in next(iter(per_seed.values())).items()
                if r["violated"]]

    summary = {}
    for c in clips:
        in_band_per_seed, abst, widths, cond_band = [], [], [], []
        for seed, rows in per_seed.items():
            cov = [rows[n]["by_clip"][c]["coverage"] for n in eligible]
            ab = [rows[n]["by_clip"][c]["abstention"] for n in eligible]
            w = [rows[n]["by_clip"][c]["q_over_unweighted"] for n in eligible]
            in_band_per_seed.append(
                sum(1 for x in cov if BAND[0] <= x <= BAND[1]))
            abst.append(sum(ab) / len(ab))
            fin = [x for x in w if x != float("inf")]
            widths.append(st.median(fin) if fin else float("inf"))
            # Coverage on shards that did NOT abstain at all: the only cells
            # where the number is a coverage rather than an abstention.
            live = [x for x, a in zip(cov, ab) if a == 0.0]
            cond_band.append(
                (sum(1 for x in live if BAND[0] <= x <= BAND[1]), len(live)))
        summary[c] = {
            "n_eligible_shards": len(eligible),
            "in_band_per_seed": in_band_per_seed,
            "in_band_mean": st.mean(in_band_per_seed),
            "in_band_min": min(in_band_per_seed),
            "in_band_max": max(in_band_per_seed),
            "mean_abstention_rate": st.mean(abst),
            "median_width_vs_unweighted": st.median(
                [w for w in widths if w != float("inf")]) if any(
                w != float("inf") for w in widths) else None,
            "no_abstention_shards_in_band": cond_band,
            "below_no_inf_bound": next(
                iter(per_seed.values()))[eligible[0]]["by_clip"][c]["below_bound"],
        }

    res = {"score": args.score, "n_seeds": len(per_seed),
           "seeds": sorted(per_seed), "clips": clips,
           "no_inf_bound": bound, "band": list(BAND),
           "n_eligible_shards": len(eligible),
           "eligible_shards": eligible,
           "excluded_operator_shift_shards": violated,
           "why_excluded": ("operator shifts change p(y|x); weighted "
                            "conformal's covariate-shift assumption does not "
                            "hold there and no clip repairs that"),
           "summary": summary, "per_seed": per_seed}
    Path(args.out).write_text(json.dumps(res, indent=1))

    print(f"{len(per_seed)} seeds, {len(eligible)} eligible shards, "
          f"no-infinity bound clip <= {bound:.2f}\n")
    print(f"{'clip':>5} {'<=bound':>8} {'in band /'+str(len(eligible)):>14} "
          f"{'abstain':>9} {'width':>8}  per-seed")
    for c in clips:
        v = summary[c]
        w = v["median_width_vs_unweighted"]
        print(f"{c:>5} {str(v['below_no_inf_bound']):>8} "
              f"{v['in_band_mean']:6.2f} [{v['in_band_min']}, {v['in_band_max']}] "
              f"{v['mean_abstention_rate']*100:8.1f}% "
              f"{(f'{w:.2f}x' if w else 'inf'):>8}  {v['in_band_per_seed']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
