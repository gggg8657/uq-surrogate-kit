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

    # --- selection honesty ------------------------------------------------
    # The argmax clip over all 32 shards is selected and evaluated on the same
    # shards, which is the same class of error as tuning a threshold on the
    # test set. Two honest readings sit next to it: a held-out split, and a
    # rule that can be stated in advance and never looks at coverage at all.
    import itertools as _it
    import random as _rnd

    def in_band(seed, shard, c):
        v = per_seed[seed][shard]["by_clip"][c]["coverage"]
        return BAND[0] <= v <= BAND[1]

    tuned = max(clips, key=lambda c: summary[c]["in_band_mean"])
    rng, held = _rnd.Random(0), []
    for _ in range(200):
        sh = list(eligible)
        rng.shuffle(sh)
        a, b = sh[:len(sh) // 2], sh[len(sh) // 2:]
        pick = max(clips, key=lambda c: st.mean(
            [sum(in_band(s, n, c) for n in a) for s in per_seed]))
        held.append((pick, st.mean(
            [sum(in_band(s, n, pick) for n in b) for s in per_seed]) / len(b)))
    safe = max([c for c in clips if float(c) <= bound], key=float)

    def sign_flip(v):
        """Exact two-sided sign-flip p over the per-seed values."""
        obs = sum(v)
        hits = sum(1 for sg in _it.product([1, -1], repeat=len(v))
                   if abs(sum(a * b for a, b in zip(sg, v))) >= abs(obs))
        return hits / 2 ** len(v)

    d_inband = [x - y for x, y in zip(summary[safe if False else tuned]
                                      ["in_band_per_seed"],
                                      summary["20"]["in_band_per_seed"])] \
        if "20" in summary else None
    gain = [st.mean([abs(per_seed[s][n]["group"] - 0.90)
                     - abs(per_seed[s][n]["by_clip"][tuned]["coverage"] - 0.90)
                     for n in eligible]) for s in per_seed]
    cells = [(per_seed[s][n]["by_clip"][c]["coverage"], c)
             for c in clips for s in per_seed for n in eligible]
    fail_mode = {c: {
        "over_0p92": sum(1 for v, cc in cells if cc == c and v > BAND[1]),
        "in_band": sum(1 for v, cc in cells if cc == c and BAND[0] <= v <= BAND[1]),
        "under_0p88": sum(1 for v, cc in cells if cc == c and v < BAND[0]),
        "median_coverage": st.median([v for v, cc in cells if cc == c]),
        "n_cells": sum(1 for _, cc in cells if cc == c)} for c in clips}

    selection = {
        "tuned_clip": tuned,
        "tuned_in_band_mean": summary[tuned]["in_band_mean"],
        "tuned_is_optimistic": ("selected and evaluated on the same 32 shards; "
                                "quote the held-out or pre-registerable row "
                                "instead"),
        "held_out": {
            "n_splits": len(held), "split": "random half/half over shards",
            "clips_chosen": {c: sum(1 for p, _ in held if p == c)
                             for c in clips},
            "in_band_fraction_mean": st.mean([r for _, r in held]),
            "in_band_equivalent_of_n": st.mean([r for _, r in held])
            * len(eligible)},
        "pre_registerable": {
            "rule": "largest swept clip <= the no-infinity bound",
            "clip": safe, "bound": bound,
            "in_band_mean": summary[safe]["in_band_mean"],
            "why": ("this rule never looks at a coverage number, so it carries "
                    "no selection effect at all")},
        "vs_shipped_clip_20": {
            "in_band_per_seed_diff": d_inband,
            "exact_sign_flip_p": sign_flip(d_inband) if d_inband else None},
        "vs_group_calibrator": {
            "metric": "mean |coverage - 0.90| improvement, per seed",
            "per_seed": gain, "mean": st.mean(gain),
            "exact_sign_flip_p": sign_flip(gain)},
        "failure_mode_by_clip": fail_mode,
    }

    res = {"score": args.score, "selection": selection, "n_seeds": len(per_seed),
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
    sel = res["selection"]
    print(f"\nselection: tuned clip {sel['tuned_clip']} reads "
          f"{sel['tuned_in_band_mean']:.2f}/{len(eligible)} but is selected on "
          f"the same shards it is scored on.")
    print(f"  held out (200 half/half splits): "
          f"{sel['held_out']['in_band_equivalent_of_n']:.2f}/{len(eligible)}")
    print(f"  pre-registerable (clip <= bound -> {sel['pre_registerable']['clip']}): "
          f"{sel['pre_registerable']['in_band_mean']:.2f}/{len(eligible)}")
    print(f"  vs shipped clip 20: exact sign-flip p = "
          f"{sel['vs_shipped_clip_20']['exact_sign_flip_p']:.4f}")
    print(f"  vs the `group` calibrator: mean |cov-0.90| improvement "
          f"{sel['vs_group_calibrator']['mean']:.4f}, exact p = "
          f"{sel['vs_group_calibrator']['exact_sign_flip_p']:.4f}")
    print("\n failure mode per clip (cells = 8 seeds x %d shards):" % len(eligible))
    for c in clips:
        f = sel["failure_mode_by_clip"][c]
        print(f"  clip {c:>3}: over {f['over_0p92']:4d}  in band {f['in_band']:4d}"
              f"  under {f['under_0p88']:4d}  median coverage "
              f"{f['median_coverage']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
