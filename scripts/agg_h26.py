"""Aggregate H26: the residual-scale family's ceiling as a function of gamma.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h26.py

Emits runs/h26_gamma.json. Three readings, each carrying the protocol that
selected its gamma, because they are NOT interchangeable:

  global    one gamma for all families, selected on the shards it scores
            -> a CEILING on the one-knob route
  family    one gamma per family, selected on the shards it scores
            -> a CEILING, strictly above `global`
  loso      one gamma per family, selected on the OTHER shards of that family
            -> honest, and priced: needs labelled shifted shards per family

`gamma=1` is the only clause-eligible arm: it is selected by nothing.

Refuses any per-seed file whose in-run gates did not pass, because gamma=0 must
reproduce the unmodulated base arm exactly and gamma=1 must reproduce H25
exactly, and a sweep that fails either is not interpretable.
"""
from __future__ import annotations

import itertools
import json
import statistics as st
from pathlib import Path

BAND = (0.88, 0.92)
ROOT = Path(__file__).resolve().parents[1]


def in_band(c):
    return c is not None and BAND[0] <= c <= BAND[1]


def sign_flip_p(deltas):
    """Exact one-sided paired sign-flip test on the sum of `deltas`."""
    n = len(deltas)
    obs = sum(deltas)
    hits = sum(1 for sg in itertools.product([1, -1], repeat=n)
               if sum(s * d for s, d in zip(sg, deltas)) >= obs)
    return hits / 2 ** n


def main():
    files = sorted((ROOT / "runs").glob("rgam_u*_het.json"))
    runs, refused = [], []
    for f in files:
        d = json.loads(f.read_text())
        if not d.get("gates", {}).get("ok"):
            refused.append(f.name)
            continue
        runs.append(d)
    if refused:
        raise SystemExit(f"refusing files whose gates failed: {refused}")
    if not runs:
        raise SystemExit("no H26 runs")

    gammas = runs[0]["gammas"]
    keys = [f"g{g:g}" for g in gammas]
    shard_names = sorted(runs[0]["shards"])
    fams = sorted({runs[0]["shards"][n]["parent"] for n in shard_names})

    out = {
        "n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
        "gammas": gammas, "band": list(BAND),
        "n_shards": len(shard_names), "families": fams,
        "gates_all_passed": True,
        "gate_detail": {r["seed"]: r["gates"] for r in runs},
        "protocol": {
            "gamma1": "selected by nothing -- the shipped arm, clause-eligible",
            "global": "one gamma, all families, selected on the shards it "
                      "scores -- A CEILING, not a clause claim",
            "family": "one gamma per family, selected on the shards it scores "
                      "-- A CEILING, not a clause claim",
            "loso": "one gamma per family, selected on the OTHER shards of "
                    "that family -- honest, priced in labelled shifted shards",
        },
    }

    # ---- in-distribution coverage vs gamma: the identifiability claim -------
    idf = {}
    for t in runs[0]["in_dist"]:
        idf[t] = {k: [r["in_dist"][t][k]["coverage"] for r in runs]
                  for k in keys}
    out["in_dist_vs_gamma"] = {
        t: {"median_per_gamma": {k: st.median(v) for k, v in d.items()},
            "range_over_gammas_of_median":
                [min(st.median(v) for v in d.values()),
                 max(st.median(v) for v in d.values())],
            "all_gammas_in_band_on_all_seeds":
                all(in_band(c) for v in d.values() for c in v)}
        for t, d in idf.items()}
    spans = [out["in_dist_vs_gamma"][t]["range_over_gammas_of_median"]
             for t in idf]
    out["in_dist_max_span_over_gamma"] = max(hi - lo for lo, hi in spans)

    # ---- reading: one global gamma -----------------------------------------
    nb = {k: [sum(1 for n in shard_names if in_band(r["shards"][n][k]["coverage"]))
              for r in runs] for k in keys + ["base"]}
    out["global"] = {
        "n_band_per_gamma": {k: nb[k] for k in keys},
        "median_n_band_per_gamma": {k: st.median(nb[k]) for k in keys},
        "best_gamma_by_median": max(keys, key=lambda k: st.median(nb[k])),
        "best_median_n_band": max(st.median(nb[k]) for k in keys),
        "best_n_band_any_gamma_any_seed": max(max(nb[k]) for k in keys),
    }

    # ---- reading: per-family oracle gamma ----------------------------------
    # mean |coverage - 0.90| over that family's shards, per seed, per gamma
    def fam_err(r, fam, k):
        v = [r["shards"][n][k]["coverage"] for n in shard_names
             if r["shards"][n]["parent"] == fam]
        return st.fmean(abs(c - 0.90) for c in v)

    per_fam = {}
    for fam in fams:
        rows = {k: [fam_err(r, fam, k) for r in runs] for k in keys}
        best = min(keys, key=lambda k: st.median(rows[k]))
        per_fam[fam] = {
            "n_shards": sum(1 for n in shard_names
                            if runs[0]["shards"][n]["parent"] == fam),
            "median_abs_cov_err_per_gamma": {k: st.median(rows[k])
                                             for k in keys},
            "oracle_gamma": float(best[1:]),
            "oracle_median_abs_cov_err": st.median(rows[best]),
            "gamma1_median_abs_cov_err": st.median(rows["g1"]),
            "oracle_gamma_per_seed": [
                float(min(keys, key=lambda k: fam_err(r, fam, k))[1:])
                for r in runs],
        }
    out["family"] = {"per_family": per_fam}

    # n_band under the per-family oracle, per seed (gamma chosen per seed too:
    # this is the loosest ceiling and is labelled as such)
    fam_nb = []
    for r in runs:
        g = {fam: min(keys, key=lambda k: fam_err(r, fam, k)) for fam in fams}
        fam_nb.append(sum(1 for n in shard_names
                          if in_band(r["shards"][n][g[r["shards"][n]["parent"]]]
                                     ["coverage"])))
    out["family"]["n_band_per_seed"] = fam_nb
    out["family"]["median_n_band"] = st.median(fam_nb)
    out["family"]["selection_objective"] = (
        "mean |coverage-0.90| over that family's shards -- NOT the clause's "
        "metric, which is the in-band COUNT. Reported because it is the "
        "natural continuous objective, but `family_inband` below is the "
        "ceiling that answers the clause.")

    # The ceiling that actually answers the clause: per-family gamma chosen to
    # maximise the IN-BAND COUNT directly. The mean-|cov-0.90| oracle above can
    # and does score LOWER on this metric, because the two objectives disagree:
    # moving a shard from 0.50 to 0.80 helps the mean and not the count.
    def fam_band(r, fam, k):
        return sum(1 for n in shard_names if r["shards"][n]["parent"] == fam
                   and in_band(r["shards"][n][k]["coverage"]))

    fib, fib_g = [], {f: [] for f in fams}
    for r in runs:
        tot = 0
        for fam in fams:
            g = max(keys, key=lambda k: fam_band(r, fam, k))
            fib_g[fam].append(float(g[1:]))
            tot += fam_band(r, fam, g)
        fib.append(tot)
    out["family_inband"] = {
        "selection_objective": "in-band count on that family's own shards "
                               "-- AN ORACLE CEILING for the clause metric",
        "n_band_per_seed": fib, "median_n_band": st.median(fib),
        "best_n_band_any_seed": max(fib),
        "oracle_gamma_per_seed": fib_g,
    }

    # ---- reading: leave-one-shard-out gamma per family ---------------------
    loso_nb, loso_choices = [], {}
    for r in runs:
        hit = 0
        for n in shard_names:
            fam = r["shards"][n]["parent"]
            peers = [m for m in shard_names
                     if r["shards"][m]["parent"] == fam and m != n]
            g = min(keys, key=lambda k: st.fmean(
                abs(r["shards"][m][k]["coverage"] - 0.90) for m in peers))
            loso_choices.setdefault(n, []).append(float(g[1:]))
            if in_band(r["shards"][n][g]["coverage"]):
                hit += 1
        loso_nb.append(hit)
    out["loso"] = {
        "n_band_per_seed": loso_nb,
        "median_n_band": st.median(loso_nb),
        "gamma_chosen_per_shard_median": {n: st.median(v)
                                          for n, v in loso_choices.items()},
    }

    # ---- gamma=1 vs the base arm, on the continuous metric ------------------
    def all_err(r, k):
        return st.fmean(abs(r["shards"][n][k]["coverage"] - 0.90)
                        for n in shard_names)

    d_g1 = [all_err(r, "base") - all_err(r, "g1") for r in runs]
    out["gamma1"] = {
        "median_abs_cov_err": st.median([all_err(r, "g1") for r in runs]),
        "base_median_abs_cov_err": st.median([all_err(r, "base")
                                              for r in runs]),
        "improvement_per_seed": d_g1,
        "sign_flip_p": sign_flip_p(d_g1),
        "n_band_per_seed": nb["g1"],
        "base_n_band_per_seed": nb["base"],
    }

    # ---- does the LOSO reading beat gamma=1, and is it significant? ---------
    d_loso = [loso_nb[i] - nb["g1"][i] for i in range(len(runs))]
    out["loso"]["vs_gamma1_delta_n_band"] = d_loso
    out["loso"]["vs_gamma1_sign_flip_p"] = (
        sign_flip_p(d_loso) if any(d_loso) else 1.0)

    p = ROOT / "runs" / "h26_gamma.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"H26 aggregate over {out['n_seeds']} seeds, "
          f"{out['n_shards']} covariate shards, gates passed on all\n")
    print("in-distribution coverage vs gamma (the identifiability claim):")
    for t, d in out["in_dist_vs_gamma"].items():
        lo, hi = d["range_over_gammas_of_median"]
        print(f"  {t:10s} median coverage spans {lo:.4f}-{hi:.4f} over "
              f"gamma in [{gammas[0]:g},{gammas[-1]:g}]  "
              f"all-in-band {d['all_gammas_in_band_on_all_seeds']}")
    print(f"  max span over gamma, any family: "
          f"{out['in_dist_max_span_over_gamma']:.4f}\n")
    print("shards in 90+/-2%, one GLOBAL gamma (a ceiling):")
    for k in keys:
        print(f"  {k:8s} median {st.median(nb[k]):4.1f}/{out['n_shards']}  "
              f"per seed {nb[k]}")
    print(f"  best global gamma {out['global']['best_gamma_by_median']} at "
          f"median {out['global']['best_median_n_band']}/{out['n_shards']}\n")
    print("per-family oracle gamma (a ceiling):")
    for fam, d in per_fam.items():
        print(f"  {fam:10s} n={d['n_shards']:2d}  oracle gamma "
              f"{d['oracle_gamma']:.3f}  mean|cov-0.90| "
              f"{d['gamma1_median_abs_cov_err']:.4f} (g=1) -> "
              f"{d['oracle_median_abs_cov_err']:.4f} (oracle)  "
              f"per-seed gammas {d['oracle_gamma_per_seed']}")
    print(f"  n_band under per-family oracle (mean-|cov-0.90| objective, "
          f"NOT the clause metric): median "
          f"{out['family']['median_n_band']}/{out['n_shards']}  "
          f"per seed {fam_nb}\n")
    print(f"per-family oracle gamma maximising the IN-BAND COUNT "
          f"(THE CEILING for the clause): median "
          f"{out['family_inband']['median_n_band']}/{out['n_shards']}  "
          f"per seed {fib}  best any seed "
          f"{out['family_inband']['best_n_band_any_seed']}")
    for fam in fams:
        print(f"    {fam:10s} gamma per seed {fib_g[fam]}")
    print()
    print(f"leave-one-shard-out gamma per family (honest, priced): median "
          f"{out['loso']['median_n_band']}/{out['n_shards']}  "
          f"per seed {loso_nb}")
    print(f"  vs gamma=1: delta {d_loso}, sign-flip p "
          f"{out['loso']['vs_gamma1_sign_flip_p']:.4f}\n")
    print(f"gamma=1 (clause-eligible): mean|cov-0.90| "
          f"{out['gamma1']['base_median_abs_cov_err']:.4f} -> "
          f"{out['gamma1']['median_abs_cov_err']:.4f}, sign-flip p "
          f"{out['gamma1']['sign_flip_p']:.4f}; in band "
          f"{nb['g1']} vs base {nb['base']}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
