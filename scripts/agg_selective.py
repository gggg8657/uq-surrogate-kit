"""Aggregate the H14 gated-certificate runs across seeds into one JSON.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_selective.py \
        --glob 'runs/sel_u*_het.json' --out runs/selective.json

Everything `scripts/report.py` prints about H14 comes from here, so the
disciplines H14 registered are enforced in one place rather than at each call
site:

* a shard whose `n_accepted` is below `min_accepted` never counts as in-band --
  it is `[not measured]`, because "coverage on the survivors" is exactly the
  abstention-wearing-a-coverage-number mistake H13 caught;
* every coverage carried out of here travels with its abstention rate;
* the ungated marginal reading is aggregated in the same pass over the same
  shards, so the strict and the deliberate reading are always adjacent;
* `oracle_err` rows stay flagged `uses_ground_truth` and are summarised
  separately, as a ceiling, never as a detector.
"""
from __future__ import annotations

import argparse
import glob as _glob
import itertools as _it
import json
import statistics as st
from pathlib import Path

BAND = (0.88, 0.92)
COVARIATE_KINDS = {"input_shift", "graded_rough"}


def sign_flip(v):
    """Exact two-sided sign-flip p over the per-seed differences."""
    v = [float(x) for x in v]
    obs = sum(v)
    hits = sum(1 for sg in _it.product([1, -1], repeat=len(v))
               if abs(sum(a * b for a, b in zip(sg, v))) >= abs(obs))
    return hits / 2 ** len(v)


def in_band(c):
    return c is not None and BAND[0] <= c <= BAND[1]


def _avg_ranks(v):
    """Ranks with ties averaged.

    Breaking ties by array order -- which an `argsort(argsort(.))` does -- can
    manufacture a perfect rank correlation out of a constant series. The ladder
    correlations reported for H14 are over six rungs, where two rungs sharing
    an abstention rate is entirely possible, so the tie handling is load
    bearing rather than cosmetic. `tests/test_selective.py` caught this.
    """
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(x, y):
    """Rank correlation, ties averaged; None when either side has no variance."""
    n = len(x)
    if n < 3:
        return None
    a, b = _avg_ranks(list(x)), _avg_ranks(list(y))
    ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return None if den == 0 else num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/sel_u*_het.json")
    ap.add_argument("--out", default="runs/selective.json")
    ap.add_argument("--price-glob", default="runs/selp_u*_het.json",
                    help="H14b abstention price-curve runs, aggregated into "
                         "the same JSON so the operating point and its price "
                         "can never be quoted from different files")
    ap.add_argument("--score", default="field_max",
                    help="headline score; every score present is aggregated")
    args = ap.parse_args()

    paths = sorted(_glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no runs matched {args.glob}")
    runs = [json.loads(Path(p).read_text()) for p in paths]
    seeds = [r.get("seed") for r in runs]
    min_acc = runs[0]["min_accepted"]
    beta = runs[0]["beta"]
    for r in runs:
        if r["min_accepted"] != min_acc or r["beta"] != beta:
            raise SystemExit("runs disagree on beta/min_accepted; refusing to "
                             "pool arms measured under different protocols")

    gates = sorted(set().union(*[set(r["gates"]) for r in runs]))
    scores = sorted(set().union(*[set(r["gates"][g]["per_score"])
                                  for r in runs for g in r["gates"]]))
    oracle = set(runs[0].get("oracle_gates", []))

    out = {"n_seeds": len(runs), "seeds": seeds, "runs": paths,
           "beta": beta, "min_accepted": min_acc, "band": list(BAND),
           "alpha": runs[0]["alpha"],
           "sigma_source": runs[0]["sigma_source"],
           "forward_passes_per_interval":
               runs[0]["forward_passes_per_interval"],
           "oracle_gates": sorted(oracle),
           "headline_score": args.score,
           "gate_available": runs[0]["gate_available"],
           "by_gate": {}}

    for gate in gates:
        gsum = {"uses_ground_truth": gate in oracle, "by_score": {}}
        for sname in scores:
            per_run = [r["gates"].get(gate, {}).get("per_score", {}).get(sname)
                       for r in runs]
            if any(v is None or not v.get("available") for v in per_run):
                continue
            shard_names = sorted(set().union(*[set(v["shards"])
                                               for v in per_run]))
            shards = {}
            for n in shard_names:
                cells = [v["shards"][n] for v in per_run if n in v["shards"]]
                if len(cells) != len(per_run):
                    continue  # a shard not scored on every seed is not pooled
                sel = [c["selective"]["coverage"] for c in cells]
                marg = [c["marginal_ungated"]["coverage"] for c in cells]
                meas = [c["measured"] for c in cells]
                sp = [c["spearman_gate_vs_score"] for c in cells
                      if c["spearman_gate_vs_score"] is not None]
                shards[n] = {
                    "kind": cells[0]["kind"], "parent": cells[0]["parent"],
                    "N": cells[0]["N"],
                    "rel_l2_mean": st.median([c["rel_l2_mean"]
                                              for c in cells]),
                    "abstention_median": st.median([c["abstention"]
                                                    for c in cells]),
                    "abstention_range": [min(c["abstention"] for c in cells),
                                         max(c["abstention"] for c in cells)],
                    "n_accepted_median": st.median([c["n_accepted"]
                                                    for c in cells]),
                    "measured_seeds": sum(meas),
                    # a coverage is only reported where it was measurable
                    "selective_median": (
                        st.median([c for c, m in zip(sel, meas)
                                   if m and c is not None])
                        if any(m and c is not None
                               for c, m in zip(sel, meas)) else None),
                    "marginal_ungated_median": st.median(
                        [c for c in marg if c is not None]),
                    # in-band requires BOTH measurable and in band, per seed
                    "in_band_seeds_selective": sum(
                        1 for c, m in zip(sel, meas) if m and in_band(c)),
                    "in_band_seeds_marginal": sum(1 for c in marg
                                                  if in_band(c)),
                    "spearman_gate_vs_score_median": (st.median(sp)
                                                      if sp else None),
                    # how far the shard's median gate score sits past the
                    # refusal threshold: >1 means the gate wants to refuse the
                    # typical sample, <1 means it barely notices the shift
                    "gate_q50_over_tau_median": st.median(
                        [c["gate_q50_over_tau"] for c in cells]),
                    "width_rel_accepted_median": st.median(
                        [c["width_rel_accepted"] for c in cells
                         if c["width_rel_accepted"] is not None]) if any(
                        c["width_rel_accepted"] is not None for c in cells)
                        else None,
                    "width_rel_all_median": st.median([c["width_rel_all"]
                                                       for c in cells]),
                }
            cov = {n: v for n, v in shards.items()
                   if v["kind"] in COVARIATE_KINDS}
            # per-seed tallies over the covariate population, for the exact test
            sel_per_seed, marg_per_seed = [], []
            for i, v in enumerate(per_run):
                sel_per_seed.append(sum(
                    1 for n in cov
                    if v["shards"][n]["measured"]
                    and in_band(v["shards"][n]["selective"]["coverage"])))
                marg_per_seed.append(sum(
                    1 for n in cov
                    if in_band(v["shards"][n]["marginal_ungated"]["coverage"])))
            diffs = [a - b for a, b in zip(sel_per_seed, marg_per_seed)]
            # monotonicity of abstention in shift strength, per ladder, per seed
            ladders = {}
            for fam in ("poisson", "darcy"):
                rungs = sorted([n for n, v in cov.items()
                                if v["kind"] == "graded_rough"
                                and v["parent"] == fam],
                               key=lambda n: shards[n]["rel_l2_mean"])
                if len(rungs) >= 3:
                    rho = [spearman([per_run[i]["shards"][n]["rel_l2_mean"]
                                     for n in rungs],
                                    [per_run[i]["shards"][n]["abstention"]
                                     for n in rungs])
                           for i in range(len(per_run))]
                    ladders[fam] = {
                        "rungs": rungs,
                        "spearman_abstention_vs_rel_l2_per_seed": rho,
                        "spearman_median": st.median([x for x in rho
                                                      if x is not None]),
                    }
            gsum["by_score"][sname] = {
                "n_covariate_shards": len(cov),
                "shards": shards,
                "in_band_selective_per_seed": sel_per_seed,
                "in_band_marginal_per_seed": marg_per_seed,
                "in_band_selective_median": st.median(sel_per_seed),
                "in_band_marginal_median": st.median(marg_per_seed),
                "not_measured_shards_median": st.median(
                    [sum(1 for n in cov if not v["shards"][n]["measured"])
                     for v in per_run]),
                "abstention_median_over_shards": st.median(
                    [v["abstention_median"] for v in cov.values()]),
                "vs_ungated_marginal": {
                    "diff_per_seed": diffs,
                    "exact_sign_flip_p": (sign_flip(diffs)
                                          if any(d != 0 for d in diffs)
                                          else 1.0),
                },
                "spearman_gate_vs_score_median_over_shards": st.median(
                    [v["spearman_gate_vs_score_median"]
                     for v in cov.values()
                     if v["spearman_gate_vs_score_median"] is not None]),
                "ladders": ladders,
                "in_dist": {
                    t: {"selective_median": st.median(
                            [v["in_dist"][t]["selective"]["coverage"]
                             for v in per_run]),
                        "marginal_ungated_median": st.median(
                            [v["in_dist"][t]["marginal_ungated"]["coverage"]
                             for v in per_run]),
                        "abstention_median": st.median(
                            [v["in_dist"][t]["abstention"] for v in per_run])}
                    for t in per_run[0]["in_dist"]},
            }
        out["by_gate"][gate] = gsum

    # ---- H14b: the abstention price curve -----------------------------------
    ppaths = sorted(_glob.glob(args.price_glob))
    if ppaths:
        pruns = [json.loads(Path(p).read_text()) for p in ppaths]
        betas = pruns[0]["betas"]
        for r in pruns:
            if r["betas"] != betas or r["min_accepted"] != min_acc:
                raise SystemExit("price runs disagree on the beta grid or "
                                 "min_accepted; refusing to pool them")
        price = {"n_seeds": len(pruns), "seeds": [r.get("seed") for r in pruns],
                 "runs": ppaths, "betas": betas,
                 "score": pruns[0]["score"],
                 "registered_operating_beta":
                     pruns[0]["registered_operating_beta"],
                 "by_gate": {}}
        for gate in sorted(pruns[0]["by_gate"]):
            names = sorted(set().union(*[set(r["by_gate"][gate]["shards"])
                                         for r in pruns]))
            shards = {}
            for n in names:
                cells = [r["by_gate"][gate]["shards"][n] for r in pruns
                         if n in r["by_gate"][gate]["shards"]]
                if len(cells) != len(pruns):
                    continue
                mb = [c["min_beta_in_band"] for c in cells]
                shards[n] = {
                    "kind": cells[0]["kind"], "parent": cells[0]["parent"],
                    "rel_l2_mean": st.median([c["rel_l2_mean"]
                                              for c in cells]),
                    # a shard counts as reachable only where a MAJORITY of
                    # seeds reach the band; one seed in eight is noise, and
                    # H13 is the reason that is spelled out here
                    "seeds_ever_in_band": sum(1 for x in mb if x is not None),
                    "min_beta_median": (st.median([x for x in mb
                                                   if x is not None])
                                        if any(x is not None for x in mb)
                                        else None),
                    "curve_median": {
                        f"{b:g}": {
                            "abstention": st.median(
                                [c["curve"][f"{b:g}"]["abstention"]
                                 for c in cells if f"{b:g}" in c["curve"]]),
                            "coverage": (st.median(
                                [c["curve"][f"{b:g}"]["coverage"]
                                 for c in cells
                                 if c["curve"].get(f"{b:g}", {}).get(
                                     "coverage") is not None])
                                if any(c["curve"].get(f"{b:g}", {}).get(
                                    "coverage") is not None for c in cells)
                                else None),
                            "n_accepted": st.median(
                                [c["curve"][f"{b:g}"]["n_accepted"]
                                 for c in cells if f"{b:g}" in c["curve"]]),
                            "in_band_seeds": sum(
                                1 for c in cells
                                if c["curve"].get(f"{b:g}", {}).get("in_band")),
                        } for b in betas},
                }
            cov = {n: v for n, v in shards.items()
                   if v["kind"] in COVARIATE_KINDS}
            maj = [n for n, v in cov.items()
                   if v["seeds_ever_in_band"] > len(pruns) / 2]
            price["by_gate"][gate] = {
                "uses_ground_truth": gate in oracle,
                "n_covariate_shards": len(cov),
                "n_reachable_majority_of_seeds": len(maj),
                "reachable_shards": sorted(maj),
                "min_beta_median_over_reachable": (
                    st.median([cov[n]["min_beta_median"] for n in maj])
                    if maj else None),
                "in_band_by_beta": {
                    f"{b:g}": st.median(
                        [sum(1 for n in cov
                             if r["by_gate"][gate]["shards"][n]["curve"]
                             .get(f"{b:g}", {}).get("in_band"))
                         for r in pruns]) for b in betas},
                "shards": cov,
            }
        out["price_curve"] = price

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for gate in gates:
        b = out["by_gate"][gate]["by_score"].get(args.score)
        if not b:
            continue
        tag = " (ORACLE, uses ground truth)" if gate in oracle else ""
        print(f"[{gate}{tag}] selective in band "
              f"{b['in_band_selective_median']}/{b['n_covariate_shards']} "
              f"vs ungated marginal {b['in_band_marginal_median']}; "
              f"median abstention {b['abstention_median_over_shards']:.3f}; "
              f"p={b['vs_ungated_marginal']['exact_sign_flip_p']:.4f}")
    if "price_curve" in out:
        for gate, v in out["price_curve"]["by_gate"].items():
            print(f"[price/{gate}] reachable at SOME beta on a majority of "
                  f"seeds: {v['n_reachable_majority_of_seeds']}"
                  f"/{v['n_covariate_shards']}; in-band by beta "
                  + ", ".join(f"{b}:{n}" for b, n
                              in v["in_band_by_beta"].items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
