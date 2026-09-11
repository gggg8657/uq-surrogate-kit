"""Aggregate H27: can any deployment-observable scalar reproduce s*?

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h27.py

H27 computed, per shard, the sigma scale `s*` that makes coverage exactly 0.90
with the calibration quantile frozen. `s*` is an ORACLE target computed from
the truth. This asks how well each deployment-observable per-shard scalar
predicts it, and -- the number that matters -- how many shards land in band if
the best monotone link from that observable is fitted LEAVE-ONE-SHARD-OUT.

That last count is the ceiling of the whole family

    per-sample scalar modulation of sigma, from a deployment-observable
    quantity, with ANY link function

which contains H25 (link = identity on relresid), H26 (link = a power) and
codex's conditioning-proxy routes as special cases. Emits runs/h27_target.json.
"""
from __future__ import annotations

import itertools
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAND = (0.88, 0.92)


def spearman(x, y):
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return o.astype(float)
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def sign_flip_p(d):
    n, obs = len(d), sum(d)
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if sum(a * b for a, b in zip(s, d)) >= obs)
    return hits / 2 ** n


def main():
    runs = [json.loads(f.read_text())
            for f in sorted((ROOT / "runs").glob("starget_u*_het.json"))]
    if not runs:
        raise SystemExit("no H27 runs")
    for r in runs:
        if r.get("clause_eligible") is not False:
            raise SystemExit(f"{r['ckpt']}: s* runs must be clause_eligible "
                             "false; refusing to aggregate")

    names = sorted(runs[0]["shards"])
    obs_keys = sorted(runs[0]["shards"][names[0]]["observables"])
    out = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
           "n_shards": len(names), "band": list(BAND),
           "clause_eligible": False,
           "note": "s* is an oracle target computed from the truth. The "
                   "`loso` counts below are CEILINGS on the scalar-modulation "
                   "family, not method results.",
           "shards": {}}

    # ---- s* itself ----------------------------------------------------------
    S = {n: [r["shards"][n]["s_star"] for r in runs] for n in names}
    missing = {n: sum(1 for v in S[n] if v is None) for n in names}
    out["n_sstar_missing"] = {k: v for k, v in missing.items() if v}
    if out["n_sstar_missing"]:
        raise SystemExit(f"s* missing on {out['n_sstar_missing']}")
    med = {n: st.median(S[n]) for n in names}
    out["s_star_median"] = med
    out["s_star_span"] = [min(med.values()), max(med.values())]
    out["s_star_span_orders"] = float(np.log10(max(med.values())
                                               / min(med.values())))
    out["s_star_below_1"] = sorted(n for n in names if med[n] < 1.0)
    out["in_dist_s_star_median"] = {
        t: st.median([r["in_dist"][t]["s_star"] for r in runs])
        for t in runs[0]["in_dist"]}
    for n in names:
        out["shards"][n] = {
            "parent": runs[0]["shards"][n]["parent"],
            "base_coverage_median": st.median(
                [r["shards"][n]["base"]["coverage"] for r in runs]),
            "s_star_median": med[n],
            "s_star_range": [min(S[n]), max(S[n])],
        }

    # ---- how well does each observable rank s*? -----------------------------
    y = np.log([med[n] for n in names])
    corr = {}
    for k in obs_keys:
        x = [st.median([r["shards"][n]["observables"][k] for r in runs])
             for n in names]
        per_seed = [spearman(
            [r["shards"][n]["observables"][k] for n in names],
            [np.log(r["shards"][n]["s_star"]) for n in names]) for r in runs]
        corr[k] = {"spearman_on_medians": spearman(x, y),
                   "spearman_per_seed": per_seed,
                   "spearman_median_over_seeds": st.median(per_seed),
                   "observable_median_per_shard": dict(zip(names, x))}
    out["observables"] = corr
    best = max(obs_keys, key=lambda k: abs(corr[k]["spearman_on_medians"]))
    out["best_observable_by_rank_corr"] = best
    out["best_rank_corr"] = corr[best]["spearman_on_medians"]

    # ---- the ceiling: best monotone link, fitted leave-one-shard-out -------
    # The link is isotonic in the observable, which is the most expressive
    # monotone function there is, so this is an upper bound on every
    # parametric link (identity, power, log-linear) at once.
    def isotonic(x, w):
        """Pool-adjacent-violators fit of w on x, both 1-D, x sorted."""
        y_, wt = list(map(float, w)), [1.0] * len(w)
        i = 0
        while i < len(y_) - 1:
            if y_[i] <= y_[i + 1] + 1e-15:
                i += 1
                continue
            tw = wt[i] + wt[i + 1]
            av = (y_[i] * wt[i] + y_[i + 1] * wt[i + 1]) / tw
            y_[i:i + 2] = [av]
            wt[i:i + 2] = [tw]
            i = max(i - 1, 0)
        o, k = [], 0
        for v, c in zip(y_, wt):
            o += [v] * int(round(c))
            k += 1
        return o

    def loso_count(r, key, per_family):
        """Shards in band when the link is fitted on the OTHER shards."""
        hit = 0
        for n in names:
            fam = r["shards"][n]["parent"]
            peers = [m for m in names if m != n and
                     (not per_family or r["shards"][m]["parent"] == fam)]
            if len(peers) < 2:
                continue
            xs = np.array([r["shards"][m]["observables"][key] for m in peers])
            ys = np.log([r["shards"][m]["s_star"] for m in peers])
            o = np.argsort(xs)
            fit = isotonic(xs[o], ys[o])
            xt = r["shards"][n]["observables"][key]
            # isotonic prediction = value at the nearest fitted knot below
            j = int(np.searchsorted(xs[o], xt, side="right")) - 1
            j = min(max(j, 0), len(fit) - 1)
            s_pred = float(np.exp(fit[j]))
            # coverage at s_pred, interpolated from the bisection's own
            # monotone relation: we only KNOW coverage at s*. So score the
            # prediction by whether it is inside the multiplicative tolerance
            # that keeps coverage in band, measured per shard below.
            lo, hi = r["shards"][n]["s_band"]
            if lo <= s_pred <= hi:
                hit += 1
        return hit

    # s_band: the multiplicative window on s that keeps coverage in [0.88,0.92]
    # is not in the H27 JSON, so the ceiling needs it. Recorded as missing and
    # computed by scripts/eval_scale_band.py rather than approximated here.
    have_band = all("s_band" in r["shards"][n] for r in runs for n in names)
    out["s_band_present"] = have_band
    if have_band:
        for key, tag in ((best, "best"), ("relresid_ratio_med", "relresid")):
            out[f"loso_{tag}"] = {
                "observable": key,
                "global_link_n_band_per_seed":
                    [loso_count(r, key, False) for r in runs],
                "per_family_link_n_band_per_seed":
                    [loso_count(r, key, True) for r in runs]}
            for sub in ("global_link", "per_family_link"):
                v = out[f"loso_{tag}"][f"{sub}_n_band_per_seed"]
                out[f"loso_{tag}"][f"{sub}_median"] = st.median(v)
    else:
        out["loso_note"] = ("the in-band tolerance window on s is not in the "
                            "H27 JSON, so the link ceiling is [not measured] "
                            "here; scripts/eval_scale_band.py adds it")

    p = ROOT / "runs" / "h27_target.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"H27 aggregate, {out['n_seeds']} seeds, {out['n_shards']} shards\n")
    print(f"in-distribution s* (must be ~1.00): "
          + "  ".join(f"{t} {v:.4f}"
                      for t, v in out["in_dist_s_star_median"].items()))
    lo, hi = out["s_star_span"]
    print(f"s* over the 24 shards: {lo:.4g} .. {hi:.4g} "
          f"= {out['s_star_span_orders']:.2f} orders of magnitude")
    print(f"shards needing s* < 1 (i.e. a NARROWER interval): "
          f"{[n.split('/')[1] for n in out['s_star_below_1']]}\n")
    print("rank correlation of each deployment-observable scalar with log s*:")
    for k in sorted(obs_keys, key=lambda k: -abs(corr[k]["spearman_on_medians"])):
        c = corr[k]
        print(f"  {k:24s} rho(medians) {c['spearman_on_medians']:+.3f}   "
              f"median over seeds {c['spearman_median_over_seeds']:+.3f}")
    print(f"\nbest observable: {best} at rho = {out['best_rank_corr']:+.3f}")
    if have_band:
        for tag in ("best", "relresid"):
            d = out[f"loso_{tag}"]
            print(f"\nLOSO isotonic link on {d['observable']} (A CEILING):")
            print(f"  one global link : median "
                  f"{d['global_link_median']}/{out['n_shards']}  "
                  f"{d['global_link_n_band_per_seed']}")
            print(f"  per-family link : median "
                  f"{d['per_family_link_median']}/{out['n_shards']}  "
                  f"{d['per_family_link_n_band_per_seed']}")
    else:
        print(f"\nlink ceiling: [not measured] -- {out['loso_note']}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
