"""H28: regress log s* on two features that span both shift axes.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h28.py

H27 measured that no single deployment-observable scalar predicts the required
sigma scale s* well enough: best Spearman 0.472 against a tolerance of +/-2.52%
(the median in-band window on the scale is 1.0503 wide). The mechanism is that
the suite contains TWO shift axes and each candidate feature is blind to one:
`relresid` tracks the dam ladder and is EXACTLY invariant to a rescaling of the
linear channel (so it misses helmholtz_amp2 by 58.8x and poisson_amp2 by 125x),
while `input_amp` sees the amplitude shift and is constant along the ladder.

This fits log s* on BOTH, leave-one-shard-out, and reports the in-band count
against each shard's own measured window. Post-processing only -- it reads the
H27 JSONs and touches no GPU.

s* is an oracle target computed from the truth, so nothing here is a method
result either: this is the CEILING of a two-feature scalar rule. What makes it
worth measuring is that it is the smallest extension of the family H27 bounded
that could plausibly clear the clause, and it costs nothing to check.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["relresid_ratio_med", "input_amp"]


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return float(np.corrcoef(r(x), r(y))[0, 1])


def fit_predict(X, y, xt):
    """Least squares on [1, X] in log space; returns the prediction at xt."""
    A = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.concatenate([[1.0], xt]) @ beta), beta


def main():
    runs = [json.loads(f.read_text())
            for f in sorted((ROOT / "runs").glob("starget_u*_het.json"))]
    if not runs:
        raise SystemExit("no H27 runs")
    names = sorted(runs[0]["shards"])
    out = {"n_seeds": len(runs), "n_shards": len(names), "features": FEATS,
           "clause_eligible": False,
           "note": "ceiling of a two-feature scalar rule. s* is an oracle "
                   "target computed from the truth; these counts bound a "
                   "family, they are not a method result.",
           "arms": {}}

    def feats(r, n):
        return [np.log(max(r["shards"][n]["observables"][f], 1e-30))
                for f in FEATS]

    for tag, keep in (("2feat", FEATS), ("relresid_only", FEATS[:1]),
                      ("input_amp_only", FEATS[1:])):
        idx = [FEATS.index(k) for k in keep]
        nb_g, nb_f, rho = [], [], []
        worst = {}
        for r in runs:
            y_all = {n: np.log(r["shards"][n]["s_star"]) for n in names}
            preds = {}
            hg = hf = 0
            for n in names:
                for per_family, bucket in ((False, "g"), (True, "f")):
                    peers = [m for m in names if m != n and
                             (not per_family or r["shards"][m]["parent"]
                              == r["shards"][n]["parent"])]
                    if len(peers) <= len(idx) + 1:
                        continue
                    X = np.array([[feats(r, m)[i] for i in idx]
                                  for m in peers])
                    yv = np.array([y_all[m] for m in peers])
                    p, _ = fit_predict(X, yv,
                                       np.array([feats(r, n)[i] for i in idx]))
                    s_pred = float(np.exp(p))
                    lo, hi = r["shards"][n]["s_band"]
                    ok = lo <= s_pred <= hi
                    if bucket == "g":
                        hg += ok
                        preds[n] = s_pred
                    else:
                        hf += ok
            nb_g.append(hg)
            nb_f.append(hf)
            rho.append(spearman([preds[n] for n in names],
                                [np.exp(y_all[n]) for n in names]))
            for n in names:
                worst.setdefault(n, []).append(
                    abs(np.log(preds[n] / r["shards"][n]["s_star"])))
        out["arms"][tag] = {
            "features": keep,
            "global_link_n_band_per_seed": nb_g,
            "global_link_median": st.median(nb_g),
            "per_family_link_n_band_per_seed": nb_f,
            "per_family_link_median": st.median(nb_f),
            "loso_pred_spearman_per_seed": rho,
            "loso_pred_spearman_median": st.median(rho),
            "median_abs_log_error_per_shard":
                {n: st.median(v) for n, v in worst.items()},
        }

    p = ROOT / "runs" / "h28_twofeat.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"H28: log s* regressed leave-one-shard-out, {out['n_seeds']} seeds, "
          f"{out['n_shards']} shards\n")
    print(f"{'arm':16s} {'global /24':>11s} {'per-family /24':>15s} "
          f"{'rank corr of prediction':>25s}")
    for tag, d in out["arms"].items():
        print(f"{tag:16s} {d['global_link_median']:11.1f} "
              f"{d['per_family_link_median']:15.1f} "
              f"{d['loso_pred_spearman_median']:25.3f}")
    print()
    d = out["arms"]["2feat"]
    print("2-feature fit, worst shards by |log(predicted / s*)| (median):")
    for n, v in sorted(d["median_abs_log_error_per_shard"].items(),
                       key=lambda kv: -kv[1])[:6]:
        print(f"  {n.split('/')[1]:24s} off by {np.exp(v):7.2f}x")
    print("\nbest shards:")
    for n, v in sorted(d["median_abs_log_error_per_shard"].items(),
                       key=lambda kv: kv[1])[:4]:
        print(f"  {n.split('/')[1]:24s} off by {np.exp(v):7.2f}x")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
