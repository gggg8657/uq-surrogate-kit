"""H16: how accurately would the interval width have to be predicted?

    CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/pdeno/bin/python \
        scripts/eval_width_tolerance.py --ckpt runs/u0/best.pt \
        --sigma-source het --out runs/wtol_u0_het.json

H15 measured a width model h(z) whose median prediction lands within a factor
of about 0.76-2.56 of the quantile it is trying to predict, and got coverage
anywhere from 0.000 to 0.990 out of it. Those two facts are only compatible if
coverage is *hypersensitive* to the width -- and if it is, the binding
constraint on clause 1 under shift is a property of the score, not of any
uncertainty method, and no amount of work on h can fix it.

This script measures that sensitivity exactly, with no method in the loop at
all. For a per-sample score S and a band [mu - w, mu + w], coverage is
P(S <= w/sigma-scale), so the width that achieves *exactly* a target coverage p
is the p-th quantile of S. Therefore:

    the widths that keep coverage inside the KPI band [0.88, 0.92]
    span exactly [Q_0.88(S), Q_0.92(S)],

and the **relative width tolerance**

    tol = (Q_0.92(S) - Q_0.88(S)) / Q_0.90(S)

is the fractional error a width predictor is allowed to make on that shard
before the clause fails. It is three order statistics of S. There is nothing
to tune, nothing to fit, and no way for it to flatter anything.

Why this is the right number to have: it converts "the KPI is hard" into "the
width must be right to within tol", and tol is a property of the *score
definition*. `field_max` is a maximum over 4,096 pixels, so extreme-value
concentration makes its distribution tight and tol small; `norm_ratio` and
`rel_l2` aggregate instead of maximising. All three are already computed by
this repo, so which of them is even *capable* of a +/-2pp band under an
imperfect width model is measurable rather than arguable.

The tolerance is reported per shard and per score, in distribution and under
every covariate shift. Nothing here changes a protocol: it is a measurement of
the protocol the repo already uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import GroupConformal, get_score  # noqa: E402
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

SCORES = ["field_max", "norm_ratio", "rel_l2"]
COVARIATE_KINDS = {"input_shift", "graded_rough"}
BAND = (0.88, 0.92)


def tolerance(s, band=BAND, target=0.90):
    """The relative width tolerance of a per-sample score array.

    Returns the widths that bracket the KPI band and the fractional slack
    between them. `q_target` is also the width a *perfect* width model would
    emit on this shard, which is what makes the ratio interpretable.
    """
    s = np.asarray(s, dtype=np.float64)
    q_lo, q_mid, q_hi = (float(np.quantile(s, band[0])),
                         float(np.quantile(s, target)),
                         float(np.quantile(s, band[1])))
    return {
        "q_lo": q_lo, "q_target": q_mid, "q_hi": q_hi,
        # fractional slack: how wrong a width predictor may be, either way
        "tol_rel": (q_hi - q_lo) / q_mid if q_mid > 0 else None,
        "tol_rel_up": (q_hi - q_mid) / q_mid if q_mid > 0 else None,
        "tol_rel_down": (q_mid - q_lo) / q_mid if q_mid > 0 else None,
        # the same thing as a multiplicative factor, which is how a width
        # model's error is naturally expressed
        "factor_hi": q_hi / q_mid if q_mid > 0 else None,
        "factor_lo": q_lo / q_mid if q_mid > 0 else None,
        "n": int(len(s)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default=None,
                    choices=["het", "cqr", "const"])
    args = ap.parse_args()
    single = args.sigma_source is not None
    if single and len(args.ckpt) != 1:
        ap.error("--sigma-source is a single-network mode; pass one checkpoint")

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    if single:
        model, ck = load_model(args.ckpt[0], args.device)
        if not ck["args"].get("uq"):
            raise SystemExit(f"{args.ckpt[0]} is not a UQ checkpoint")
        models = [model]
    else:
        models, ck = load_members(args.ckpt, args.device)
    stats = ck["stats"]

    def predict(blob):
        if single:
            return predict_shard_single(models[0], blob, stats, args.device,
                                        sigma_source=args.sigma_source)
        return predict_shard(models, blob, stats, args.device)

    # sigma floor frozen on pooled calibration, exactly as everywhere else
    cal = {}
    parts = []
    for t in in_tasks:
        m, s, tr, a = predict(load_shard(args.root, t, "cal"))
        cal[t] = (m, s, tr)
        parts.append(s.flatten())
    MED = float(torch.cat(parts).median())
    del parts

    res = {"alpha": args.alpha, "band": list(BAND), "ckpt": args.ckpt,
           "seed": ck["args"].get("seed"),
           "sigma_source": args.sigma_source or "ensemble_spread",
           "sigma_floor_median": MED, "sigma_floor_frac": 0.05,
           "note": ("tol_rel is a property of the score distribution on the "
                    "shard, computed from three order statistics. No "
                    "uncertainty method enters it."),
           "by_score": {}}

    specs = []
    for kind, tasks in man["ood_suite"].items():
        if kind in COVARIATE_KINDS:
            specs += [(kind, t, 64) for t in tasks]

    # cache predictions once; the three scores share them
    shard_pred = {}
    for kind, task, N in specs:
        m, s, tr, a = predict(load_shard(args.root, task, "ood", N))
        shard_pred[f"{kind}/{task}/N{N}"] = {
            "kind": kind, "parent": PARENT.get(task, task), "N": N,
            "m": m, "s": s, "tr": tr,
            "rel_l2": float(rel_l2(m, tr).mean())}
    test = {}
    for t in in_tasks:
        m, s, tr, _a = predict(load_shard(args.root, t, "test"))
        test[t] = (m, s, tr)

    for sname in SCORES:
        fn, per_sample = get_score(sname)
        if not per_sample:
            continue
        cal_sc = {t: fn(cal[t][0], cal[t][1], cal[t][2], med=MED).cpu().numpy()
                  for t in in_tasks}
        pooled = np.concatenate([cal_sc[t] for t in in_tasks])
        grp = np.concatenate([[t] * len(cal_sc[t]) for t in in_tasks])
        group = GroupConformal(args.alpha).fit(pooled, grp)

        out = {"q_group": {k: float(v) for k, v in group.q.items()},
               "in_dist": {}, "shards": {}}
        for t in in_tasks:
            m, s, tr = test[t]
            sc = fn(m, s, tr, med=MED).cpu().numpy()
            tol = tolerance(sc)
            # what the calibrated width already is, relative to the width that
            # would hit 0.90 exactly on this set
            tol["deployed_width_over_ideal"] = (
                float(group.q[t] / tol["q_target"]) if tol["q_target"] else None)
            out["in_dist"][t] = tol
        for name, d in shard_pred.items():
            sc = fn(d["m"], d["s"], d["tr"], med=MED).cpu().numpy()
            tol = tolerance(sc)
            qg = group.q.get(d["parent"], group.q_pooled)
            tol.update({
                "kind": d["kind"], "parent": d["parent"], "N": d["N"],
                "rel_l2_mean": d["rel_l2"],
                # how far the *deployed* (in-distribution-calibrated) width is
                # from the width that would be exactly right on this shard
                "deployed_width_over_ideal": (float(qg / tol["q_target"])
                                              if tol["q_target"] else None),
            })
            out["shards"][name] = tol
        tols = [v["tol_rel"] for v in out["shards"].values()
                if v["tol_rel"] is not None]
        out["tol_rel_median_over_shards"] = float(np.median(tols))
        out["tol_rel_min_over_shards"] = float(np.min(tols))
        out["tol_rel_max_over_shards"] = float(np.max(tols))
        out["tol_rel_median_in_dist"] = float(np.median(
            [v["tol_rel"] for v in out["in_dist"].values()]))
        res["by_score"][sname] = out
        print(f"[{sname:11s}] width tolerance, median over 32 covariate "
              f"shards: {out['tol_rel_median_over_shards']*100:6.2f}%  "
              f"(in distribution {out['tol_rel_median_in_dist']*100:.2f}%, "
              f"range over shards "
              f"{out['tol_rel_min_over_shards']*100:.2f}-"
              f"{out['tol_rel_max_over_shards']*100:.2f}%)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
