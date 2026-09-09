"""How many labelled samples does it take to notice an operator shift?

    CUDA_VISIBLE_DEVICES=3 python scripts/eval_label_probe.py --ckpts runs/m*/best.pt

`eval_ood.py` measures that every unsupervised detector sits at chance on an
operator shift whose input distribution is unchanged. That is not a tuning
failure, it is an identity: `spread`, `mahalanobis` and `residual` are all
functions of the input and the *configured* operator only, and an operator shift
changes neither. A surrogate handed `f` and configured as Poisson returns the
Poisson solution, which has a small Poisson residual and a small ensemble
spread, and is a perfectly good answer to a question nobody asked.

So the useful question is not "can it be detected unsupervised" -- it cannot --
but **how expensive is the cheapest thing that does detect it**: a handful of
labelled probes, run through the reference solver, whose observed error is
compared against the calibrated in-distribution error distribution.

The test is a conformal p-value, so it needs no distributional assumption:
under exchangeability with the calibration errors, the rank of a probe's error
is uniform, so `p = (1 + #{cal >= probe}) / (n_cal + 1)` is a valid p-value and
Fisher's method combines k of them. Reported per shift: the smallest k reaching
**95% detection power at a 5% false-alarm rate**, with the false-alarm rate
verified on held-out in-distribution samples rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402

KS = [1, 2, 3, 5, 8, 12, 20, 32, 64]
ALPHA = 0.05
POWER = 0.95


def conformal_p(cal_err, x):
    """One-sided conformal p-values: small p means 'larger error than calibration'."""
    n = len(cal_err)
    ge = (cal_err[None, :] >= np.asarray(x)[:, None]).sum(1)
    return (1.0 + ge) / (n + 1.0)


def fisher(ps):
    """-2 sum log p, over axis -1. Larger = stronger evidence of shift."""
    return -2.0 * np.log(np.clip(ps, 1e-300, 1.0)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/label_probe.json")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]
    rng = np.random.default_rng(args.seed)

    def errs(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        mean, _, truth, _ = predict_shard(models, blob, stats, args.device)
        e = rel_l2(mean, truth).cpu().numpy()
        del mean, truth
        torch.cuda.empty_cache()
        return e

    cal = {t: errs(t, "cal") for t in in_tasks}
    # the in-distribution test split plays the null: it is exchangeable with
    # cal, so the measured false-alarm rate there is the real size of the test
    null = {t: errs(t, "test") for t in in_tasks}

    # critical values of Fisher's statistic, calibrated *empirically* on the
    # null rather than taken from a chi-square table -- the p-values are
    # discrete (n_cal = 1024), so the asymptotic table is not exact
    crit = {}
    for k in KS:
        draws = []
        for t in in_tasks:
            p = conformal_p(cal[t], null[t])
            idx = rng.integers(0, len(p), (args.trials, k))
            draws.append(fisher(p[idx]))
        crit[k] = float(np.quantile(np.concatenate(draws), 1 - ALPHA))

    res = {"alpha": ALPHA, "target_power": POWER, "ks": KS,
           "n_cal": {t: int(len(cal[t])) for t in in_tasks},
           "trials": args.trials,
           "critical_values": crit,
           "false_alarm_measured": {}, "shifts": {}}
    for t in in_tasks:
        p = conformal_p(cal[t], null[t])
        fa = {}
        for k in KS:
            idx = rng.integers(0, len(p), (args.trials, k))
            fa[k] = float((fisher(p[idx]) > crit[k]).mean())
        res["false_alarm_measured"][t] = fa

    specs = []
    for kind, tasks in man["ood_suite"].items():
        specs += [(kind, t, 64) for t in tasks]
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                specs.append((f"resolution_{N}", t, N))

    for kind, task, N in specs:
        e = errs(task, "ood", N)
        parent = PARENT.get(task, task)
        p = conformal_p(cal[parent], e)
        power = {}
        for k in KS:
            idx = rng.integers(0, len(p), (args.trials, k))
            power[k] = float((fisher(p[idx]) > crit[k]).mean())
        k_needed = next((k for k in KS if power[k] >= POWER), None)
        res["shifts"][f"{kind}/{task}/N{N}"] = {
            "kind": kind, "parent": parent, "N": N,
            "rel_l2_mean": float(e.mean()),
            "rel_l2_in_dist": float(null[parent].mean()),
            "power": power, "k_for_95pct_power": k_needed,
            "frac_probes_above_cal_max": float((e > cal[parent].max()).mean())}
        print(f"{kind}/{task}/N{N:<4d} err {e.mean():10.4f}  k95="
              f"{k_needed if k_needed else '>' + str(KS[-1]):>4}  "
              f"power@1={power[1]:.2f} power@8={power[8]:.2f}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
