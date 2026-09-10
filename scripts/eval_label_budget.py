"""H7: how many labels does it take to re-certify the interval after a shift?

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_label_budget.py \
        --ckpts runs/m0/best.pt --out runs/label_budget.json

Two-sided 90 +- 2% coverage under arbitrary covariate shift, without labels, is
not attainable: weighted conformal gives one-sided conservative validity only,
and when the calibration and test inputs are separable it returns `q = +inf`.
That is not a theoretical worry here -- `runs/conformal.json` has probe AUC
1.000 on every shard, 30/32 shards with at least one infinite quantile and 14/32
infinite for all 512 test points. An infinite interval covers everything and
certifies nothing, which is why this file reports **width** next to every
coverage number and refuses to report one without it.

So the honest question is not "can it be done unlabelled" but "what does it
cost". This measures that, in labels:

* **k-label recalibration** -- draw k labelled samples from the shifted shard,
  take the split-conformal quantile of their scores, and test coverage on the
  samples that were NOT drawn. Repeated over `--repeats` independent draws.
  `conformal_quantile` returns +inf whenever `ceil((k+1)(1-alpha)) > k`, i.e.
  for every k < 9 at alpha = 0.1, so the curve is *forced* to start at infinite
  width and the k at which it stops being infinite is arithmetic, not a
  finding. What is measured is where coverage enters [88, 92] and how wide the
  interval is when it does.
* **the oracle target split** (`k = n/2`) -- the same thing at the largest
  budget the shard can pay. It separates two explanations of the unlabelled
  failure: if the oracle lands in band, `field_max` is a sound score under
  shift and the whole failure is calibration transfer; if it does not, the
  score distribution itself is pathological under shift and no reweighting can
  fix it.

**Both are oracles. Neither is a deployable method** -- they consume labels
from the shard they certify, which is exactly what a surrogate exists to avoid.
They are reported as an upper bound on what any calibrator could achieve on
this data, and as a price list. Every table this writes says so in its own row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import _floor, conformal_quantile, get_score  # noqa: E402
from uqkit.metrics import binom_ci  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

BAND = (0.88, 0.92)
KS = (1, 2, 3, 5, 8, 9, 12, 16, 24, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/label_budget.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-source", default=None,
                    choices=["het", "cqr", "const"])
    args = ap.parse_args()
    single = args.sigma_source is not None
    if single and len(args.ckpts) != 1:
        ap.error("--sigma-source is a single-network mode; pass one checkpoint")

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    if single:
        model, ck = load_model(args.ckpts[0], args.device)
        if not ck["args"].get("uq"):
            raise SystemExit(f"{args.ckpts[0]} is not a UQ checkpoint")
        models = [model]
    else:
        models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]
    fn, _per_sample = get_score(args.score)

    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        if single:
            m, s, t, a = predict_shard_single(models[0], blob, stats,
                                              args.device,
                                              sigma_source=args.sigma_source)
        else:
            m, s, t, a = predict_shard(models, blob, stats, args.device)
        return {"mean": m, "sigma": s, "truth": t, "parent": PARENT.get(task, task)}

    # the sigma floor is frozen on the pooled in-distribution calibration split
    # and passed to every score call, exactly as `eval_conformal.py` does it.
    # Letting each shard set its own floor would move every band width in the
    # direction of apparent coverage.
    cal = {t: gather(t, "cal") for t in in_tasks}
    MED = float(torch.cat([cal[t]["sigma"].flatten() for t in in_tasks]).median())
    FRAC = 0.05

    ood_specs = []
    for kind, tasks in man["ood_suite"].items():
        ood_specs += [(kind, t, 64) for t in tasks]
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                ood_specs.append((f"resolution_{N}", t, N))

    rng = np.random.default_rng(args.seed)
    res = {"alpha": args.alpha, "score": args.score, "band": list(BAND),
           "ks": list(KS), "repeats": args.repeats, "ckpts": args.ckpts,
           "n_members": len(models), "sigma_floor_median": MED,
           "sigma_floor_frac": FRAC,
           "sigma_source": args.sigma_source or "ensemble_spread",
           "k_min_finite": int(np.ceil((1 - args.alpha) / args.alpha)),
           "oracle_warning":
               "Every number in this file consumes labels from the shard it "
               "certifies. It is an upper bound on what a calibrator could do "
               "on this data and a price list in labels -- NOT a deployable "
               "method, and not evidence for the unlabelled KPI clause.",
           "shards": {}}

    def curve(s, half_width):
        """Coverage and width vs k, averaged over independent labelled draws."""
        s = np.asarray(s, dtype=np.float64)
        hw = np.asarray(half_width, dtype=np.float64)
        n = len(s)
        out = {}
        for k in list(KS) + [n // 2]:
            if k >= n:
                continue
            covs, qs, n_inf = [], [], 0
            for _ in range(args.repeats):
                idx = rng.permutation(n)
                q = conformal_quantile(s[idx[:k]], args.alpha)
                rest = idx[k:]
                if not np.isfinite(q):
                    n_inf += 1
                    covs.append(1.0)          # an infinite interval covers all
                    continue
                qs.append(float(q))
                covs.append(float((s[rest] <= q).mean()))
            key = "oracle_half" if k == n // 2 else f"k{k}"
            mean_cov = float(np.mean(covs))
            lo, hi = binom_ci(int(round(mean_cov * (n - k))), n - k)
            out[key] = {
                "k": int(k), "coverage": mean_cov,
                "coverage_ci95": [lo, hi],
                "frac_draws_infinite": n_inf / args.repeats,
                "q_median": float(np.median(qs)) if qs else None,
                # the band's own half-width in the field's units, so a 100%
                # coverage row that is infinitely wide cannot read as a pass
                "mean_half_width": (float(np.median(qs) * hw.mean())
                                    if qs else float("inf")),
                "in_band": bool(BAND[0] <= mean_cov <= BAND[1])}
        return out

    def half_width_unit(d):
        """Mean per-sample sigma after the floor -- the band is q * this."""
        return _floor(d["sigma"], FRAC, MED).flatten(1).mean(1).cpu().numpy()

    todo = [("in_dist", t, t, 64) for t in in_tasks] + \
           [(kind, t, PARENT.get(t, t), N) for kind, t, N in ood_specs]
    for kind, task, parent, N in todo:
        d = gather(task, "test" if kind == "in_dist" else "ood", N)
        s = fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy()
        c = curve(s, half_width_unit(d))
        res["shards"][f"{kind}/{task}/N{N}"] = {
            "kind": kind, "parent": parent, "N": N, "n": int(len(s)),
            "curve": c,
            "k_min_in_band": next((c[f"k{k}"]["k"] for k in KS
                                   if f"k{k}" in c and c[f"k{k}"]["in_band"]),
                                  None)}
        o = c.get("oracle_half", {})
        k9 = c.get("k9", {})
        print(f"{kind[:14]:14s} {task:16s} N{N:<4d} n={len(s):4d}  "
              f"k=9 cov {k9.get('coverage', float('nan')):.3f}  "
              f"oracle(k={o.get('k', 0)}) cov {o.get('coverage', float('nan')):.3f} "
              f"{'IN BAND' if o.get('in_band') else '       '}  "
              f"k_min_in_band={res['shards'][f'{kind}/{task}/N{N}']['k_min_in_band']}",
              flush=True)
        del d
        torch.cuda.empty_cache()

    ood = [v for k, v in res["shards"].items() if v["kind"] != "in_dist"]
    res["summary"] = {
        "n_ood_shards": len(ood),
        "oracle_half_in_band": sum(1 for v in ood
                                   if v["curve"].get("oracle_half", {}).get("in_band")),
        "k9_in_band": sum(1 for v in ood
                          if v["curve"].get("k9", {}).get("in_band")),
        "median_k_min_in_band": float(np.median(
            [v["k_min_in_band"] for v in ood if v["k_min_in_band"] is not None]))
        if any(v["k_min_in_band"] is not None for v in ood) else None,
        "n_never_in_band": sum(1 for v in ood if v["k_min_in_band"] is None)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
