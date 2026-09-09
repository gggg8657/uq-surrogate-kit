"""The timing harness's own run-to-run spread, measured before any ratio is quoted.

    CUDA_VISIBLE_DEVICES=3 python scripts/bench_noise.py --ckpts runs/m*/best.pt

Two invocations of `bench_isoaccuracy.py` an hour apart reported 3.36x and 2.12x
for the same Darcy iso-accuracy comparison. That is a 37% swing on the number
the speedup clause turns on, and until it is quantified every ratio in
`RESULTS.md` is quoted to more precision than it has.

The weekend rule for seeds applies to wall-clock too: repeat one configuration,
report the spread, and do not claim an effect smaller than it. This repeats the
whole measurement -- fresh process state, fresh warmup, fresh allocator -- R
times and reports the median and the full range for the solver, the surrogate
and their ratio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.bench import env_report, timeit, warmup_device  # noqa: E402
from uqkit.sims import pde2d as P  # noqa: E402
from uqkit.sims.pde2d import TASK_ID  # noqa: E402
from uqkit.sims.predict import load_members, load_shard  # noqa: E402


def spread(xs):
    xs = np.asarray(xs, dtype=float)
    return {"median": float(np.median(xs)), "min": float(xs.min()),
            "max": float(xs.max()),
            "rel_range": float((xs.max() - xs.min()) / np.median(xs)),
            "values": xs.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/bench_noise.json")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    ramp = warmup_device("cuda")
    print(f"clock ramp: {ramp}", flush=True)
    dev = "cuda"
    models, ck = load_members(args.ckpts, dev)
    stats = ck["stats"]
    B = args.batch
    blob = load_shard(args.root, "darcy", "test", 64)
    a = blob["a"][:B].to(dev)
    coef, f = torch.exp(a[:, 0]), a[:, 1]
    st = stats["darcy"]
    a_norm = (a - st["a_mean"].to(dev)) / st["a_std"].to(dev)
    tid = torch.full((B,), TASK_ID["darcy"], device=dev, dtype=torch.long)

    def sur():
        preds = []
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(m(a_norm, tid).float())
        p = torch.stack(preds)
        _ = p.std(0, unbiased=True)
        return p.mean(0)

    cfgs = {
        "solver_tol_1e-10": lambda: P.solve_darcy(coef, f, tol=1e-10),
        "solver_tol_1e-1": lambda: P.solve_darcy(coef, f, tol=1e-1),
        "surrogate_5member": sur,
    }
    obs = {k: [] for k in cfgs}
    for r in range(args.repeats):
        torch.cuda.empty_cache()
        for k, fn in cfgs.items():
            obs[k].append(timeit(fn, 5, args.iters, dev)["median_s"])
        print(f"repeat {r}: " + "  ".join(f"{k}={obs[k][-1]*1e3:.2f}ms"
                                          for k in cfgs), flush=True)

    res = {"env": env_report(dev), "clock_ramp": ramp, "repeats": args.repeats, "iters": args.iters,
           "batch": B, "timings": {k: spread(v) for k, v in obs.items()}}
    for name, num, den in (("headline_speedup", "solver_tol_1e-10", "surrogate_5member"),
                           ("isoaccuracy_speedup", "solver_tol_1e-1", "surrogate_5member")):
        ratios = [n / d for n, d in zip(obs[num], obs[den])]
        res[name] = spread(ratios)
        s = res[name]
        print(f"{name}: median {s['median']:.2f}x  range "
              f"{s['min']:.2f}-{s['max']:.2f}x  (+/-{100*s['rel_range']/2:.0f}%)",
              flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
