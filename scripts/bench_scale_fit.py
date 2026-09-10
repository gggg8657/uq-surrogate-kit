"""Time `QuantileScale.fit` against torch's thread count, and write the table.

    ~/miniforge3/envs/pdeno/bin/python scripts/bench_scale_fit.py \
        --out runs/scale_fit_threads.json

`uqkit/scale.py` pins the fit to 4 threads and its docstring cites this file for
the reason. It exists because the docstring originally cited a number that had
not been measured at the size it was quoted for -- a 96-thread timing that was
arithmetic on a 50-step run at a third of the rows. This measures the grid at
the sizes the fit actually runs at, so the constant in the code has a run
behind it.

The problem is small (about 2e4 rows x 44 features, a few thousand full-batch
steps), which is exactly the regime where one-thread-per-core costs more in
synchronisation than it buys in arithmetic.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch


def bench(n_rows, n_feat, steps, threads, seed=0):
    """One timing. Same arithmetic as `QuantileScale.fit`, no model object."""
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n_feat, generator=g, dtype=torch.float64)
    y = torch.randn(n_rows, generator=g, dtype=torch.float64)
    w = torch.zeros(n_feat, dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(5):                     # warm up the allocator and threads
        opt.zero_grad()
        d = y - (x @ w + b)
        (torch.maximum(0.9 * d, -0.1 * d).mean() + 1e-3 * (w @ w)).backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        d = y - (x @ w + b)
        (torch.maximum(0.9 * d, -0.1 * d).mean() + 1e-3 * (w @ w)).backward()
        opt.step()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/scale_fit_threads.json")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    grids = [(21760, 44), (7360, 44)]
    thread_grid = sorted({1, 2, 4, 8, 16, default_threads})
    rows = []
    for n_rows, n_feat in grids:
        for th in thread_grid:
            ts = [bench(n_rows, n_feat, args.steps, th, seed=r)
                  for r in range(args.repeats)]
            rows.append({"n_rows": n_rows, "n_feat": n_feat,
                         "steps": args.steps, "threads": th,
                         "seconds_median": float(np.median(ts)),
                         "seconds_min": float(min(ts)),
                         "seconds_max": float(max(ts)),
                         "repeats": args.repeats,
                         "is_torch_default": th == default_threads})
            print(f"  {n_rows:6d} rows, {th:3d} threads: "
                  f"{np.median(ts):.3f}s median of {args.repeats} "
                  f"({min(ts):.3f}-{max(ts):.3f})", flush=True)
    torch.set_num_threads(default_threads)

    best = {}
    for r in rows:
        k = r["n_rows"]
        if k not in best or r["seconds_median"] < best[k]["seconds_median"]:
            best[k] = r
    out = {"host": platform.node(), "torch": torch.__version__,
           "torch_default_threads": default_threads,
           "steps": args.steps, "repeats": args.repeats,
           "chosen_in_code": 4,
           "rows": rows,
           "fastest_by_size": {str(k): {"threads": v["threads"],
                                        "seconds_median": v["seconds_median"]}
                               for k, v in best.items()},
           "speedup_of_4_over_default": {
               str(k): (
                   [r for r in rows
                    if r["n_rows"] == k and r["is_torch_default"]][0]
                   ["seconds_median"]
                   / [r for r in rows
                      if r["n_rows"] == k and r["threads"] == 4][0]
                   ["seconds_median"])
               for k in best}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
