"""How repeatable is the *denominator* of a speedup row?

Written because two runs of `bench_speedup.py` an hour apart, same GPU, same
shard, same code, read the darcy batch-1 solver at **265.1 ms** and
**386.5 ms** -- a 46% swing that moved the eager surrogate arm from 71.7x to
105.8x without the surrogate changing at all. A speedup whose denominator moves
that much is not a measurement of the surrogate.

This repeats the solver timing as N independent *processes-worth* of trials
inside one process (fresh warmup each trial, so a trial is comparable to what a
bench run does once), and reports the distribution rather than a point. It also
records the PCG iteration count per trial when the simulator exposes it, which
is the obvious candidate cause: an iterative solve to a fixed tolerance takes
as long as that sample needs.

    python scripts/probe_solver_repeat.py --task darcy --batches 1 64 \
        --trials 12 --out runs/solver_repeat.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.bench import env_report, timeit, warmup_device  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--task", default="darcy")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="runs/solver_repeat.json")
    args = ap.parse_args()

    ramp = warmup_device("cuda")
    sim = PDE2DSimulator(args.task, device="cuda")
    blob = load_shard(args.root, args.task, "test", 64)
    res = {"env": env_report("cuda"), "task": args.task, "clock_ramp": ramp,
           "trials": args.trials, "iters_per_trial": args.iters, "batches": {},
           "note": ("Each trial is a full `timeit` (3 warmup + --iters timed "
                    "calls) on the same tensor, so one trial is exactly what a "
                    "bench run measures once. The spread ACROSS trials is the "
                    "run-to-run uncertainty a single bench row does not show. "
                    "Nothing here involves the surrogate.")}

    for B in args.batches:
        a_raw = blob["a"][:B].to("cuda")
        if a_raw.shape[0] < B:
            continue
        meds, iters = [], []
        for _ in range(args.trials):
            tm = timeit(lambda: sim.solve(a_raw), 3, args.iters, "cuda")
            meds.append(tm["median_s"])
            n_it = getattr(sim, "last_iters", None)
            if n_it is not None:
                iters.append(int(n_it))
        meds_s = sorted(meds)
        lo, hi = meds_s[0], meds_s[-1]
        res["batches"][str(B)] = {
            "batch": B,
            "trial_median_s": meds,
            "min_s": lo, "max_s": hi,
            "median_of_medians_s": statistics.median(meds),
            "mean_s": statistics.fmean(meds),
            "sd_s": statistics.stdev(meds) if len(meds) > 1 else 0.0,
            "spread_ratio_max_over_min": hi / max(lo, 1e-12),
            "rel_range_pct": 100.0 * (hi - lo) / statistics.fmean(meds),
            "pcg_iters_per_trial": iters or None,
        }
        r = res["batches"][str(B)]
        print(f"{args.task} B={B:<3d} median-of-medians "
              f"{r['median_of_medians_s']*1e3:8.2f} ms  "
              f"range {lo*1e3:.2f}-{hi*1e3:.2f} ms  "
              f"max/min {r['spread_ratio_max_over_min']:.3f}  "
              f"rel-range {r['rel_range_pct']:.1f}%", flush=True)
        del a_raw
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
