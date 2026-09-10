"""What the H17 scale wrapper costs, so "the 100x row is untouched" is measured.

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/bench_equivar_cost.py --ckpt runs/u0/best.pt \
        --out runs/equivar_cost.json

The H17 write-up says the wrapper is "one extra reduction per sample, so the
100x row is untouched". That was an argument, not a measurement, and this repo
has been wrong before about exactly this kind of "obviously negligible" claim
-- the cached-weight-permute finding was worth 1.246x and had been invisible
for three rounds of scrutiny.

So: time the unwrapped forward and the wrapped forward on the same hardware, at
batch 1 and batched, with the same warmup and repeat discipline the rest of the
benchmarks use, and report the added cost as a fraction. The clause-2 headline
is the worst field at 117.0x, so the question this answers is whether the
wrapper's overhead can push any field back under 100x -- i.e. whether the
overhead exceeds 14.5% of the forward.

Thread count and device are recorded. The wrapper's work is a mean-square
reduction over the linear channels plus two broadcast multiplies, all in the
same precision as the forward.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.equivar import (predict_equivariant,  # noqa: E402
                           reference_scale, sample_scale)
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402


def timeit(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return {"median_s": float(np.median(a)), "min_s": float(a.min()),
            "p90_s": float(np.quantile(a, 0.9)), "iters": iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/equivar_cost.json")
    ap.add_argument("--task", default="darcy",
                    help="darcy is the clause-2 headline family")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default="het")
    args = ap.parse_args()

    model, ck = load_model(args.ckpt, args.device)
    stats = ck["stats"]
    blob = load_shard(args.root, args.task, "test")
    ref = reference_scale(load_shard(args.root, args.task, "cal")["a"],
                          args.task)

    res = {"ckpt": args.ckpt, "task": args.task, "device": args.device,
           "sigma_source": args.sigma_source,
           "reference_scale": ref,
           "torch_threads": torch.get_num_threads(),
           "gpu": (torch.cuda.get_device_name(0)
                   if args.device.startswith("cuda") else None),
           "machine": platform.processor() or platform.machine(),
           "warmup": args.warmup,
           "note": ("overhead_frac is the wrapper's added wall-clock as a "
                    "fraction of the unwrapped forward ON THIS PATH -- eager "
                    "`predict_shard_single`, which includes normalization and "
                    "host transfers. The clause-2 number (117.0x on the worst "
                    "field) is measured under CUDA-graph replay on a "
                    "different path, so this fraction MUST NOT be multiplied "
                    "into it: an earlier version of this script did exactly "
                    "that and reported an 'implied speedup' that mixed two "
                    "protocols. Deciding the clause-2 impact requires running "
                    "`bench_fair.py` with the wrapper in the captured graph, "
                    "which has not been done."),
           "clause2_impact": "[not measured]",
           "by_batch": {}}

    for n in args.batches:
        sub = {"a": blob["a"][:n].contiguous(), "u": blob["u"][:n].contiguous(),
               "task": args.task, "N": blob["N"], "split": "test",
               "parent": args.task}

        def bare():
            return predict_shard_single(model, sub, stats, args.device,
                                        sigma_source=args.sigma_source)

        def wrapped():
            return predict_equivariant(
                lambda b: predict_shard_single(
                    model, b, stats, args.device,
                    sigma_source=args.sigma_source),
                sub, args.task, ref, device=args.device)

        a_dev = sub["a"].to(args.device)

        def scale_only():
            return sample_scale(a_dev, args.task, ref)

        b = timeit(bare, args.warmup, args.iters, args.device)
        w = timeit(wrapped, args.warmup, args.iters, args.device)
        sc = timeit(scale_only, args.warmup, args.iters, args.device)
        frac = (w["median_s"] - b["median_s"]) / b["median_s"]
        res["by_batch"][str(n)] = {
            "bare": b, "wrapped": w, "scale_reduction_only": sc,
            "overhead_s": w["median_s"] - b["median_s"],
            "overhead_frac": frac,
        }
        print(f"  batch {n:3d}: bare {b['median_s']*1e3:.3f} ms, wrapped "
              f"{w['median_s']*1e3:.3f} ms, overhead {frac*100:+.2f}%  "
              f"(scale reduction alone {sc['median_s']*1e6:.1f} us)",
              flush=True)

    # equivalence: the wrapper must not change the prediction beyond fp noise
    # at the reference scale, and this is the cheap guard on that
    m0, s0, _t, _a = predict_shard_single(model, {**blob, "a": blob["a"][:64],
                                                  "u": blob["u"][:64]},
                                          stats, args.device,
                                          sigma_source=args.sigma_source)
    m1, s1, _t1, _a1 = predict_equivariant(
        lambda b: predict_shard_single(model, b, stats, args.device,
                                       sigma_source=args.sigma_source),
        {"a": blob["a"][:64], "u": blob["u"][:64], "task": args.task,
         "N": blob["N"], "split": "test", "parent": args.task},
        args.task, ref)
    rel = ((m1 - m0).flatten(1).norm(dim=1)
           / m0.flatten(1).norm(dim=1).clamp_min(1e-30))
    res["in_dist_prediction_change"] = {
        "rel_l2_median": float(rel.median()),
        "rel_l2_max": float(rel.max()),
        "note": ("the wrapper is NOT the identity in distribution -- s varies "
                 "per sample around 1 -- so this is a magnitude, not a "
                 "bit-equality check")}
    print(f"  in-distribution prediction change: median "
          f"{float(rel.median()):.2e}, max {float(rel.max()):.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
