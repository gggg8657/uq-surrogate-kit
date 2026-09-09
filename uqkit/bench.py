"""Timing harness for the speedup claim, built so the claim is falsifiable.

A surrogate speedup is meaningless without four things attached, so this module
refuses to produce a number without them:

* **the same hardware for both sides.** The reference solvers here are torch
  ops, so the solver runs on the same GPU as the surrogate. A GPU surrogate
  against a CPU solver is a hardware comparison wearing a modelling costume.
* **the thread count.** Reported for every run, CPU and GPU, because a solver
  left on one thread makes any surrogate look good.
* **per-sample and batched, both.** Batched throughput flatters the surrogate
  (it amortizes kernel launches over 64 samples); per-sample latency is what a
  controller in the loop actually sees. They differ here by more than an order
  of magnitude, so quoting only one is a choice about what to hide.
* **the accuracy at which the speedup holds.** 1000x at 10% error is not
  comparable to a solver converged to 1e-10. Every record carries the
  surrogate's measured rel-L2 and the solver's own accuracy setting.
"""
from __future__ import annotations

import os
import platform
import statistics
import time

import torch


def env_report(device="cuda"):
    """Everything a reader needs to know the two sides ran on the same machine."""
    out = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "<unset>"),
        "cpu_count": os.cpu_count(),
    }
    if device == "cuda" and torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0)
        out["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        out["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    return out


def warmup_device(device="cuda", seconds=6.0, n=4096):
    """Drive the GPU to steady clocks before anything is timed.

    An idle H100 boosts over the first few seconds of load. Measured by
    `scripts/bench_noise.py`: the same Darcy solve took 376 ms on the first two
    repeats and 218 ms from the third on, and the 5-member surrogate went
    15.6 ms to 8.9 ms over the same transition -- a 42% drift on both sides.
    Per-call warmup does not fix it, because each configuration is warmed
    separately and the ramp spans several configurations. The damage is not the
    absolute times, which cancel in a ratio, but the *transition*: a
    configuration timed cold against one timed hot produced a 41.6x reading of
    a comparison whose steady-state value is 24.4x, and that inflated reading is
    what an earlier version of `RESULTS.md` reported as the headline speedup.

    Called once at the top of every benchmark script, before any measurement.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {"ramped": False, "reason": "not cuda"}
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < seconds:
        a @ b
        k += 1
        if k % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()
    return {"ramped": True, "seconds": seconds, "matmuls": k}


def timeit(fn, n_warmup=15, n_iter=20, device="cuda"):
    """Median wall-clock seconds per call, with the spread kept.

    Median rather than mean: a single stray 40 ms from a driver hiccup moves a
    20-run mean by 2 ms, which at these magnitudes is the whole measurement.
    The full sample is returned so `report.py` can show the IQR.
    """
    use_cuda = device == "cuda" and torch.cuda.is_available()
    for _ in range(n_warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    return {
        "median_s": statistics.median(times),
        "min_s": times[0],
        "p25_s": times[max(0, n // 4 - 1)],
        "p75_s": times[min(n - 1, (3 * n) // 4)],
        "n_iter": n_iter,
    }


def speedup_record(name, batch, solver_fn, surrogate_fn, rel_l2, solver_accuracy,
                   n_iter=20, device="cuda", n_warmup=5):
    """One row of the speedup table: both sides timed the same way.

    `rel_l2` is the surrogate's measured error on this family and
    `solver_accuracy` a free-form string describing what the reference solver
    was converged to. Both are required arguments rather than optional metadata
    -- a speedup without them is not a result this kit will emit.
    """
    ts = timeit(solver_fn, n_warmup, n_iter, device)
    tm = timeit(surrogate_fn, n_warmup, n_iter, device)
    return {
        "task": name,
        "batch": batch,
        "solver_s_per_batch": ts["median_s"],
        "surrogate_s_per_batch": tm["median_s"],
        "solver_s_per_sample": ts["median_s"] / batch,
        "surrogate_s_per_sample": tm["median_s"] / batch,
        "speedup": ts["median_s"] / max(tm["median_s"], 1e-12),
        "surrogate_rel_l2": float(rel_l2),
        "solver_accuracy": solver_accuracy,
        "solver_timing": ts,
        "surrogate_timing": tm,
    }
