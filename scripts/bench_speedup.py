"""Surrogate-vs-solver timing, with everything the claim needs attached.

    CUDA_VISIBLE_DEVICES=2 python scripts/bench_speedup.py --ckpts runs/m*/best.pt

What this measures, and the three choices that decide whether the number means
anything:

1. **Both sides on the same device.** The reference solvers are torch ops, so
   the GPU rows time the solver on the same H100 as the surrogate. The CPU rows
   time both on CPU. There is no row that compares a GPU surrogate to a CPU
   solver, because that is a hardware benchmark.
2. **The uncertainty is not free.** Three surrogate columns are timed:
   `single` (one member), `ensemble` (all M, which is what produces sigma), and
   `ensemble+residual` (what the kit actually runs when it also wants a trust
   score). Quoting `single` while shipping `ensemble+residual` would inflate the
   speedup by roughly M.
3. **The accuracy is quoted with the ratio.** Each row carries the surrogate's
   measured rel-L2 on that family and a verbatim description of what the
   reference solver was converged to. For the four families with an exact
   spectral propagator the "solver" is a single FFT pair, and the surrogate is
   expected to *lose*; that row is kept rather than dropped, because dropping it
   is how a speedup table becomes a selection effect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.bench import env_report, timeit  # noqa: E402
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402


def make_surrogate_fn(models, a_norm, tid, k, residual=None, a_raw=None,
                      autocast=True):
    """Closure timing the first `k` members, optionally plus the residual check."""
    ms = models[:k]

    def fn():
        preds = []
        for m in ms:
            if autocast and a_norm.is_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(m(a_norm, tid).float())
            else:
                preds.append(m(a_norm, tid))
        p = torch.stack(preds)
        mean = p.mean(0)
        _ = p.std(0, unbiased=True)
        if residual is not None:
            _ = residual(a_raw, mean)
        return mean
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/bench.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cpu-iters", type=int, default=5)
    ap.add_argument("--tasks", default="poisson,helmholtz,diffusion,advdiff,darcy")
    ap.add_argument("--extra-tasks", default="navier_stokes",
                    help="families the surrogate was NOT trained on, timed for "
                         "the solver cost only")
    ap.add_argument("--no-cpu", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    models, ck = load_members(args.ckpts, "cuda")
    stats = ck["stats"]
    M = len(models)
    res = {"env": env_report("cuda"), "n_members": M, "rows": [],
           "note": ("rel_l2 is the ensemble mean's error on that family's test "
                    "split, measured in the same bf16-autocast configuration "
                    "the GPU rows are timed in.")}

    # accuracy first: a speedup row without it is not emitted
    acc = {}
    for t in tasks:
        blob = load_shard(args.root, t, "test", 64)
        mean, _, truth, _ = predict_shard(models, blob, stats, "cuda")
        acc[t] = float(rel_l2(mean, truth).mean())
        del mean, truth
    torch.cuda.empty_cache()
    res["accuracy"] = acc
    print("ensemble rel-L2:", {k: round(v, 5) for k, v in acc.items()}, flush=True)

    for device in (["cuda"] if args.no_cpu else ["cuda", "cpu"]):
        if device == "cpu":
            models_d = [m.to("cpu").eval() for m in models]
            torch.set_num_threads(torch.get_num_threads())
        else:
            models_d = [m.to("cuda").eval() for m in models]
        n_iter = args.iters if device == "cuda" else args.cpu_iters
        for t in tasks + [x for x in args.extra_tasks.split(",") if x]:
            sim = PDE2DSimulator(t, device=device)
            trained = t in tasks
            for B in args.batches:
                if device == "cpu" and B > 8:
                    continue          # a 64-sample fp64 PCG on CPU is minutes
                blob = load_shard(args.root, t, "test" if trained else "ood", 64)
                a_raw = blob["a"][:B].to(device)
                if a_raw.shape[0] < B:
                    continue
                st = stats[PARENT.get(t, t)]
                a_norm = ((a_raw - st["a_mean"].to(device))
                          / st["a_std"].to(device))
                tid = torch.full((B,), TASK_ID[PARENT.get(t, t)],
                                 device=device, dtype=torch.long)
                sol = timeit(lambda: sim.solve(a_raw), 3, n_iter, device)
                row = {"task": t, "device": device, "batch": B,
                       "trained": trained,
                       "solver_s": sol["median_s"],
                       "solver_s_per_sample": sol["median_s"] / B,
                       "solver_accuracy": sim.solver_accuracy(),
                       "surrogate_rel_l2": acc.get(t),
                       "threads": torch.get_num_threads(),
                       "solver_timing": sol, "surrogate": {}}
                has_res = sim.residual(a_raw, sim.solve(a_raw)) is not None
                variants = {"single": (1, False), "ensemble": (M, False)}
                if has_res:
                    variants["ensemble+residual"] = (M, True)
                for name, (k, use_res) in variants.items():
                    fn = make_surrogate_fn(
                        models_d, a_norm, tid, k,
                        residual=(sim.residual if use_res else None),
                        a_raw=a_raw)
                    tm = timeit(fn, 3, n_iter, device)
                    row["surrogate"][name] = {
                        "s": tm["median_s"], "s_per_sample": tm["median_s"] / B,
                        "speedup": sol["median_s"] / max(tm["median_s"], 1e-12),
                        "timing": tm}
                res["rows"].append(row)
                sp = {k: round(v["speedup"], 1) for k, v in row["surrogate"].items()}
                print(f"{t:14s} {device:4s} B={B:<3d} solver "
                      f"{sol['median_s']*1e3:9.3f} ms  speedup {sp}", flush=True)
                del a_raw, a_norm
                torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
