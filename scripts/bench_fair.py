"""Clause 2 with the subsidies removed from BOTH sides.

`bench_speedup.py` times the surrogate eagerly with autograd on, and times
`solve_darcy` with `check_every=1` -- a device-to-host sync on every one of up
to 2000 PCG iterations. H8 removed the surrogate's dispatch overhead (CUDA-graph
replay, bit-exact) and got 602x at darcy b=1. That number is only meaningful if
the reference solver gets the *same* treatment, and it does not get it in that
script. This one gives it.

Three things this measures that the existing benches do not:

1. **The solver's own dispatch fix.** `check_every` in {1, 10, 50, 100}. The
   convergence test can only fire late, never early, so a larger stride returns
   an iterate that is at least as converged -- the achieved residual is recorded
   per setting and a setting that misses `tol` is marked inadmissible rather
   than quietly used.
2. **A denominator with an interval.** Repeated trials, because the darcy b=1
   solver was measured at 221.8-379.9 ms across 12 trials of identical work
   (`runs/solver_repeat.json`, max/min 1.71). A point estimate here is worth
   +-35%.
3. **A conservative pairing.** The headline ratio uses the *fastest* admissible
   solver trial over the *slowest* surrogate trial, which is the reading least
   favourable to us. The median/median reading is reported next to it.

    python scripts/bench_fair.py --task darcy --batches 1 64 --trials 5 \
        --out runs/bench_fair.json
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
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.sims import pde2d as P  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard_single)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_speedup import graph_wrap, make_uq_fn, no_grad_wrap  # noqa: E402

TOL = 1e-10


def trials(fn, n_trials, iters, device="cuda"):
    """Repeat a full `timeit` n_trials times; keep every median."""
    meds = []
    for _ in range(n_trials):
        meds.append(timeit(fn, 3, iters, device)["median_s"])
    meds.sort()
    return {"trial_median_s": meds, "min_s": meds[0], "max_s": meds[-1],
            "median_s": statistics.median(meds),
            "spread_ratio": meds[-1] / max(meds[0], 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--task", default="darcy")
    ap.add_argument("--uq-source", default="het", choices=["het", "cqr"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--check-every", type=int, nargs="+",
                    default=[1, 10, 50, 100])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--precond", nargs="+", default=["continuous"],
                    choices=["continuous", "discrete"],
                    help="FFT preconditioner symbol(s) to time. `discrete` "
                         "matches the 5-point stencil the operator actually "
                         "applies; a preconditioner change cannot alter the "
                         "solution, only the iteration count, so it is the "
                         "same solver and an admissible denominator.")
    ap.add_argument("--fast-apply", action="store_true",
                    help="also time the solver with the face coefficients "
                         "hoisted out of the PCG iteration. Verified "
                         "bit-identical to the default path, so it is the same "
                         "solver and an admissible denominator -- and it can "
                         "only make our own speedup clause harder.")
    ap.add_argument("--sample-sweep", type=int, default=24,
                    help="also time batch 1 on this many DISTINCT samples. "
                         "The per-batch cells above repeat one coefficient "
                         "field, which measures timing noise rather than the "
                         "spread of solve difficulty that actually sets the "
                         "denominator.")
    ap.add_argument("--sweep-trials", type=int, default=2)
    ap.add_argument("--out", default="runs/bench_fair.json")
    args = ap.parse_args()

    if args.task != "darcy":
        ap.error("only darcy has an iterative solver with a sync to amortize; "
                 "the spectral families are single-shot propagators")

    ramp = warmup_device("cuda")
    print(f"clock ramp: {ramp}", flush=True)
    models, ck = load_members([args.ckpt], "cuda")
    if not ck["args"].get("uq"):
        ap.error(f"{args.ckpt} is not a UQ checkpoint")
    model, stats = models[0].eval(), ck["stats"]
    sim = PDE2DSimulator(args.task, device="cuda")
    blob = load_shard(args.root, args.task, "test", 64)

    # Accuracy of the timed model, in the configuration it is timed in.
    mean, _, truth, _ = predict_shard_single(
        model, {"a": blob["a"], "u": blob["u"], "task": args.task}, stats,
        "cuda", autocast=True, sigma_source=args.uq_source)
    surrogate_rel_l2 = float(rel_l2(mean, truth).mean())
    del mean, truth
    torch.cuda.empty_cache()
    print(f"surrogate rel-L2 = {surrogate_rel_l2:.5f}", flush=True)

    res = {"env": env_report("cuda"), "clock_ramp": ramp, "task": args.task,
           "uq_source": args.uq_source, "seed": ck["args"].get("seed"),
           "surrogate_rel_l2": surrogate_rel_l2, "tol": TOL,
           "trials": args.trials, "iters_per_trial": args.iters,
           "forward_passes_per_interval": 1, "batches": {},
           "note": (
               "Surrogate: CUDA-graph replay, gated against eager on a "
               "MUTATED input so a stale captured buffer cannot pass. Solver: "
               "the PCG convergence test amortized over `check_every`. "
               "`ratio_conservative` divides the fastest admissible solver "
               "trial by the slowest surrogate trial.\n\n"
               "THREE THINGS THIS DOES NOT CLAIM, each found by an adversarial "
               "review (`logs/critic_codex_graph.log`) and corrected here:\n"
               "(1) This is NOT 'both sides equally optimized'. The PCG "
               "iteration is still a Python loop launching individual kernels, "
               "and `_darcy_apply` (uqkit/sims/pde2d.py:198) recomputes four "
               "coefficient-face arrays every iteration although `a` is fixed "
               "for the whole solve. Those are real, named, UNMEASURED solver "
               "optimizations. `solver_s_to_erase_100x` below is the latency "
               "an optimized reference solver would have to reach to take the "
               "batch-1 clause back under 100x.\n"
               "(2) `check_every` is the fastest of FOUR convergence-check "
               "strides of this repo's FP64 PCG -- not a globally optimal "
               "reference. A discrete-Laplacian or multigrid preconditioner, "
               "mixed-precision inner iterations with FP64 refinement, or a "
               "sparse direct factorization are all admissible alternatives "
               "and none has been measured.\n"
               "(3) Admissibility is the PRE-CAST FP64 residual, which does "
               "not certify the delivered lower-precision field to `tol`. And "
               "a later convergence check does not in general return a more "
               "converged iterate -- CG's Euclidean residual is not monotone; "
               "what licenses each admission is the explicit final residual "
               "recorded per setting, not that argument.")}

    for B in args.batches:
        a_raw = blob["a"][:B].to("cuda")
        if a_raw.shape[0] < B:
            continue
        st = stats[PARENT.get(args.task, args.task)]
        tid = torch.full((B,), TASK_ID[PARENT.get(args.task, args.task)],
                         device="cuda", dtype=torch.long)
        entry = {"batch": B, "solver": {}, "surrogate": {}}

        a_coef, f_rhs = torch.exp(a_raw[:, 0]), a_raw[:, 1]
        # (check_every, fast_apply). `fast_apply` hoists the face coefficients
        # out of the PCG iteration -- the redundant work an adversarial review
        # named. It is verified bit-identical to the default path, so it is the
        # SAME solver and therefore an admissible denominator. Including it can
        # only make our own clause harder, which is the point.
        combos = [(ce, fa, pc)
                  for pc in args.precond
                  for fa in ([False, True] if args.fast_apply else [False])
                  for ce in args.check_every]
        for ce, fap, pc in combos:
            _, resid = P.solve_darcy(a_coef, f_rhs, tol=TOL, check_every=ce,
                                     fast_apply=fap, precond=pc)
            n_it = P.solve_darcy.last_iters
            t = trials(lambda: P.solve_darcy(a_coef, f_rhs, tol=TOL,
                                             check_every=ce, fast_apply=fap,
                                             precond=pc),
                       args.trials, args.iters)
            t["achieved_residual"] = float(resid)
            t["admissible"] = bool(resid <= TOL)
            t["check_every"] = ce
            t["fast_apply"] = fap
            t["precond"] = pc
            t["pcg_iters"] = int(n_it)
            key = (f"check_every={ce}" + ("+fast_apply" if fap else "")
                   + ("" if pc == "continuous" else f"+{pc}_precond"))
            entry["solver"][key] = t
            print(f"  B={B:<3d} solver {key:<40s} "
                  f"{t['median_s']*1e3:8.2f} ms (min {t['min_s']*1e3:.2f}) "
                  f"{n_it:5d} it resid={resid:.2e} "
                  f"{'OK' if t['admissible'] else 'INADMISSIBLE'}", flush=True)

        eager = make_uq_fn(model, a_raw, st, tid, args.uq_source, autocast=True)
        arms = {"eager": eager, "nograd": no_grad_wrap(eager)}
        with torch.no_grad():
            ref = eager().clone()
        gfn, _ = graph_wrap(eager)
        dev = float((gfn() - ref).abs().max()) / (float(ref.abs().max()) + 1e-12)
        if dev > 1e-5:
            raise RuntimeError(f"graph replay deviates {dev:.3e}; row void")

        # Staleness gate. Comparing one replay against eager on an UNCHANGED
        # input would also pass if `replay()` returned a stale captured buffer
        # and computed nothing -- the review was right that the first check
        # cannot tell those apart. Mutate the captured input in place (same
        # storage, so the graph's fixed addresses still point at it), replay,
        # and require the output to TRACK the new input. A cached tensor fails
        # this; a live graph passes it.
        with torch.no_grad():
            # Restore from a saved copy, NOT by inverting the perturbation.
            # `mul_(1.05).add_(0.01)` followed by `sub_(0.01).div_(1.05)` is
            # not bit-exact in floating point, and the residue scaled with
            # tensor size: this gate rejected its own batch-64 row at 1.4e-3
            # deviation on the restore check. That was an artefact of the gate,
            # not staleness -- and the fix is an exact restore, not a looser
            # tolerance.
            orig = a_raw.clone()
            a_raw.mul_(1.05).add_(0.01)
            live_ref = eager().clone()
            live_out = gfn()
            live_dev = (float((live_out - live_ref).abs().max())
                        / (float(live_ref.abs().max()) + 1e-12))
            moved = (float((live_ref - ref).abs().max())
                     / (float(ref.abs().max()) + 1e-12))
            a_raw.copy_(orig)
            restored = gfn()
            restore_dev = (float((restored - ref).abs().max())
                           / (float(ref.abs().max()) + 1e-12))
            del orig
        if moved < 1e-6:
            raise RuntimeError(
                "the input mutation did not change the eager output, so the "
                "staleness gate proves nothing; pick a larger perturbation")
        if live_dev > 1e-5:
            raise RuntimeError(
                f"graph replay did not track a mutated input (deviation "
                f"{live_dev:.3e} while the output moved {moved:.3e}) -- the "
                f"replay is returning a stale buffer and every graph timing "
                f"below is void")
        if restore_dev > 1e-5:
            raise RuntimeError(
                f"graph replay did not return to the original output after the "
                f"input was restored (deviation {restore_dev:.3e}); row void")
        entry["graph_gate"] = {
            "rel_dev_same_input": dev,
            "rel_dev_after_input_mutation": live_dev,
            "output_movement_from_mutation": moved,
            "rel_dev_after_restore": restore_dev,
            "why": ("a replay that returned a cached tensor would show "
                    "rel_dev_after_input_mutation ~= output_movement_from_"
                    "mutation instead of ~0, so this distinguishes a live "
                    "graph from a stale buffer"),
        }
        arms["graph"] = gfn
        for name, fn in arms.items():
            t = trials(fn, args.trials, args.iters)
            t["rel_l2"] = surrogate_rel_l2
            if name == "graph":
                t["graph_vs_eager_rel_dev"] = dev
            entry["surrogate"][name] = t
            print(f"  B={B:<3d} surrogate {name:<8s} "
                  f"{t['median_s']*1e3:8.3f} ms (max {t['max_s']*1e3:.3f})",
                  flush=True)

        adm = [v for v in entry["solver"].values() if v["admissible"]]
        if adm:
            fastest = min(adm, key=lambda v: v["min_s"])
            entry["fair_denominator"] = {
                "check_every": fastest["check_every"],
                "fast_apply": fastest.get("fast_apply", False),
                "precond": fastest.get("precond", "continuous"),
                "pcg_iters": fastest.get("pcg_iters"),
                "min_s": fastest["min_s"], "median_s": fastest["median_s"],
                "achieved_residual": fastest["achieved_residual"],
                "why": ("fastest solver setting that still meets tol; using a "
                        "slower admissible setting would inflate every ratio "
                        "below"),
            }
            entry["ratios"] = {}
            for name, s in entry["surrogate"].items():
                entry["ratios"][name] = {
                    "ratio_conservative": fastest["min_s"] / max(s["max_s"], 1e-12),
                    "ratio_median": fastest["median_s"] / max(s["median_s"], 1e-12),
                    "ratio_vs_check_every_1_median": (
                        entry["solver"]["check_every=1"]["median_s"]
                        / max(s["median_s"], 1e-12)),
                    "meets_100x_conservative": bool(
                        fastest["min_s"] / max(s["max_s"], 1e-12) >= 100.0),
                    # What an optimized reference solver would have to reach to
                    # take this arm back under the KPI. Named because the
                    # solver-side optimizations are real and unmeasured.
                    "solver_s_to_erase_100x": 100.0 * s["max_s"],
                    "solver_speedup_needed_to_erase_100x": (
                        fastest["min_s"] / max(100.0 * s["max_s"], 1e-12)),
                    "arm_semantics": {
                        "eager": "autograd ENABLED, eager dispatch",
                        "nograd": "autograd disabled, eager dispatch",
                        "graph": "autograd disabled, CUDA-graph replay",
                    }[name],
                    # eager->graph changes autograd AND dispatch. The
                    # dispatch-only comparison is nograd->graph, and the review
                    # was right that calling eager->graph 'only dispatch' was
                    # inaccurate.
                    "dispatch_only_gain_vs_nograd": (
                        entry["surrogate"]["nograd"]["median_s"]
                        / max(s["median_s"], 1e-12)),
                }
                r = entry["ratios"][name]
                print(f"  B={B:<3d} RATIO {name:<8s} "
                      f"conservative {r['ratio_conservative']:8.1f}x  "
                      f"median {r['ratio_median']:8.1f}x  "
                      f"(unfair check_every=1 reading "
                      f"{r['ratio_vs_check_every_1_median']:.1f}x)", flush=True)
        res["batches"][str(B)] = entry
        del a_raw, ref, gfn
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- #
    # Per-sample sweep at batch 1.
    #
    # Every batch-1 cell above times `blob["a"][:1]` -- the SAME first
    # coefficient field, repeated. Repetition characterizes timing noise and
    # says nothing about the spread of Darcy solve difficulty, which is the
    # thing that actually sets the denominator: PCG iteration count depends on
    # the coefficient contrast of the sample being solved. An adversarial
    # review raised this and it is the strongest of its objections, because a
    # 296x measured on one easy or one hard field is not a measurement of the
    # family. This sweeps distinct samples and reports the DISTRIBUTION of the
    # ratio, including how many individual samples clear the KPI.
    # ---------------------------------------------------------------- #
    if args.sample_sweep > 0:
        fd = res["batches"]["1"]["fair_denominator"]
        ce, fap = fd["check_every"], fd.get("fast_apply", False)
        pc = fd.get("precond", "continuous")
        print(f"\nper-sample sweep at batch 1, check_every={ce}, "
              f"fast_apply={fap}, precond={pc}, "
              f"{args.sample_sweep} distinct samples",
              flush=True)
        st = stats[PARENT.get(args.task, args.task)]
        tid = torch.full((1,), TASK_ID[PARENT.get(args.task, args.task)],
                         device="cuda", dtype=torch.long)
        cells = []
        for idx in range(min(args.sample_sweep, len(blob["a"]))):
            a_one = blob["a"][idx:idx + 1].to("cuda")
            a_c, f_r = torch.exp(a_one[:, 0]), a_one[:, 1]
            # Per-sample denominator selection. Picking ONE configuration on
            # the reference sample and applying it to all 24 handicaps the
            # solver on the others: `check_every` rounds the stopping iteration
            # up to a multiple of itself, so a stride tuned to a field needing
            # 800 iterations forces a field needing 350 to run 400. That is
            # exactly what happened -- selecting check_every=100 on sample 0
            # made the easiest fields *slower* and inflated our own ratio. The
            # solver gets its best admissible configuration on every field
            # independently, which can only make our clause harder.
            per_cfg = {}
            for ce_i in args.check_every:
                _, r_i = P.solve_darcy(a_c, f_r, tol=TOL, check_every=ce_i,
                                       fast_apply=fap, precond=pc)
                if r_i > TOL:
                    continue
                it_i = P.solve_darcy.last_iters
                t_i = trials(lambda: P.solve_darcy(a_c, f_r, tol=TOL,
                                                   check_every=ce_i,
                                                   fast_apply=fap, precond=pc),
                             args.sweep_trials, args.iters)
                t_i["achieved_residual"] = float(r_i)
                t_i["pcg_iters"] = int(it_i)
                t_i["check_every"] = ce_i
                per_cfg[f"check_every={ce_i}"] = t_i
            if not per_cfg:
                print(f"  sample {idx:3d} no admissible solver config; "
                      f"excluded", flush=True)
                del a_one
                torch.cuda.empty_cache()
                continue
            best_key = min(per_cfg, key=lambda k: per_cfg[k]["min_s"])
            ts = per_cfg[best_key]
            resid, n_it = ts["achieved_residual"], ts["pcg_iters"]
            ce_used = ts["check_every"]
            fn1 = make_uq_fn(model, a_one, st, tid, args.uq_source,
                             autocast=True)
            tg = trials(no_grad_wrap(fn1), args.sweep_trials, args.iters)
            gf, _ = graph_wrap(fn1)
            tgr = trials(gf, args.sweep_trials, args.iters)
            cell = {
                "sample_index": idx,
                "check_every_selected": ce_used,
                "solver_configs_timed": {
                    k: {"min_s": v["min_s"], "median_s": v["median_s"],
                        "pcg_iters": v["pcg_iters"],
                        "achieved_residual": v["achieved_residual"]}
                    for k, v in per_cfg.items()},
                "solver_median_s": ts["median_s"], "solver_min_s": ts["min_s"],
                "achieved_residual": float(resid),
                "admissible": bool(resid <= TOL),
                "pcg_iters": int(n_it),
                "nograd_median_s": tg["median_s"], "nograd_max_s": tg["max_s"],
                "graph_median_s": tgr["median_s"], "graph_max_s": tgr["max_s"],
                "ratio_graph_conservative": ts["min_s"] / max(tgr["max_s"], 1e-12),
                "ratio_graph_median": ts["median_s"] / max(tgr["median_s"], 1e-12),
                "ratio_nograd_conservative": ts["min_s"] / max(tg["max_s"], 1e-12),
            }
            cells.append(cell)
            print(f"  sample {idx:3d} solver {ts['median_s']*1e3:8.2f} ms "
                  f"(min {ts['min_s']*1e3:.2f}, best {best_key}, "
                  f"{n_it} it) resid {resid:.1e} "
                  f"{'OK' if cell['admissible'] else 'INADM'} | graph "
                  f"{tgr['median_s']*1e3:.3f} ms | ratio "
                  f"{cell['ratio_graph_conservative']:8.1f}x", flush=True)
            del a_one, gf
            torch.cuda.empty_cache()

        adm = [c for c in cells if c["admissible"]]
        rg = sorted(c["ratio_graph_conservative"] for c in adm)
        rn = sorted(c["ratio_nograd_conservative"] for c in adm)
        sv = sorted(c["solver_median_s"] for c in adm)
        res["sample_sweep"] = {
            "check_every": "selected per sample from "
                             + str(args.check_every),
            "fast_apply": fap, "precond": pc, "n_samples": len(cells),
            "n_admissible": len(adm), "trials_per_cell": args.sweep_trials,
            "cells": cells,
            "solver_median_s": {"min": sv[0], "max": sv[-1],
                                "median": statistics.median(sv),
                                "spread_ratio": sv[-1] / max(sv[0], 1e-12)},
            "ratio_graph_conservative": {
                "min": rg[0], "max": rg[-1], "median": statistics.median(rg),
                "n_ge_100x": sum(1 for x in rg if x >= 100.0), "n": len(rg)},
            "ratio_nograd_conservative": {
                "min": rn[0], "max": rn[-1], "median": statistics.median(rn),
                "n_ge_100x": sum(1 for x in rn if x >= 100.0), "n": len(rn)},
            "note": ("Distinct coefficient fields, one per cell, so the "
                     "denominator is a distribution over solve difficulty "
                     "rather than one repeated sample. `n_ge_100x` is the "
                     "honest form of the clause at batch 1: the fraction of "
                     "individual problems that clear it, not the ratio of one "
                     "field."),
        }
        print(f"\n  graph ratio over {len(rg)} samples: "
              f"min {rg[0]:.1f}x  median {statistics.median(rg):.1f}x  "
              f"max {rg[-1]:.1f}x  >=100x on {sum(1 for x in rg if x >= 100)}"
              f"/{len(rg)}", flush=True)
        print(f"  solver spread across samples: "
              f"{sv[0]*1e3:.1f}-{sv[-1]*1e3:.1f} ms "
              f"({sv[-1]/sv[0]:.2f}x)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
