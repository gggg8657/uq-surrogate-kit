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
from uqkit.bench import env_report, timeit, warmup_device  # noqa: E402
from uqkit.metrics import rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.equivar import rescale_inputs, sample_scale  # noqa: E402
from uqkit.sims.fno2d_uq import sigma_from, split_heads  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)


def make_surrogate_fn(models, a_raw, st, tid, k, residual=None, autocast=True):
    """Closure timing the first `k` members, optionally plus the residual check.

    **Normalization and de-normalization are inside the timed region.** They are
    part of what a deployment runs, and the reference solver has no equivalent
    preprocessing step, so hoisting them out of the loop is a small subsidy to
    the surrogate. They are cheap; the point is that the comparison should not
    need the reader to check whether they were counted.
    """
    ms = models[:k]
    a_mean, a_std = st["a_mean"].to(a_raw.device), st["a_std"].to(a_raw.device)
    u_mean, u_std = st["u_mean"].to(a_raw.device), st["u_std"].to(a_raw.device)

    def fn():
        if equivar_parent is None:
            a_in, s_eq = a_raw, None
        else:
            s_eq = sample_scale(a_raw, equivar_parent, equivar_ref)
            a_in = rescale_inputs(a_raw, equivar_parent, s_eq)
        a_norm = (a_in - a_mean) / a_std
        preds = []
        for m in ms:
            if autocast and a_raw.is_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(m(a_norm, tid).float())
            else:
                preds.append(m(a_norm, tid))
        p = torch.stack(preds) * u_std + u_mean
        mean = p.mean(0)
        _ = p.std(0, unbiased=True)
        if residual is not None:
            _ = residual(a_raw, mean)
        return mean
    return fn


def make_uq_fn(model, a_raw, st, tid, source, q_hat=1.0, residual=None,
               autocast=True, equivar_parent=None, equivar_ref=None):
    """Closure timing ONE forward pass that emits the mean *and* the interval.

    The ensemble path pays M forward passes for a quantity this head produces in
    one, which is why `runs/bench.json` could put 108.5x and "has an interval"
    on different rows and never on the same one. Everything the deployment runs
    is inside the timed region, on the same terms as `make_surrogate_fn`:
    normalization, the forward pass, splitting the heads, de-normalizing the
    mean, scaling sigma, and **forming the conformal band** `mean +- q_hat *
    sigma`. `q_hat` is a scalar fitted offline on the calibration split -- the
    same offline step the ensemble path also needs -- so its *value* does not
    change the timing and only the multiply is counted here.

    sigma is scaled by `u_std` and not shifted by `u_mean`: it is a spread, not
    a field value. Same rule as `predict_shard_single`.

    `equivar_parent`/`equivar_ref` put the H17 scale wrapper **inside the timed
    region and inside the captured graph**: the per-sample scale reduction, the
    input rescale and the output rescale are all ordinary device tensor ops, so
    they capture like everything else. This is what makes the wrapper's effect
    on the clause-2 reading measurable rather than extrapolated -- the earlier
    figure came from the eager `predict_shard_single` path and could not be
    multiplied into a CUDA-graph number.
    """
    a_mean, a_std = st["a_mean"].to(a_raw.device), st["a_std"].to(a_raw.device)
    u_mean, u_std = st["u_mean"].to(a_raw.device), st["u_std"].to(a_raw.device)

    def fn():
        if equivar_parent is None:
            a_in, s_eq = a_raw, None
        else:
            s_eq = sample_scale(a_raw, equivar_parent, equivar_ref)
            a_in = rescale_inputs(a_raw, equivar_parent, s_eq)
        a_norm = (a_in - a_mean) / a_std
        if autocast and a_raw.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(a_norm, tid).float()
        else:
            y = model(a_norm, tid)
        mean = split_heads(y)[0] * u_std + u_mean
        sigma = sigma_from(y, source) * u_std
        if s_eq is not None:
            v = s_eq.view(-1, *([1] * (mean.dim() - 1)))
            mean = mean * v
            sigma = sigma * v
        _ = mean - q_hat * sigma
        _ = mean + q_hat * sigma
        if residual is not None:
            _ = residual(a_raw, mean)
        return mean
    return fn


def no_grad_wrap(fn):
    """Same closure, autograd tracking off.

    Every speedup this repo has published timed the surrogate with autograd
    tracking ON while `PDE2DSimulator.solve` carries `@torch.no_grad()`
    (`uqkit/sims/pde2d_sim.py:42`). That charges the surrogate for building a
    graph nobody uses and does not charge the solver for it -- a bias against
    our own claim, but an unfair comparison all the same. This arm measures how
    much of the batch-1 cost that asymmetry was, instead of leaving it as a
    silent margin. The eager arm is left exactly as it was so no published
    number changes underneath.
    """
    def wrapped():
        with torch.no_grad():
            return fn()
    return wrapped


def graph_wrap(fn, n_warmup=3):
    """Capture `fn` into a CUDA graph and return a replay closure.

    `fn` closes over an input tensor that is already resident on the device and
    never mutated, and every shape is static, which is exactly the case CUDA
    graphs are for. Replay issues the whole kernel sequence with one launch, so
    it isolates per-kernel launch latency -- the thing the batch-1 rows appear
    to be spending their time on (64x the arithmetic for 1.32x the wall clock
    between b=1 and b=64).

    This is an execution mode, not a different model: same weights, same
    kernels, same tensors. The caller asserts the output is unchanged; if it is
    not, the row is void and the runner refuses to emit it.
    """
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.no_grad():
            for _ in range(n_warmup):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            static_out = fn()

    def replay():
        g.replay()
        return static_out
    return replay, static_out


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
    ap.add_argument("--uq-source", default=None, choices=["het", "cqr"],
                    help="time the single-network UQ model instead of the "
                         "ensemble: ONE forward pass that emits the mean and "
                         "the interval. Requires exactly one UQ checkpoint. "
                         "The solver denominator, batches, iteration counts "
                         "and clock ramp are unchanged, so the rows are "
                         "directly comparable to an ensemble run.")
    ap.add_argument("--exec-modes", action="store_true",
                    help="additionally time each CUDA variant with autograd "
                         "off and under CUDA-graph replay. Same weights and "
                         "same solver denominator; only kernel dispatch "
                         "changes. The eager row is still emitted unchanged, "
                         "and a graph row whose output deviates from eager is "
                         "rejected rather than reported.")
    ap.add_argument("--cpu-acc-n", type=int, default=128,
                    help="samples used for the CPU accuracy check")
    args = ap.parse_args()

    ramp = warmup_device("cuda")
    print(f"clock ramp: {ramp}", flush=True)
    tasks = args.tasks.split(",")
    models, ck = load_members(args.ckpts, "cuda")
    stats = ck["stats"]
    M = len(models)
    uq = args.uq_source
    if uq:
        if M != 1:
            ap.error("--uq-source times one network; pass one checkpoint")
        if not ck["args"].get("uq"):
            ap.error(f"{args.ckpts[0]} is not a UQ checkpoint")
        res_uq_note = (
            "Timed model is FNO2dUQ, ONE forward pass emitting mean, log-sigma "
            "and two residual quantiles. The 108.5x M=1 row in "
            "runs/members.json was measured on a 1-channel ensemble MEMBER and "
            "does NOT transfer to this model -- the extra 1x1 projection is a "
            "cost to be measured, not assumed. That is what this run measures.")
    res = {"env": env_report("cuda"), "n_members": M, "rows": [],
           "clock_ramp": ramp,
           "note": ("rel_l2 is the ensemble mean's error on that family's test "
                    "split, measured in the same bf16-autocast configuration "
                    "the GPU rows are timed in. Normalization and "
                    "de-normalization are inside the timed surrogate region; "
                    "host-device transfer is outside it on both sides, since "
                    "both operate on tensors already resident on the device. "
                    "The solver here is the one the corpus was generated with; "
                    "`bench_isoaccuracy.py` reports the ratio against the "
                    "cheapest solver setting that matches the surrogate's own "
                    "accuracy, which is the smaller and more honest number.")}

    # Accuracy first: a speedup row without it is not emitted -- and it is
    # measured **per variant and per device**, not once. A single member is
    # ~M times faster than the ensemble and also less accurate, so pinning the
    # ensemble's rel-L2 next to the single-member speedup (which an earlier
    # version did) advertises a ratio the quoted accuracy never achieved. The
    # CPU rows run fp32 without autocast, so they get their own number too.
    acc = {}
    for t in tasks:
        blob = load_shard(args.root, t, "test", 64)
        for dev_, ac_ in (("cuda", True), ("cpu", False)):
            if dev_ == "cpu" and args.no_cpu:
                continue
            n = len(blob["a"]) if dev_ == "cuda" else args.cpu_acc_n
            sub = {"a": blob["a"][:n], "u": blob["u"][:n], "task": t}
            ms = [m.to(dev_).eval() for m in models]
            pairs = ((1, "uq_single"),) if uq else ((1, "single"), (M, "ensemble"))
            for k, name in pairs:
                if uq:
                    mean, _, truth, _ = predict_shard_single(
                        ms[0], sub, stats, dev_, autocast=ac_,
                        sigma_source=args.uq_source)
                else:
                    mean, _, truth, _ = predict_shard(ms[:k], sub, stats, dev_,
                                                      autocast=ac_)
                acc[f"{t}|{dev_}|{name}"] = float(rel_l2(mean, truth).mean())
                del mean, truth
        [m.to("cuda") for m in models]
        torch.cuda.empty_cache()
    res["accuracy"] = acc
    if uq:
        res["uq_source"] = args.uq_source
        res["forward_passes_per_interval"] = 1
        res["uq_note"] = res_uq_note
        res["seed"] = ck["args"].get("seed")
    res["accuracy_key"] = "task|device|variant; cuda = bf16 autocast, cpu = fp32"
    print("rel-L2:", {k: round(v, 5) for k, v in acc.items()
                      if k.endswith("|cuda|ensemble")}, flush=True)

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
                tid = torch.full((B,), TASK_ID[PARENT.get(t, t)],
                                 device=device, dtype=torch.long)
                sol = timeit(lambda: sim.solve(a_raw), 3, n_iter, device)
                row = {"task": t, "device": device, "batch": B,
                       "trained": trained,
                       "solver_s": sol["median_s"],
                       "solver_s_per_sample": sol["median_s"] / B,
                       "solver_accuracy": sim.solver_accuracy(),
                       "threads": torch.get_num_threads(),
                       "solver_timing": sol, "surrogate": {}}
                has_res = sim.residual(a_raw, sim.solve(a_raw)) is not None
                if uq:
                    variants = {"uq_single": (1, False)}
                    if has_res:
                        variants["uq_single+residual"] = (1, True)
                else:
                    variants = {"single": (1, False), "ensemble": (M, False)}
                    if has_res:
                        variants["ensemble+residual"] = (M, True)
                for name, (k, use_res) in variants.items():
                    res_fn = sim.residual if use_res else None
                    if uq:
                        fn = make_uq_fn(models_d[0], a_raw, st, tid,
                                        args.uq_source, residual=res_fn,
                                        autocast=(device == "cuda"))
                        variant_acc = acc.get(f"{t}|{device}|uq_single")
                    else:
                        fn = make_surrogate_fn(models_d, a_raw, st, tid, k,
                                               residual=res_fn)
                        variant_acc = acc.get(
                            f"{t}|{device}|{'single' if k == 1 else 'ensemble'}")
                    tm = timeit(fn, 3, n_iter, device)
                    row["surrogate"][name] = {
                        "s": tm["median_s"], "s_per_sample": tm["median_s"] / B,
                        "speedup": sol["median_s"] / max(tm["median_s"], 1e-12),
                        "rel_l2": variant_acc, "timing": tm,
                        "exec_mode": "eager", "grad": True}

                    # Execution-mode arms (H8). Same weights, same kernels, same
                    # solver denominator -- what changes is only how the kernels
                    # are dispatched. Both are reported next to the eager row,
                    # never instead of it.
                    if not args.exec_modes or device != "cuda":
                        continue
                    with torch.no_grad():
                        ref = fn().clone()
                    ng = no_grad_wrap(fn)
                    tm_ng = timeit(ng, 3, n_iter, device)
                    row["surrogate"][name + "+nograd"] = {
                        "s": tm_ng["median_s"],
                        "s_per_sample": tm_ng["median_s"] / B,
                        "speedup": sol["median_s"] / max(tm_ng["median_s"], 1e-12),
                        "rel_l2": variant_acc, "timing": tm_ng,
                        "exec_mode": "eager", "grad": False}
                    try:
                        gfn, _ = graph_wrap(fn)
                        out = gfn()
                        # Hard gate: a graph replay that does not reproduce the
                        # eager output is a different model and the row is void.
                        dev_max = float((out - ref).abs().max())
                        scale = float(ref.abs().max()) + 1e-12
                        rel_dev = dev_max / scale
                        if rel_dev > 1e-5:
                            raise RuntimeError(
                                f"graph replay deviates from eager by "
                                f"{rel_dev:.3e} relative -- row rejected")
                        tm_g = timeit(gfn, 3, n_iter, device)
                        row["surrogate"][name + "+graph"] = {
                            "s": tm_g["median_s"],
                            "s_per_sample": tm_g["median_s"] / B,
                            "speedup": sol["median_s"] / max(tm_g["median_s"], 1e-12),
                            "rel_l2": variant_acc, "timing": tm_g,
                            "exec_mode": "cuda_graph", "grad": False,
                            "graph_vs_eager_rel_dev": rel_dev}
                        del gfn
                    except Exception as e:                       # noqa: BLE001
                        # Capture can legitimately fail (a solver residual that
                        # allocates, a non-capturable op). Record why; do not
                        # emit a number.
                        row["surrogate"][name + "+graph"] = {
                            "s": None, "speedup": None, "rel_l2": variant_acc,
                            "exec_mode": "cuda_graph",
                            "capture_failed": f"{type(e).__name__}: {e}"[:300]}
                        print(f"  graph capture failed for {name}: "
                              f"{type(e).__name__}: {e}"[:200], flush=True)
                    del ref
                    torch.cuda.empty_cache()
                res["rows"].append(row)
                sp = {k: round(v["speedup"], 1) for k, v in row["surrogate"].items()}
                print(f"{t:14s} {device:4s} B={B:<3d} solver "
                      f"{sol['median_s']*1e3:9.3f} ms  speedup {sp}", flush=True)
                del a_raw
                torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
