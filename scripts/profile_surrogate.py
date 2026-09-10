"""Where does the batch-1 surrogate latency actually go?

Clause 2 now fails at 21/24 fields (worst 94.0x, `runs/bench_fair.json`) after
three rounds of fixing the *reference* solver. `critique_log.md` H11 ends by
naming the surrogate side as the only remaining source of the missing factor.
Before changing the surrogate, measure it: this script attributes the timed
closure's wall clock to its parts, and counts the CUDA kernels a single replay
issues, so the next change is aimed at a measured cost rather than a guessed
one.

Nothing here is a KPI number. It writes `runs/profile_surrogate.json`.

    CUDA_VISIBLE_DEVICES=2 python scripts/profile_surrogate.py \
        --ckpt runs/u0/best.pt --out runs/profile_surrogate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.bench import env_report, timeit, warmup_device  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.predict import load_members, load_shard  # noqa: E402
from uqkit.sims.fno2d_uq import sigma_from, split_heads  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_speedup import graph_wrap, make_uq_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--task", default="darcy")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="runs/profile_surrogate.json")
    args = ap.parse_args()

    ramp = warmup_device("cuda")
    models, ck = load_members([args.ckpt], "cuda")
    model = models[0].eval()
    blob = load_shard(args.root, args.task, "test", 64)
    parent = PARENT.get(args.task, args.task)
    stats = ck["stats"][parent]
    a_raw = blob["a"][: args.batch].cuda()
    tid = torch.full((args.batch,), TASK_ID[parent], device="cuda",
                     dtype=torch.long)

    res = {"env": env_report("cuda"), "clock_ramp": ramp, "task": args.task,
           "batch": args.batch, "ckpt": args.ckpt,
           "model": {k: ck["args"].get(k) for k in
                     ("width", "modes", "layers", "out_ch")},
           "note": ("Diagnostic only -- no KPI number is derived here. "
                    "`parts` times nested prefixes of the deployed closure, "
                    "each under CUDA-graph replay, so the differences are "
                    "attributable. `kernels` aggregates one profiled replay.")}

    # ---- part times: nested prefixes of the deployed closure --------------
    a_mean, a_std = stats["a_mean"].cuda(), stats["a_std"].cuda()
    u_mean, u_std = stats["u_mean"].cuda(), stats["u_std"].cuda()
    a_norm = (a_raw - a_mean) / a_std

    def f_norm():
        return (a_raw - a_mean) / a_std

    def f_lift():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            B, _, H, W = a_norm.shape
            emb = model.task_emb(tid)[:, :, None, None].expand(-1, -1, H, W)
            x = torch.cat([a_norm, model.coords(B, H, W, a_norm.device,
                                                a_norm.dtype),
                           emb.to(a_norm.dtype)], dim=1)
            return model.lift(x)

    def f_trunk():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.trunk(a_norm, tid)

    def f_fwd():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(a_norm, tid).float()

    full = make_uq_fn(model, a_raw, stats, tid, "het", q_hat=1.0)

    # spectral-only: one block's spectral conv, and all four, on a trunk-width
    # activation. Isolates the einsum path from everything around it.
    x_w = torch.randn(args.batch, model.width, 64, 64, device="cuda")

    def f_spec1():
        return model.blocks[0].spectral(x_w)

    def f_spec_all():
        y = x_w
        for b in model.blocks:
            y = b.spectral(y)
        return y

    def f_blocks_all():
        y = x_w
        for b in model.blocks:
            y = b(y)
        return y

    parts = {}
    for name, fn in [("normalize", f_norm), ("lift", f_lift),
                     ("trunk", f_trunk), ("forward", f_fwd),
                     ("deployed_closure", full),
                     ("spectral_x1", f_spec1), ("spectral_x4", f_spec_all),
                     ("blocks_x4", f_blocks_all)]:
        rep, _ = graph_wrap(fn)
        t = timeit(rep, 5, args.iters, "cuda")
        parts[name] = {"median_s": t["median_s"], "min_s": t["min_s"],
                       "p75_s": t.get("p75_s")}
        print(f"{name:20s} {t['median_s']*1e3:8.4f} ms", flush=True)
    res["parts"] = parts

    # ---- kernel census on one eager forward -------------------------------
    # Profiled eagerly: a captured graph reports as one node, which hides the
    # thing we are trying to count.
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        for _ in range(5):
            full()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                     ) as prof:
            for _ in range(10):
                full()
            torch.cuda.synchronize()

    agg = defaultdict(lambda: {"n": 0, "cuda_us": 0.0})
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0):
            agg[e.key]["n"] += e.count
            agg[e.key]["cuda_us"] += float(e.self_device_time_total)
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["cuda_us"])
    total_us = sum(v["cuda_us"] for _, v in rows)
    res["kernels"] = {
        "n_forward_calls_profiled": 10,
        "total_self_cuda_us": total_us,
        "total_self_cuda_us_per_call": total_us / 10,
        "n_launches_per_call": sum(v["n"] for _, v in rows) / 10,
        "top": [{"op": k, "n_per_call": v["n"] / 10,
                 "self_cuda_us_per_call": v["cuda_us"] / 10,
                 "frac": v["cuda_us"] / max(total_us, 1e-9)}
                for k, v in rows[:25]],
    }
    print(f"\nlaunches/call {res['kernels']['n_launches_per_call']:.1f}  "
          f"self-CUDA {res['kernels']['total_self_cuda_us_per_call']:.1f} us")
    for r in res["kernels"]["top"][:12]:
        print(f"  {r['op'][:48]:48s} n={r['n_per_call']:5.1f} "
              f"{r['self_cuda_us_per_call']:8.1f} us  {r['frac']*100:5.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
