"""The ensemble-size trade-off: what does each member cost, and what does it buy?

    CUDA_VISIBLE_DEVICES=3 python scripts/ablate_members.py --ckpts runs/m*/best.pt

The speedup clause and the uncertainty clauses pull against each other through
exactly one knob. M members cost M forward passes, and the KPI wants >=100x
*and* a calibrated interval *and* an OOD score -- but the interval and the score
are both computed from the spread, which does not exist at M=1.

So this sweeps M and reports, on the same axis: the Darcy speedup, the
in-distribution coverage, the spread-error correlation, and the shift AUROC on
a shift the spread can actually see. If 100x is only reachable at an M that
cannot produce an interval, that is the answer to the KPI, and it is a
structural fact about the design rather than a training deficiency.

Members are taken in checkpoint order (m0, m1, ...), which is arbitrary but
fixed; the M=2 result therefore has whatever luck two particular seeds carry.
With 5 members there is no room for 8 seeds per arm, so this is a **screen, not
a verdict**, and the spread across the C(5,M) subsets is reported for each M so
the reader can see how much of the trend is subset luck.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.bench import env_report, timeit  # noqa: E402
from uqkit.conformal import SplitConformal, get_score  # noqa: E402
from uqkit.metrics import binom_ci, pearson, rel_l2  # noqa: E402
from uqkit.ood import shift_auroc, spread_score  # noqa: E402
from uqkit.sims import pde2d as P  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402

SHIFT_TASK = "poisson_dam1"        # a shift the spread demonstrably sees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/members.json")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]
    M = len(models)
    fn, _ = get_score("field_max")

    blob_c = load_shard(args.root, "poisson", "cal", 64)
    blob_t = load_shard(args.root, "poisson", "test", 64)
    blob_o = load_shard(args.root, SHIFT_TASK, "ood", 64)
    blob_d = load_shard(args.root, "darcy", "test", 64)

    # timing setup for the Darcy speedup, identical to bench_speedup.py
    B = args.batch
    a_d = blob_d["a"][:B].to(args.device)
    coef, f = torch.exp(a_d[:, 0]), a_d[:, 1]
    st_d = stats["darcy"]
    tid_d = torch.full((B,), TASK_ID["darcy"], device=args.device,
                       dtype=torch.long)
    t_solver = timeit(lambda: P.solve_darcy(coef, f), 3, args.iters, args.device)

    res = {"env": env_report(args.device), "batch": B, "shift_task": SHIFT_TASK,
           "solver_s": t_solver["median_s"], "n_members_available": M,
           "note": ("screen, not verdict: 5 checkpoints cannot give 8 seeds per "
                    "arm, so every row also reports the spread over the C(5,M) "
                    "member subsets"),
           "rows": []}

    for k in range(1, M + 1):
        subsets = list(itertools.combinations(range(M), k))
        if len(subsets) > 10:
            subsets = subsets[:10]
        per_subset = []
        for sub in subsets:
            ms = [models[i] for i in sub]
            mc, sc, tc, _ = predict_shard(ms, blob_c, stats, args.device)
            mt, stt, tt, _ = predict_shard(ms, blob_t, stats, args.device)
            mo, so, to, _ = predict_shard(ms, blob_o, stats, args.device)
            err_t = rel_l2(mt, tt)
            row = {"members": list(sub), "rel_l2": float(err_t.mean())}
            if k >= 2:
                med = float(sc.median())
                cal = SplitConformal(0.1).fit(fn(mc, sc, tc, med=med))
                cov = cal.covered(fn(mt, stt, tt, med=med))
                row["coverage"] = float(cov.mean())
                row["spread_error_pearson"] = pearson(
                    spread_score(stt, mt).cpu(), err_t.cpu())
                row["shift_auroc_spread"] = shift_auroc(
                    spread_score(stt, mt).cpu().numpy(),
                    spread_score(so, mo).cpu().numpy())
            else:
                row["coverage"] = None
                row["spread_error_pearson"] = None
                row["shift_auroc_spread"] = None
            per_subset.append(row)
            del mc, sc, tc, mt, stt, tt, mo, so, to
            torch.cuda.empty_cache()

        a_norm = (a_d - st_d["a_mean"].to(args.device)) / st_d["a_std"].to(args.device)
        ms = models[:k]

        def sur():
            preds = []
            for m in ms:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(m(a_norm, tid_d).float())
            p = torch.stack(preds) * st_d["u_std"].to(args.device) \
                + st_d["u_mean"].to(args.device)
            mean = p.mean(0)
            if k >= 2:
                _ = p.std(0, unbiased=True)
            return mean
        t_sur = timeit(sur, 3, args.iters, args.device)

        def agg(key):
            v = [r[key] for r in per_subset if r[key] is not None]
            if not v:
                return None
            return {"mean": float(np.mean(v)), "min": float(np.min(v)),
                    "max": float(np.max(v)), "n_subsets": len(v)}

        res["rows"].append({
            "M": k, "n_subsets_scored": len(subsets),
            "surrogate_s": t_sur["median_s"],
            "darcy_speedup": t_solver["median_s"] / t_sur["median_s"],
            "rel_l2": agg("rel_l2"), "coverage": agg("coverage"),
            "spread_error_pearson": agg("spread_error_pearson"),
            "shift_auroc_spread": agg("shift_auroc_spread"),
            "has_uncertainty": k >= 2})
        r = res["rows"][-1]
        cov = "--" if not r["coverage"] else f"{100*r['coverage']['mean']:.1f}%"
        rho = "--" if not r["spread_error_pearson"] else \
            f"{r['spread_error_pearson']['mean']:+.3f}"
        print(f"M={k}  speedup {r['darcy_speedup']:7.1f}x  "
              f"rel-L2 {r['rel_l2']['mean']:.5f}  coverage {cov}  "
              f"corr(spread,err) {rho}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
