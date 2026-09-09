"""Measure the residual check's noise floor: ||L u_exact - f|| / ||f||.

    CUDA_VISIBLE_DEVICES=3 python scripts/measure_residual_floor.py

The residual detector's whole claim is that a large residual means the
surrogate is wrong. That claim needs a floor: how large is the residual when
the answer is *exactly right*? Applying L amplifies the round-off already in u
by the operator's symbol, so the floor is a property of the differential order
and the working precision, and it decides which families the detector may be
used on at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402

TASKS = ["poisson", "helmholtz", "frac_s0p5", "frac_s1p5", "darcy",
         "biharmonic", "frac_s2p5", "frac_s3"]
SYMBOL_ORDER = {"poisson": 2, "helmholtz": 2, "frac_s0p5": 1, "frac_s1p5": 3,
                "darcy": 2, "biharmonic": 4, "frac_s2p5": 5, "frac_s3": 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/residual_floor.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    res = {"device": dev, "solution_dtype": "float32",
           "apply_dtype": "float64", "n_samples": a.n, "grids": {}}
    for N in (32, 64, 128):
        row = {}
        for t in TASKS:
            sim = PDE2DSimulator(t, device=dev, N=N)
            x = sim.sample_inputs(a.n, seed=0, N=N)
            r = sim.residual(x, sim.solve(x))
            rel = (r.flatten(1).norm(dim=1)
                   / sim.rhs(x).flatten(1).norm(dim=1))
            row[t] = {"order": SYMBOL_ORDER[t], "mean": float(rel.mean()),
                      "max": float(rel.max())}
            del x, r
            torch.cuda.empty_cache()
        res["grids"][str(N)] = row
        print(f"N={N}: " + "  ".join(f"{k}={v['mean']:.1e}" for k, v in row.items()),
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
