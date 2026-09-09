"""Generate every shard the kit measures on: in-distribution and shifted.

    CUDA_VISIBLE_DEVICES=2 python scripts/gen_data.py --root data

Splits, and why there are three of them rather than two:

* `train` -- what the ensemble members fit.
* `cal`   -- what the conformal quantile is computed on. Disjoint from train by
             seed; if it were not, the residuals would be optimistically small
             and the interval would undercover on anything real.
* `test`  -- what coverage is measured on. Disjoint from both.

The OOD shards are generated once here and never seen by training, by
calibration, or by any detector fit, which is what lets the AUROC be quoted as
a number about a detector rather than about a tuning loop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.sims import pde2d as P  # noqa: E402

IN_TASKS = list(P.PRETRAIN_TASKS)                       # 5 trained families
# seeds are offset per split so no sample is ever shared between them
SPLIT_SEED = {"train": 1000, "cal": 5000, "test": 9000}

# --- the OOD suite, grouped by *kind of shift*, because the KPI needs each
# --- reported separately and an average over these would be meaningless.
OOD_SUITE = {
    # same operator, shifted input distribution
    "input_shift": [f"{p}_{s}" for p in IN_TASKS for s in P.SHIFT_SPECS],
    # same family, parameter outside the trained range
    "param_oor": ["darcy_c3", "frac_s3", "frac_s0p25", "frac_s0p5"],
    # a governing equation the surrogate has never been trained on, presented
    # under a task id it does have (`PARENT`) -- the deployment failure where
    # the process changed and nobody told the model
    "unseen_operator": ["biharmonic", "navier_stokes", "ns_T0p25"],
    # a graded version of the roughness axis, so the weighted-conformal result
    # is a threshold rather than a cliff -- see pde2d.GRADED_DALPHA
    "graded_rough": list(P.GRADED_TASKS),
}
RES_TASKS = IN_TASKS                                    # resolution shift


def save(root, task, split, N, a, u, extra):
    p = Path(root) / f"{task}_{split}_N{N}.pt"
    torch.save({"a": a, "u": u, "task": task, "N": N, "split": split,
                "parent": P.PARENT.get(task, task), **extra}, p)
    return p


def gen(root, task, split, n, N, seed, device, force):
    p = Path(root) / f"{task}_{split}_N{N}.pt"
    if p.exists() and not force:
        print(f"  skip {p.name} (exists)", flush=True)
        return
    t0 = time.time()
    a, u, extra = P.generate(task, n, N=N, seed=seed, device=device)
    save(root, task, split, N, a, u, extra)
    print(f"  {p.name}: {tuple(a.shape)} -> {tuple(u.shape)}  "
          f"{time.time() - t0:.1f}s {extra.get('cg_residual', '')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--train", type=int, default=4096)
    ap.add_argument("--cal", type=int, default=1024)
    ap.add_argument("--test", type=int, default=1024)
    ap.add_argument("--ood", type=int, default=512)
    ap.add_argument("--hr", type=int, default=256, help="samples at 128^2 / 256^2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["all", "indist", "ood", "res"])
    a = ap.parse_args()
    Path(a.root).mkdir(parents=True, exist_ok=True)

    if a.stage in ("all", "indist"):
        print("in-distribution shards", flush=True)
        for t in IN_TASKS:
            for split, n in (("train", a.train), ("cal", a.cal), ("test", a.test)):
                gen(a.root, t, split, n, 64, SPLIT_SEED[split] + P.TASK_ID[t],
                    a.device, a.force)

    if a.stage in ("all", "ood"):
        for kind, tasks in OOD_SUITE.items():
            print(f"OOD shards -- {kind}", flush=True)
            for i, t in enumerate(tasks):
                gen(a.root, t, "ood", a.ood, 64, 20000 + 37 * i, a.device, a.force)

    if a.stage in ("all", "res"):
        print("resolution-shift shards (the model trains only at 64^2)", flush=True)
        for N in (128, 256):
            for t in RES_TASKS:
                gen(a.root, t, "ood", a.hr, N, 30000 + N + P.TASK_ID[t],
                    a.device, a.force)

    man = {"in_tasks": IN_TASKS, "ood_suite": OOD_SUITE,
           "res_tasks": RES_TASKS, "split_seed": SPLIT_SEED,
           "n": {"train": a.train, "cal": a.cal, "test": a.test, "ood": a.ood,
                 "hr": a.hr}}
    Path(a.root, "manifest.json").write_text(json.dumps(man, indent=2))
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
