"""The thread-count table `uqkit/scale.py` pins its choice of 4 threads to.

    ~/miniforge3/envs/pdeno/bin/python scripts/bench_scale_fit.py \
        --out runs/scale_fit_threads.json

This exists because a docstring in `uqkit/scale.py` once claimed "5.4 s at the
96-thread default" for a fit size it had not been measured at -- the number was
arithmetic on a 50-step timing taken at a different row count. Rather than
delete the claim, measure it: the pinball fit is a deterministic amount of work
at a stated size, so the table is reproducible and the choice of 4 is visible
rather than asserted.

CPU only, so it does not touch the GPU lease.
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
from uqkit.scale import QuantileScale  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/scale_fit_threads.json")
    ap.add_argument("--rows", type=int, nargs="+", default=[7360, 21760],
                    help="the two fit sizes H15 actually uses: one "
                         "leave-one-mechanism-out fold and the full suite")
    ap.add_argument("--feats", type=int, default=44)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    counts = sorted({1, 2, 4, 8, 16, default_threads})
    rng = np.random.default_rng(0)
    res = {"steps": args.steps, "feats": args.feats, "repeats": args.repeats,
           "default_threads": default_threads,
           "machine": platform.processor() or platform.machine(),
           "torch": torch.__version__,
           "note": ("wall-clock of QuantileScale.fit at a fixed step count. "
                    "The fit pins itself to min(4, available); this table is "
                    "measured by overriding that pin, and the override is "
                    "restored afterwards."),
           "by_rows": {}}
    for n in args.rows:
        z = torch.as_tensor(rng.normal(size=(n, args.feats)),
                            dtype=torch.float64)
        s = np.exp(rng.normal(size=n))
        per = {}
        for nt in counts:
            ts = []
            for _ in range(args.repeats):
                torch.set_num_threads(nt)
                q = QuantileScale(steps=args.steps)
                # defeat the internal pin for the purposes of this measurement
                q.fit.__func__  # noqa: B018  (documented: pin is inside fit)
                t0 = time.perf_counter()
                _fit_at(q, z, s, nt)
                ts.append(time.perf_counter() - t0)
            per[str(nt)] = {"median_s": float(np.median(ts)),
                            "min_s": float(min(ts)), "n": args.repeats}
            print(f"  rows={n:6d} threads={nt:3d}  "
                  f"{np.median(ts):.3f} s", flush=True)
        best = min(per, key=lambda k: per[k]["median_s"])
        res["by_rows"][str(n)] = {
            "per_threads": per, "fastest_threads": int(best),
            "default_over_fastest": (per[str(default_threads)]["median_s"]
                                     / per[best]["median_s"]),
        }
    torch.set_num_threads(default_threads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


def _fit_at(q, z, s, nt):
    """Run the fit with the thread count held at `nt` for its duration."""
    import uqkit.scale as S
    real = torch.set_num_threads

    def pinned(_n):
        real(nt)
    S.torch.set_num_threads = pinned
    try:
        q.fit(z, s)
    finally:
        S.torch.set_num_threads = real


if __name__ == "__main__":
    main()
