"""Iso-accuracy speedup: how fast is the *cheapest solver that is as accurate*?

    CUDA_VISIBLE_DEVICES=3 python scripts/bench_isoaccuracy.py --ckpts runs/m*/best.pt

`bench_speedup.py` times the reference solver at the tolerance the corpus was
generated with -- Darcy's PCG at 1e-10, Navier-Stokes at dt=1e-3. That is the
right denominator for "how much did the surrogate save against the data
generator", and the wrong one for "should I deploy the surrogate": nobody
converges a solver to 1e-10 to get an answer they will then accept at 5% error.

So this sweeps the solver's own accuracy knob, measures rel-L2 against a
tighter reference, and reports the **iso-accuracy speedup** -- surrogate against
the cheapest solver setting that matches the surrogate's own error. It is the
number a plant engineer would ask for, it is strictly smaller than the headline,
and if it falls below 100x then the headline speedup is an artefact of an
over-converged baseline and should be reported as such.

Knobs, one per family:
* Darcy   -- PCG relative-residual tolerance (the iteration count follows).
* Navier-Stokes -- the RK4 step size.
The four exact-propagator families have no knob: their solver is two FFTs and
is already the cheapest thing that computes the answer at all.
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
from uqkit.sims import pde2d as P  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402

DARCY_TOLS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
NS_DTS = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/isoaccuracy.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    dev = "cuda"
    models, ck = load_members(args.ckpts, dev)
    stats = ck["stats"]
    M = len(models)
    B = args.batch
    res = {"env": env_report(dev), "n_members": M, "batch": B, "families": {}}

    # ---------------- Darcy: tolerance sweep --------------------------------
    blob = load_shard(args.root, "darcy", "test", 64)
    a = blob["a"][:B].to(dev)
    coef, f = torch.exp(a[:, 0]), a[:, 1]
    u_ref, r_ref = P.solve_darcy(coef, f, tol=1e-12, max_iter=5000)
    st = stats["darcy"]
    a_norm = (a - st["a_mean"].to(dev)) / st["a_std"].to(dev)
    tid = torch.full((B,), TASK_ID["darcy"], device=dev, dtype=torch.long)

    def surrogate():
        preds = []
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(m(a_norm, tid).float())
        p = torch.stack(preds)
        return p.mean(0), p.std(0, unbiased=True)

    mean, _ = surrogate()
    mean = mean * st["u_std"].to(dev) + st["u_mean"].to(dev)
    sur_err = float(rel_l2(mean, u_ref.unsqueeze(1)).mean())
    t_sur = timeit(surrogate, 3, args.iters, dev)

    rows = []
    for tol in DARCY_TOLS:
        u, _ = P.solve_darcy(coef, f, tol=tol, max_iter=2000)
        err = float(rel_l2(u.unsqueeze(1), u_ref.unsqueeze(1)).mean())
        t = timeit(lambda tol=tol: P.solve_darcy(coef, f, tol=tol, max_iter=2000),
                   2, args.iters, dev)
        rows.append({"tol": tol, "rel_l2_vs_reference": err,
                     "s": t["median_s"], "timing": t,
                     "speedup_vs_surrogate": t["median_s"] / t_sur["median_s"]})
        print(f"darcy tol={tol:.0e}  err {err:.5f}  {t['median_s']*1e3:8.2f} ms  "
              f"{t['median_s']/t_sur['median_s']:8.1f}x", flush=True)
    matched = [r for r in rows if r["rel_l2_vs_reference"] <= sur_err]
    iso = min(matched, key=lambda r: r["s"]) if matched else None
    res["families"]["darcy"] = {
        "knob": "PCG relative-residual tolerance",
        "reference": {"tol": 1e-12, "max_iter": 5000,
                      "achieved_residual": r_ref},
        "surrogate": {"rel_l2_vs_reference": sur_err, "s": t_sur["median_s"],
                      "members": M, "precision": "bf16 autocast",
                      "timing": t_sur},
        "sweep": rows,
        "iso_accuracy": (None if iso is None else
                         {"tol": iso["tol"],
                          "solver_rel_l2": iso["rel_l2_vs_reference"],
                          "solver_s": iso["s"],
                          "speedup": iso["speedup_vs_surrogate"]}),
        "headline_speedup_at_1e-10": next(
            r["speedup_vs_surrogate"] for r in rows if r["tol"] == 1e-10),
    }

    # ---------------- Navier-Stokes: step-size sweep ------------------------
    # No trained surrogate for this family, so no iso-accuracy ratio: this
    # measures only how much of the reference solver's cost is *needed*, which
    # is what decides whether a future NS speedup claim would be honest.
    blob = load_shard(args.root, "navier_stokes", "ood", 64)
    w0 = blob["a"][:B, 0].to(dev)
    cfg = P.CFG["navier_stokes"]
    w_ref, _ = P.solve_navier_stokes(w0, cfg["nu"], cfg["T"], 2.5e-4)
    ns_rows = []
    for dt in NS_DTS:
        try:
            w, _ = P.solve_navier_stokes(w0, cfg["nu"], cfg["T"], dt)
            err = float(rel_l2(w.unsqueeze(1), w_ref.unsqueeze(1)).mean())
            t = timeit(lambda dt=dt: P.solve_navier_stokes(w0, cfg["nu"],
                                                           cfg["T"], dt),
                       1, max(3, args.iters // 3), dev)
            ns_rows.append({"dt": dt, "steps": int(round(cfg["T"] / dt)),
                            "rel_l2_vs_reference": err, "s": t["median_s"]})
            print(f"ns dt={dt:.1e} ({int(cfg['T']/dt):5d} steps)  err {err:.5f}  "
                  f"{t['median_s']*1e3:8.2f} ms", flush=True)
        except Exception as e:                      # blow-up at large dt
            ns_rows.append({"dt": dt, "steps": int(round(cfg["T"] / dt)),
                            "error": repr(e)})
            print(f"ns dt={dt:.1e} failed: {e!r}", flush=True)
    res["families"]["navier_stokes"] = {
        "knob": "RK4 step size", "trained": False,
        "reference": {"dt": 2.5e-4},
        "sweep": ns_rows,
        "note": ("no surrogate is trained on this family -- it is held out as an "
                 "unseen operator for the OOD clause -- so no speedup is claimed "
                 "here. The sweep says how much of the 1,000-step reference cost "
                 "is actually required for a given accuracy.")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nsurrogate rel-L2 {sur_err:.5f}; iso-accuracy speedup: "
          f"{res['families']['darcy']['iso_accuracy']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
