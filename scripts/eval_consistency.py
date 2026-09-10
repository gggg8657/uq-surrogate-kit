"""H6: solver-consistency OOD detection, in both deployments, kept apart.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_consistency.py \
        --ckpts runs/m0/best.pt --out runs/consistency_M1.json

`scripts/eval_ood.py` scores the PDE residual under `PARENT[task]` -- the
operator the surrogate *is configured for*. That models deployment **(A)**:
the process changed and nobody reconfigured the model. Under (A) the observable
at test time is `(f, poisson)` for an operator-shift shard exactly as it is for
an in-distribution one, so every unsupervised detector that is a function of
(input, configured operator) is at chance by construction -- which is what the
0.486-0.503 in `runs/ood.json` is, and it is an identity, not a defect.

This file measures deployment **(B)**: the request names the operator, and it
names one the surrogate was not trained on. The observable is now
`(f, biharmonic)`, the configured operator is genuinely different information,
and `L_requested u_hat` is far from `f` because `u_hat` is not a solution of the
requested equation. Both are reported; (A) stays the strict reading.

Three things keep (B) from being credited with work it did not do, all fixed in
`critique_log.md` before this ran:

* the score is **dimensionless** (`uqkit.ood.consistency_score`), because
  ||L u - f|| / ||f|| is not comparable between a second-order request and a
  sixth-order one;
* `floor_c`, the same score on the shard's **ground truth**, is reported beside
  every AUROC. It is an offline diagnostic the detector never sees. `floor_c`
  near 1 means the apply is round-off dominated and the AUROC is an
  operator-identity signal;
* the **`lookup` baseline** -- `1 if task not in PRETRAIN_TASKS else 0`, a dict
  lookup costing nothing -- is scored on every shard. If the request names the
  operator then this baseline is perfect on operator shift, and the residual
  has to earn credit over it rather than over chance.

The 33 shards that shift only the input leave `L_requested == L_parent`, so (B)
is byte-identical to (A) there. Those rows are the control and they are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.features import Mahalanobis, spectral_features  # noqa: E402
from uqkit.metrics import auroc, rel_l2  # noqa: E402
from uqkit.ood import auroc_ci, consistency_score, shift_auroc  # noqa: E402
from uqkit.sims.pde2d import PARENT, PRETRAIN_TASKS  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402


def consistency(sim, a, u):
    """`consistency_score` of field `u` under `sim`'s operator, or None.

    `applied` is recovered as `residual + rhs` rather than by a second apply, so
    the numerator and the denominator come from the same floating-point
    evaluation of L and no extra FFT is spent.
    """
    r = sim.residual(a, u)
    if r is None:
        return None
    f = sim.rhs(a)
    return consistency_score(r, r + f, f).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/consistency.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    models, ck = load_members(args.ckpts, args.device)
    TRAINED_OPS = {PDE2DSimulator(t, device="cpu").operator_key()
                   for t in PRETRAIN_TASKS}
    stats = ck["stats"]
    sims: dict[str, PDE2DSimulator] = {}

    def sim_for(task):
        if task not in sims:
            sims[task] = PDE2DSimulator(task, device=args.device)
        return sims[task]

    def scored(task, split, N=64):
        """Both presentations, the truth's floor, and the input features."""
        blob = load_shard(args.root, task, split, N)
        mean, _sigma, truth, a = predict_shard(models, blob, stats, args.device)
        parent = PARENT.get(task, task)
        out = {"task": task, "parent": parent, "N": N,
               "err": rel_l2(mean, truth).cpu().numpy(),
               "feat": spectral_features(a).cpu(),
               "same_operator":
                   sim_for(task).operator_key() == sim_for(parent).operator_key(),
               "trained_operator":
                   sim_for(task).operator_key() in TRAINED_OPS}
        out["cons_A"] = consistency(sim_for(parent), a, mean)
        out["cons_B"] = consistency(sim_for(task), a, mean)
        # offline diagnostic: what the *exact* solution of the requested
        # equation scores under the requested operator. Never fed to a detector.
        out["floor_c"] = consistency(sim_for(task), a, truth)
        del mean, _sigma, truth, a
        torch.cuda.empty_cache()
        return out

    maha = {}
    for t in in_tasks:
        blob = load_shard(args.root, t, "train", 64)
        maha[t] = Mahalanobis().fit(
            spectral_features(blob["a"][:2048].to(args.device)).cpu())
    print("fitted mahalanobis on train inputs", flush=True)

    ind = {t: scored(t, "test") for t in in_tasks}
    for t in in_tasks:
        ind[t]["mahalanobis"] = maha[t].score(ind[t]["feat"]).numpy()

    # z-standardisation for the combined detector comes from the CALIBRATION
    # split, in distribution only. Nothing OOD touches it, so the combination
    # is not fitted on anything it is scored against.
    zst = {}
    for t in in_tasks:
        c = scored(t, "cal")
        c["mahalanobis"] = maha[t].score(c["feat"]).numpy()
        zst[t] = {}
        for k, v in (("mahalanobis", c["mahalanobis"]), ("cons_B", c["cons_B"])):
            zst[t][k] = (None if v is None
                         else {"mu": float(np.mean(v)),
                               "sd": float(np.std(v) + 1e-12)})

    def _z(d, k):
        v, s = d.get(k), zst[d["parent"]].get(k)
        return None if (v is None or s is None) else (np.asarray(v) - s["mu"]) / s["sd"]

    def combo(d):
        """max over available z-scores; mahalanobis alone where L has no apply."""
        zs = [z for z in (_z(d, "mahalanobis"), _z(d, "cons_B")) if z is not None]
        return np.max(np.stack(zs), axis=0) if zs else None

    def router(d):
        """What the kit would ship: use the consistency residual only when the
        request names an operator we did not train on, else the input detector.

        The routing predicate is the free `lookup` and is known at deployment,
        so this is not fitted on anything. It is also why the router's AUROC on
        the six operator-shift shards must NOT be read as the residual beating
        the lookup -- there the lookup is already 1.000 and the router inherits
        it. What the residual adds over the lookup is severity and `floor_c`,
        not separation. Reported so the comparison is visible, not to bank it.
        """
        if not d["trained_operator"] and _z(d, "cons_B") is not None:
            return _z(d, "cons_B")
        return _z(d, "mahalanobis")

    for t in in_tasks:
        ind[t]["combo"] = combo(ind[t])
        ind[t]["router"] = router(ind[t])

    ood_specs = []
    for kind, tasks in man["ood_suite"].items():
        ood_specs += [(kind, t, 64) for t in tasks]
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                ood_specs.append((f"resolution_{N}", t, N))

    DET = ["mahalanobis", "cons_A", "cons_B", "combo", "router", "lookup"]
    res = {"n_members": len(models), "ckpts": args.ckpts,
           "pretrain_tasks": list(PRETRAIN_TASKS),
           "presentation": {
               "A": "residual under PARENT[task] -- the configured operator. "
                    "The process moved and nobody reconfigured the model. "
                    "Provably at chance for operator shift; strict reading.",
               "B": "residual under the REQUESTED operator (the shard's own "
                    "task). The request names an operator the surrogate was "
                    "not trained on."},
           "shards": {}}

    for t in in_tasks:
        d = ind[t]
        res.setdefault("in_dist", {})[t] = {
            "rel_l2_mean": float(d["err"].mean()),
            "cons_A_mean": None if d["cons_A"] is None else float(d["cons_A"].mean()),
            "floor_c_mean": None if d["floor_c"] is None else float(d["floor_c"].mean()),
            "n": int(len(d["err"]))}

    for kind, task, N in ood_specs:
        d = scored(task, "ood", N)
        p = d["parent"]
        d["mahalanobis"] = maha[p].score(d["feat"]).numpy()
        d["combo"] = combo(d)
        d["router"] = router(d)
        n_in, n_ood = len(ind[p]["err"]), len(d["err"])
        # the free baseline: the request names an operator we did not train on
        # `task not in PRETRAIN_TASKS` would fire on `poisson_rough` too, whose
        # operator is Poisson; the baseline is about the operator, not the name.
        d["lookup"] = np.full(n_ood, 0.0 if d["trained_operator"] else 1.0)
        ind_lookup = np.zeros(n_in)
        key = f"{kind}/{task}/N{N}"
        rec = {"kind": kind, "parent": p, "N": N, "n": n_ood,
               "rel_l2_mean": float(d["err"].mean()),
               "rel_l2_in_dist": float(ind[p]["err"].mean()),
               "operator_changed": not d["same_operator"],
               "B_equals_A": bool(d["same_operator"]),
               "floor_c": (None if d["floor_c"] is None
                           else float(np.median(d["floor_c"]))),
               "cons_B_median": (None if d["cons_B"] is None
                                 else float(np.median(d["cons_B"]))),
               "headroom": None, "auroc": {}}
        if d["cons_B"] is not None and d["floor_c"] is not None:
            fl = float(np.median(d["floor_c"]))
            rec["headroom"] = float(np.median(d["cons_B"]) / max(fl, 1e-12))
        for det in DET:
            a_in = ind_lookup if det == "lookup" else ind[p].get(det)
            a_ood = d.get(det)
            if a_in is None or a_ood is None:
                rec["auroc"][det] = None
                continue
            v = shift_auroc(a_in, a_ood)
            s = np.concatenate([np.asarray(a_in), np.asarray(a_ood)])
            y = np.concatenate([np.zeros(n_in), np.ones(n_ood)])
            lo, hi = auroc_ci(s, y)
            rec["auroc"][det] = {"auroc": v, "ci95": [lo, hi]}
        res["shards"][key] = rec
        fmt = lambda k: (f"{rec['auroc'][k]['auroc']:.3f}"
                         if rec["auroc"][k] else "  --")
        print(f"{key:40s} op{'!' if rec['operator_changed'] else '='} "
              f"err {d['err'].mean():8.4f}  "
              + "  ".join(f"{k}={fmt(k)}" for k in DET)
              + f"   floor_c={rec['floor_c'] if rec['floor_c'] is None else round(rec['floor_c'], 4)}"
              + f" headroom={rec['headroom'] if rec['headroom'] is None else round(rec['headroom'], 2)}",
              flush=True)

    # ---- clause arithmetic, computed here so no document hand-counts it -----
    def tally(det):
        vals = [r["auroc"][det]["auroc"] for r in res["shards"].values()
                if r["auroc"].get(det)]
        return {"n_shards_scored": len(vals),
                "n_ge_0p9": int(sum(v >= 0.9 for v in vals)),
                "n_total_shards": len(res["shards"]),
                "min": float(min(vals)) if vals else None,
                "median": float(np.median(vals)) if vals else None}
    res["summary"] = {det: tally(det) for det in DET}
    op_keys = [k for k, r in res["shards"].items() if r["operator_changed"]]
    res["operator_shift_shards"] = op_keys
    res["summary_operator_shift"] = {
        det: {"n_scored": sum(1 for k in op_keys if res["shards"][k]["auroc"].get(det)),
              "n_ge_0p9": sum(1 for k in op_keys
                              if res["shards"][k]["auroc"].get(det)
                              and res["shards"][k]["auroc"][det]["auroc"] >= 0.9),
              "n_total": len(op_keys)} for det in DET}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    print(json.dumps(res["summary_operator_shift"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
