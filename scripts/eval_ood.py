"""OOD detection: shift detection and error detection, kept apart.

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_ood.py --ckpts runs/m*/best.pt

Three detectors, all unsupervised at deployment and none fitted on anything it
is scored against:

* `spread`      -- ||sigma||/||mean||, the deep ensemble's own disagreement.
* `mahalanobis` -- squared distance in input spectral-feature space, Gaussian
                   fitted on the *training* inputs of the parent family.
* `residual`    -- ||L u_hat - f||/||f||, the PDE residual of the surrogate's
                   output under the operator the surrogate *believes* it is
                   solving (the parent's). Available for the elliptic families
                   only; `null` elsewhere, which is a fact about the family and
                   not a missing measurement.

Every shifted shard gets its own AUROC against its parent's in-distribution
test split, with a bootstrap 95% interval, because averaging an easy shift with
a hard one produces a number that describes neither.

Two things an adversarial review (codex, 2026-09-09) got right about an earlier
version of the error-detection number, both fixed here:

* the failure threshold tau was the 90th percentile of the **test** errors that
  were then scored against it, which makes the label retrospective. It is now
  taken from the `cal` split, which nothing else in this script touches.
* the three detectors were compared on different populations, because
  `residual` is undefined for the time-evolution families and those rows were
  simply dropped from its AUROC. The headline comparison is now restricted to
  the families where all three exist, and the wider per-detector number is kept
  beside it and labelled with its own population.
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
from uqkit.ood import auroc_ci, residual_score, shift_auroc, spread_score  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_members, load_shard, predict_shard  # noqa: E402

DETECTORS = ["spread", "mahalanobis", "residual"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/ood.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]
    sims = {t: PDE2DSimulator(t, device=args.device) for t in in_tasks}

    def scored(task, split, N=64):
        """All three detector scores plus the true error, for one shard."""
        blob = load_shard(args.root, task, split, N)
        mean, sigma, truth, a = predict_shard(models, blob, stats, args.device)
        parent = PARENT.get(task, task)
        sim = sims[parent]
        out = {"parent": parent, "task": task, "N": N,
               "err": rel_l2(mean, truth).cpu().numpy(),
               "spread": spread_score(sigma, mean).cpu().numpy(),
               "feat": spectral_features(a).cpu()}
        r = sim.residual(a, mean)
        out["residual"] = (residual_score(r, sim.rhs(a)).cpu().numpy()
                           if r is not None else None)
        del mean, sigma, truth, a
        torch.cuda.empty_cache()
        return out

    # detectors that need fitting see training data only
    train_feat = {}
    maha = {}
    for t in in_tasks:
        blob = load_shard(args.root, t, "train", 64)
        f = spectral_features(blob["a"][:2048].to(args.device)).cpu()
        train_feat[t] = f
        maha[t] = Mahalanobis().fit(f)
    print("fitted mahalanobis on train inputs", flush=True)

    ind = {t: scored(t, "test") for t in in_tasks}
    for t in in_tasks:
        ind[t]["mahalanobis"] = maha[t].score(ind[t]["feat"]).numpy()
    # the failure threshold comes from the calibration split, which is used
    # nowhere else in this script and is never scored
    cal_err = np.concatenate([scored(t, "cal")["err"] for t in in_tasks])

    ood_specs = []
    for kind, tasks in man["ood_suite"].items():
        ood_specs += [(kind, t, 64) for t in tasks]
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                ood_specs.append((f"resolution_{N}", t, N))

    res = {"n_members": len(models), "ckpts": args.ckpts,
           "in_dist": {}, "shift_detection": {}, "error_detection": {}}
    for t in in_tasks:
        d = ind[t]
        res["in_dist"][t] = {
            "rel_l2_mean": float(d["err"].mean()),
            "rel_l2_p90": float(np.quantile(d["err"], 0.9)),
            "residual_mean": (None if d["residual"] is None
                              else float(d["residual"].mean())),
            "n": int(len(d["err"]))}

    pool = {k: [] for k in DETECTORS + ["err", "parent", "kind", "task"]}

    def add_pool(d, kind):
        n = len(d["err"])
        pool["err"].append(d["err"])
        pool["spread"].append(d["spread"])
        pool["mahalanobis"].append(d["mahalanobis"])
        pool["residual"].append(d["residual"] if d["residual"] is not None
                                else np.full(n, np.nan))
        pool["parent"] += [d["parent"]] * n
        pool["kind"] += [kind] * n
        pool["task"] += [d["task"]] * n

    for t in in_tasks:
        add_pool(ind[t], "in_dist")

    for kind, task, N in ood_specs:
        d = scored(task, "ood", N)
        p = d["parent"]
        d["mahalanobis"] = maha[p].score(d["feat"]).numpy()
        key = f"{kind}/{task}/N{N}"
        rec = {"kind": kind, "parent": p, "N": N, "n": int(len(d["err"])),
               "rel_l2_mean": float(d["err"].mean()),
               "rel_l2_in_dist": float(ind[p]["err"].mean()),
               "auroc": {}}
        for det in DETECTORS:
            a_in, a_ood = ind[p].get(det), d.get(det)
            if a_in is None or a_ood is None:
                rec["auroc"][det] = None
                continue
            v = shift_auroc(a_in, a_ood)
            s = np.concatenate([a_in, a_ood])
            y = np.concatenate([np.zeros(len(a_in)), np.ones(len(a_ood))])
            lo, hi = auroc_ci(s, y)
            rec["auroc"][det] = {"auroc": v, "ci95": [lo, hi]}
        res["shift_detection"][key] = rec
        add_pool(d, kind)
        print(f"{key:44s} err {d['err'].mean():.4f}  " +
              "  ".join(f"{k}={rec['auroc'][k]['auroc']:.3f}"
                        if rec['auroc'][k] else f"{k}=--" for k in DETECTORS),
              flush=True)

    # ---- error detection ----------------------------------------------------
    err = np.concatenate(pool["err"])
    kinds = np.array(pool["kind"])
    parents = np.array(pool["parent"])
    tau = float(np.quantile(cal_err, 0.90))
    y = err > tau
    # the families where all three detectors are defined -- the only population
    # on which they can be ranked against each other
    common = np.isin(parents, [t for t in in_tasks
                               if ind[t]["residual"] is not None])
    ed = {"tau": tau, "tau_definition":
          "90th percentile of rel-L2 on the CALIBRATION split, pooled over the "
          "five trained families. The calibration split is scored nowhere else "
          "in this file, so the threshold is independent of every sample the "
          "AUROC is computed over.",
          "positive_rate": float(y.mean()), "n": int(len(y)),
          "common_population": [t for t in in_tasks
                                if ind[t]["residual"] is not None],
          "detectors": {}, "detectors_common_population": {}}
    for det in DETECTORS:
        sc = np.concatenate(pool[det])
        m = ~np.isnan(sc)
        lo, hi = auroc_ci(sc[m], y[m])
        ed["detectors"][det] = {
            "auroc": auroc(sc[m], y[m]), "ci95": [lo, hi], "n": int(m.sum()),
            "population": ("families with a cheap residual"
                           if det == "residual" else "all five families"),
            # stratified by parent family: a pooled AUROC can be driven by one
            # family being harder than another rather than by ranking failures
            # within a family, so both are reported
            "per_parent": {p: auroc(sc[m & (parents == p)], y[m & (parents == p)])
                           for p in in_tasks}}
        mc = m & common
        lo, hi = auroc_ci(sc[mc], y[mc])
        ed["detectors_common_population"][det] = {
            "auroc": auroc(sc[mc], y[mc]), "ci95": [lo, hi], "n": int(mc.sum())}
    res["error_detection"] = ed
    for det, v in ed["detectors"].items():
        c = ed["detectors_common_population"][det]
        print(f"error-detection AUROC  {det:12s} all {v['auroc']:.3f} "
              f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] n={v['n']}   "
              f"common {c['auroc']:.3f} n={c['n']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
