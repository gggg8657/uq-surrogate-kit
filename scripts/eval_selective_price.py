"""H14b: what abstention rate would the clause cost? The price curve.

    CUDA_VISIBLE_DEVICES=3 ~/miniforge3/envs/pdeno/bin/python \
        scripts/eval_selective_price.py --ckpt runs/u0/best.pt \
        --sigma-source het --out runs/selp_u0_het.json

H14 fixed the gate's operating point at beta = 0.05 -- a 5% in-distribution
false-abstention rate -- before any number existed, and measured 0/32 covariate
shards in band there, *including with an oracle gate on the true error*. That
answers "does gating at 5% deliver the clause" (no) but not the question a
reader will ask next, which is the one that decides whether deliberate
abstention is a product at all:

    **at what abstention rate, if any, does selective coverage enter
    [0.88, 0.92] -- and is that rate one anybody would pay?**

So this sweeps beta over a grid and reports, per shard, the smallest beta whose
selective coverage lands in band with at least `--min-accepted` points left.
Everything else is H14's protocol unchanged: tau is still the (1-beta) quantile
of the gate score on the parent family's *calibration* split, the conformal
quantile is still refit on the accepted calibration points at each beta, and
the ground-truth `oracle_err` gate is still fenced off as a ceiling rather than
presented as a detector.

This is a *measurement* of the operating curve, not a second intervention. The
registered operating point stays beta = 0.05; nothing here relocates it, and
the minimum-beta column is reported as a price, never substituted for the
0.05 row. Reading the price off this curve and then quoting the coverage at
that beta as "the result" would be selecting a threshold on the evaluation set,
which is the mistake H13 priced at a third of its own effect -- so the
per-shard minimum beta is reported *with* the leave-one-shard-out rate that a
rule fixed in advance would have paid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import GroupConformal, get_score  # noqa: E402
from uqkit.metrics import binom_ci, rel_l2  # noqa: E402
from uqkit.ood import consistency_score, spread_score  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

BAND = (0.88, 0.92)
COVARIATE_KINDS = {"input_shift", "graded_rough"}
GATES = ["sigma", "consist", "oracle_err"]
ORACLE_GATES = {"oracle_err"}
BETAS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--min-accepted", type=int, default=100)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default=None,
                    choices=["het", "cqr", "const"])
    args = ap.parse_args()
    single = args.sigma_source is not None
    if single and len(args.ckpt) != 1:
        ap.error("--sigma-source is a single-network mode; pass one checkpoint")

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    if single:
        model, ck = load_model(args.ckpt[0], args.device)
        if not ck["args"].get("uq"):
            raise SystemExit(f"{args.ckpt[0]} is not a UQ checkpoint")
        models = [model]
    else:
        models, ck = load_members(args.ckpt, args.device)
    stats = ck["stats"]
    sims: dict[str, PDE2DSimulator] = {}
    fn, _ = get_score(args.score)

    def sim_for(task):
        if task not in sims:
            sims[task] = PDE2DSimulator(task, device=args.device)
        return sims[task]

    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        if single:
            m, s, t, a = predict_shard_single(
                models[0], blob, stats, args.device,
                sigma_source=args.sigma_source)
        else:
            m, s, t, a = predict_shard(models, blob, stats, args.device)
        parent = PARENT.get(task, task)
        g = {"sigma": spread_score(s, m).cpu().numpy(),
             "oracle_err": rel_l2(m, t).cpu().numpy()}
        psim = sim_for(parent)
        r = psim.residual(a, m)
        if r is None:
            g["consist"] = None
        else:
            f = psim.rhs(a)
            g["consist"] = consistency_score(r, r + f, f).cpu().numpy()
        return {"parent": parent, "gate": g,
                "sigma_flat": s.flatten(), "mean": m, "truth": t, "sigma": s}

    cal = {t: gather(t, "cal") for t in in_tasks}
    MED = float(torch.cat([cal[t]["sigma_flat"] for t in in_tasks]).median())
    cal_s = {t: fn(cal[t]["mean"], cal[t]["sigma"], cal[t]["truth"],
                   med=MED).cpu().numpy() for t in in_tasks}
    for t in in_tasks:  # predictions are no longer needed, only the scores
        cal[t].pop("mean"), cal[t].pop("truth"), cal[t].pop("sigma")
        cal[t].pop("sigma_flat")

    specs = []
    for kind, tasks in man["ood_suite"].items():
        if kind in COVARIATE_KINDS:
            specs += [(kind, t, 64) for t in tasks]

    # tau grid and the gated quantile at each beta, all from calibration only
    cals = {}
    for gate in GATES:
        per_beta = {}
        for beta in BETAS:
            tau, acc = {}, {}
            for t in in_tasks:
                v = cal[t]["gate"][gate]
                if v is None:
                    tau[t], acc[t] = None, None
                else:
                    tau[t] = float(np.quantile(v, 1.0 - beta))
                    acc[t] = v <= tau[t]
            fams = [t for t in in_tasks if acc[t] is not None]
            if not fams:
                continue
            s = np.concatenate([cal_s[t][acc[t]] for t in fams])
            g = np.concatenate([[t] * int(acc[t].sum()) for t in fams])
            per_beta[beta] = {
                "tau": tau, "fams": fams,
                "conf": GroupConformal(args.alpha).fit(s, g),
                "n_cal": {t: int(acc[t].sum()) for t in fams}}
        cals[gate] = per_beta

    res = {"alpha": args.alpha, "betas": BETAS, "band": list(BAND),
           "min_accepted": args.min_accepted, "score": args.score,
           "ckpt": args.ckpt, "seed": ck["args"].get("seed"),
           "sigma_source": args.sigma_source or "ensemble_spread",
           "oracle_gates": sorted(ORACLE_GATES),
           "sigma_floor_median": MED,
           "registered_operating_beta": 0.05,
           "by_gate": {g: {"n_cal_by_beta": {f"{b:g}": cals[g][b]["n_cal"]
                                             for b in cals[g]},
                           "shards": {}} for g in GATES}}

    for kind, task, N in specs:
        d = gather(task, "ood", N)
        parent = d["parent"]
        s = fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy()
        for gate in GATES:
            v = d["gate"][gate]
            if v is None or parent not in cals[gate].get(BETAS[0], {})["fams"]:
                continue
            curve, first = {}, None
            for beta in BETAS:
                c = cals[gate].get(beta)
                if c is None or c["tau"][parent] is None:
                    continue
                a_te = v <= c["tau"][parent]
                n_acc = int(a_te.sum())
                grp = np.array([parent] * int(a_te.sum()))
                if n_acc:
                    cov = float(c["conf"].covered(s[a_te], grp).mean())
                    lo, hi = binom_ci(int(round(cov * n_acc)), n_acc)
                else:
                    cov, lo, hi = None, None, None
                ok = (cov is not None and n_acc >= args.min_accepted
                      and BAND[0] <= cov <= BAND[1])
                curve[f"{beta:g}"] = {
                    "abstention": float(1.0 - a_te.mean()),
                    "n_accepted": n_acc, "coverage": cov, "ci95": [lo, hi],
                    "measured": bool(n_acc >= args.min_accepted),
                    "in_band": bool(ok)}
                if ok and first is None:
                    first = beta
            res["by_gate"][gate]["shards"][f"{kind}/{task}/N{N}"] = {
                "kind": kind, "parent": parent, "N": N,
                "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean()),
                "curve": curve,
                "min_beta_in_band": first,
                "uses_ground_truth": gate in ORACLE_GATES}
        del d
        torch.cuda.empty_cache()

    for gate in GATES:
        sh = res["by_gate"][gate]["shards"]
        got = [v["min_beta_in_band"] for v in sh.values()
               if v["min_beta_in_band"] is not None]
        res["by_gate"][gate]["n_shards"] = len(sh)
        res["by_gate"][gate]["n_ever_in_band"] = len(got)
        res["by_gate"][gate]["min_beta_median"] = (
            float(np.median(got)) if got else None)
        print(f"[{gate}] {len(got)}/{len(sh)} covariate shards reach the band "
              f"at SOME beta in {BETAS}; median price "
              f"{res['by_gate'][gate]['min_beta_median']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
