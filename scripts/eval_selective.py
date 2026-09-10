"""H14: the deliberate-abstention reading of clause 1 under shift.

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/eval_selective.py --ckpt runs/u0/best.pt \
        --out runs/sel_u0_het.json --sigma-source het

H13 established that coverage under covariate shift decays monotonically with
shift strength on *both* calibrators and reaches zero on both, so past some
strength the shift moves p(y|x) and no reweighting of x can reach it. This
script measures the other reading of the same clause: **certify where the model
is still in competence and abstain deliberately elsewhere, with the abstention
rate reported as a first-class number.**

Protocol, fixed in `critique_log.md` (H14) before this ran:

* The gate is a function of the deployment observables only -- the prediction
  and its sigma, or the prediction and the *configured* operator. It never sees
  ground truth and never sees a coverage number.
* tau is the (1-beta) empirical quantile of the gate score on the parent
  family's **calibration** split, beta = 0.05. So in distribution the gate
  refuses 5% by construction and that 5% is the quoted price.
* The conformal quantile is fit on `{x in cal : gate(x) <= tau}` -- the same rule
  applied to the calibration split, so the certified and the calibrated
  population are one population.
* This is **not** a distribution-free guarantee. Acceptance is a function of x,
  the distribution of x moved, and the accepted subpopulation therefore still
  differs between cal and test. Every row is a conditional-coverage claim about
  the accepted region and is labelled `selective`.
* **The trap:** as abstention -> 1, coverage on the survivors becomes both
  meaningless and easy, which is exactly the mistake H13 caught this repo
  making. So a shard counts toward an in-band tally only if
  `n_accepted >= --min-accepted` (100 of 512), every cell carries
  `n_accepted`/`abstention`/width, and the ungated marginal coverage -- the
  strict reading -- is written into the same record.

Three gates, all computed in one run because they share the predictions. Which
one ships was fixed before the numbers existed:

* `sigma`   -- ||sigma||_2 / ||mean||_2 from the same forward pass. THE SHIPPED
               GATE: free, no operator apply, does not touch the 100x row.
* `consist` -- `uqkit.ood.consistency_score` under the configured operator, one
               extra apply and no solve. `PDE2DSimulator.residual` returns None
               for the two time-stepped families, so this gate exists for
               poisson/helmholtz/darcy only and its rows record `available`.
* `oracle_err` -- the TRUE relative error. Not a detector and not shippable: it
               is the *ceiling* of competence gating at a given abstention rate.
               Its only job is to separate "the gate score is the wrong signal"
               from "gating at 5% cannot reach the band whatever the signal".
               Every row it produces carries `uses_ground_truth: true`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import GroupConformal, _floor, get_score  # noqa: E402
from uqkit.metrics import binom_ci, rel_l2  # noqa: E402
from uqkit.ood import consistency_score, spread_score  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

SCORE_NAMES = ["field_max", "norm_ratio", "rel_l2"]
# The clause-1-under-shift population, identical to the 32 shards H13 used, so
# the gated and ungated readings are comparable shard for shard. `param_oor`
# and `unseen_operator` change the governing equation, so they are not
# covariate shift and are excluded here exactly as they were there.
COVARIATE_KINDS = {"input_shift", "graded_rough"}
GATES = ["sigma", "consist", "oracle_err"]
#: `oracle_err` gates on the TRUE relative error. It is not shippable and is
#: never presented as a detector: it is the ceiling of competence gating at a
#: given abstention rate, and it exists to separate two very different
#: conclusions -- "the gate score is the wrong signal" from "gating at this
#: abstention rate cannot reach the band whatever the signal". Rows carrying it
#: are labelled `uses_ground_truth`.
ORACLE_GATES = {"oracle_err"}


def _rankdata(v):
    """Ranks of `v` with ties assigned their average rank."""
    order = np.argsort(v, kind="stable")
    sv = v[order]
    r = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    """Rank correlation, the diagnostic that says whether a gate can work.

    A gate selects on its own score; coverage fails on samples with a large
    *conformity* score. If the two are uncorrelated within a shard then
    removing the worst 5% by gate score removes nothing in particular by
    conformity score, and the gate cannot move coverage however well it detects
    the shift. That is a different failure from "the threshold is too loose",
    and this number is what distinguishes them.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 8:
        return None
    # ties averaged: `argsort(argsort(.))` breaks them by array order, which
    # turns a constant series into a perfect correlation instead of an
    # undefined one. Harmless on 512 continuous scores, wrong in principle,
    # and the same bug mattered in `agg_selective.py`.
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return None if d == 0 else float(rx @ ry / d)


def cov_entry(covered):
    c = np.asarray(covered, dtype=bool)
    k, n = int(c.sum()), int(len(c))
    if n == 0:
        return {"coverage": None, "n": 0, "ci95": [None, None]}
    lo, hi = binom_ci(k, n)
    return {"coverage": k / n, "n": n, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one UQ checkpoint (with --sigma-source), or several "
                         "ensemble members")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.05,
                    help="in-distribution false-abstention rate the gate is "
                         "calibrated to. tau is the (1-beta) quantile of the "
                         "gate score on the parent's cal split. Fixed at 0.05 "
                         "in H14 before any number existed.")
    ap.add_argument("--min-accepted", type=int, default=100,
                    help="a shard below this many accepted points of 512 is "
                         "reported as [not measured], never as in-band")
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

    def sim_for(task):
        if task not in sims:
            sims[task] = PDE2DSimulator(task, device=args.device)
        return sims[task]

    def gather(task, split, N=64):
        """Predictions plus every gate score.

        The two *shippable* gates (`sigma`, `consist`) are functions of the
        prediction and the configured operator and never touch `truth`.
        `oracle_err` does, by construction, and is fenced off by
        `ORACLE_GATES` everywhere it could be mistaken for a detector.
        """
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
        # The residual is taken under the CONFIGURED operator -- the parent's --
        # because that is what a deployment has applied to it. On a covariate
        # shift the configured operator is also the true one, so this is the
        # honest signal and not an operator-identity lookup in disguise.
        psim = sim_for(parent)
        r = psim.residual(a, m)
        if r is None:
            g["consist"] = None
        else:
            f = psim.rhs(a)
            g["consist"] = consistency_score(r, r + f, f).cpu().numpy()
        return {"mean": m, "sigma": s, "truth": t, "a": a, "task": task,
                "parent": parent, "N": N, "gate": g}

    cal = {t: gather(t, "cal") for t in in_tasks}
    test = {t: gather(t, "test") for t in in_tasks}

    specs = []
    for kind, tasks in man["ood_suite"].items():
        if kind in COVARIATE_KINDS:
            specs += [(kind, t, 64) for t in tasks]
    # resolution shards are reported separately: they change the grid, not just
    # the input law, so they are not part of the 32-shard clause population
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                specs.append((f"resolution_{N}", t, N))

    MED = float(torch.cat([cal[t]["sigma"].flatten()
                           for t in in_tasks]).median())

    res = {"alpha": args.alpha, "beta": args.beta,
           "min_accepted": args.min_accepted,
           "oracle_gates": sorted(ORACLE_GATES),
           "ckpt": args.ckpt, "n_members": len(models),
           "sigma_source": args.sigma_source or "ensemble_spread",
           "forward_passes_per_interval": 1 if single else len(models),
           "seed": ck["args"].get("seed"),
           "sigma_floor_median": MED, "sigma_floor_frac": 0.05,
           "n_covariate_shards": sum(1 for k, _, _ in specs
                                     if k in COVARIATE_KINDS),
           "gate_available": {g: sorted(t for t in in_tasks
                                        if cal[t]["gate"][g] is not None)
                              for g in GATES},
           "tau": {}, "gates": {}}

    # ---- tau, from the calibration split alone -----------------------------
    tau = {g: {} for g in GATES}
    for g in GATES:
        for t in in_tasks:
            v = cal[t]["gate"][g]
            tau[g][t] = (None if v is None
                         else float(np.quantile(v, 1.0 - args.beta)))
    res["tau"] = tau

    def accept(d, gate):
        """Boolean accept mask under the parent's tau, or None if unavailable."""
        v = d["gate"][gate]
        th = tau[gate][d["parent"]]
        if v is None or th is None:
            return None
        return v <= th

    for gate in GATES:
        gout = {"per_score": {}}
        for sname in SCORE_NAMES:
            fn, _ = get_score(sname)
            cal_s = {t: fn(cal[t]["mean"], cal[t]["sigma"], cal[t]["truth"],
                           med=MED).cpu().numpy() for t in in_tasks}

            # ungated baseline: exactly the `group` calibrator the headline uses
            pooled = np.concatenate([cal_s[t] for t in in_tasks])
            pooled_g = np.concatenate([[t] * len(cal_s[t]) for t in in_tasks])
            ungated = GroupConformal(args.alpha).fit(pooled, pooled_g)

            # gated: same rule applied to the calibration split
            acc_cal = {t: accept(cal[t], gate) for t in in_tasks}
            fams = [t for t in in_tasks if acc_cal[t] is not None]
            if not fams:
                gout["per_score"][sname] = {"available": False}
                continue
            gsel = np.concatenate([cal_s[t][acc_cal[t]] for t in fams])
            gsel_g = np.concatenate([[t] * int(acc_cal[t].sum())
                                     for t in fams])
            gated = GroupConformal(args.alpha).fit(gsel, gsel_g)

            out = {"available": True,
                   "families": fams,
                   "q_ungated": {k: float(v) for k, v in ungated.q.items()},
                   "q_gated": {k: float(v) for k, v in gated.q.items()},
                   "n_cal_gated": {t: int(acc_cal[t].sum()) for t in fams},
                   "in_dist": {}, "shards": {}}

            # in distribution: the price of the gate, and the coverage it buys
            for t in fams:
                d, a_te = test[t], accept(test[t], gate)
                s = fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy()
                grp = np.array([t] * len(s))
                out["in_dist"][t] = {
                    "selective": cov_entry(
                        gated.covered(s[a_te], grp[a_te])),
                    "marginal_ungated": cov_entry(ungated.covered(s, grp)),
                    "marginal_gated_q": cov_entry(gated.covered(s, grp)),
                    "abstention": float(1.0 - a_te.mean()),
                    "n_accepted": int(a_te.sum()),
                    "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean()),
                }

            for kind, task, N in specs:
                d = gather(task, "ood", N)
                a_te = accept(d, gate)
                if a_te is None:
                    del d
                    torch.cuda.empty_cache()
                    continue
                parent = d["parent"]
                s = fn(d["mean"], d["sigma"], d["truth"],
                       med=MED).cpu().numpy()
                grp = np.array([parent] * len(s))
                q = gated.q.get(parent, gated.q_pooled)
                half = q * _floor(d["sigma"], res["sigma_floor_frac"], MED)
                w = (half.flatten(1).norm(dim=1)
                     / d["truth"].flatten(1).norm(dim=1)).cpu().numpy()
                n_acc = int(a_te.sum())
                rec = {
                    "kind": kind, "parent": parent, "N": N,
                    "n_test": int(len(s)),
                    "n_accepted": n_acc,
                    "abstention": float(1.0 - a_te.mean()),
                    # the strict reading, in the same record, not a footnote
                    "marginal_ungated": cov_entry(ungated.covered(s, grp)),
                    "selective": cov_entry(gated.covered(s[a_te], grp[a_te])),
                    "measured": bool(n_acc >= args.min_accepted),
                    "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean()),
                    "rel_l2_accepted": (
                        float(rel_l2(d["mean"], d["truth"])[
                            torch.as_tensor(a_te, device=d["mean"].device)
                        ].mean()) if n_acc else None),
                    "width_rel_accepted": (float(np.median(w[a_te]))
                                           if n_acc else None),
                    "width_rel_all": float(np.median(w)),
                    # why the gate did or did not move this shard
                    "spearman_gate_vs_score": spearman(d["gate"][gate], s),
                    "gate_q50_over_tau": float(
                        np.median(d["gate"][gate]) / tau[gate][parent]),
                    "uses_ground_truth": gate in ORACLE_GATES,
                }
                out["shards"][f"{kind}/{task}/N{N}"] = rec
                del d
                torch.cuda.empty_cache()
            gout["per_score"][sname] = out
            hl = out["shards"]
            inband = sum(1 for k, r in hl.items()
                         if r["kind"] in COVARIATE_KINDS and r["measured"]
                         and 0.88 <= r["selective"]["coverage"] <= 0.92)
            print(f"[{gate}/{sname}] covariate shards in band (selective, "
                  f"n_acc>={args.min_accepted}): {inband}", flush=True)
        res["gates"][gate] = gout

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
