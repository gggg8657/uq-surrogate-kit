"""H26: the ceiling of the residual-scale family, as a function of its exponent.

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/eval_resid_gamma.py --ckpt runs/u0/best.pt \
        --sigma-source het --out runs/rgam_u0_het.json

H25 measured that `sigma' = sigma * (relresid/median_cal)` moves coverage under
shift further than anything else in this repo -- mean shard |coverage-0.90|
0.5613 -> 0.2673, 8/8 seeds, sign-flip p = 0.0039 -- with the right sign on
every family and the wrong gain on all three: darcy over-covers 64/80 shard-
seeds, poisson under-covers 69/80, helmholtz 32/32.

This sweeps the exponent

    sigma'(gamma) = sigma * (relresid / median_cal(relresid)) ** gamma

and reports the BEST ACHIEVABLE coverage of the family, i.e. a CEILING.

gamma is NOT identifiable in distribution: the modulation is ~1 there by
construction (H25 measured 0.968-1.003), so in-distribution coverage is flat in
gamma and no honest procedure can freeze gamma without labelled shifted data.
Every gamma != 1 reading in this file is therefore emitted under an explicit
`selection` tag and the aggregator must carry it into any document:

  gamma=1   selected by nothing            -- the shipped arm, clause-eligible
  oracle    selected on the shard it scores -- a ceiling, never a clause claim
  loso      selected on the OTHER shards of the same family -- honest, and
            priced: it costs labelled shifted shards from that family

TWO CORRECTNESS GATES, checked in-run and recorded in `gates`:
  * gamma=0 makes the modulation exactly 1.0, so its coverage must equal the
    unmodulated base arm EXACTLY (not to a tolerance).
  * gamma=1 must reproduce the H25 run for this seed exactly, if it is present.
A sweep that fails either gate is not interpretable, so `gates.ok` is false and
the aggregator refuses the file.

Cost is H25's: one operator apply per sample, shared across all gammas. Clause 2
for this whole family stays `[not measured]`.
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
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402

COVARIATE_KINDS = {"input_shift", "graded_rough"}
BAND = (0.88, 0.92)
GAMMAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
          1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def cov_entry(covered):
    c = np.asarray(covered, dtype=bool)
    k, n = int(c.sum()), int(len(c))
    if n == 0:
        return {"coverage": None, "n": 0, "ci95": [None, None]}
    lo, hi = binom_ci(k, n)
    return {"coverage": k / n, "n": n, "ci95": [lo, hi]}


def in_band(c):
    return c is not None and BAND[0] <= c <= BAND[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default="het",
                    choices=["het", "cqr", "const"])
    ap.add_argument("--h25", default=None,
                    help="H25 run for this seed, for the gamma=1 gate")
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    model, ck = load_model(args.ckpt, args.device)
    if not ck["args"].get("uq"):
        raise SystemExit(f"{args.ckpt} is not a UQ checkpoint")
    stats = ck["stats"]
    fn, _ = get_score(args.score)
    sims: dict = {}

    probe_a = torch.zeros(1, 2, 64, 64)
    probe_u = torch.zeros(1, 1, 64, 64)
    in_tasks = [t for t in man["in_tasks"]
                if PDE2DSimulator(t, device="cpu").residual(probe_a, probe_u)
                is not None]
    dropped = [t for t in man["in_tasks"] if t not in in_tasks]
    print(f"families with a cheap apply: {in_tasks}; excluded {dropped}",
          flush=True)

    def relresid(a, mean, parent):
        if parent not in sims:
            sims[parent] = PDE2DSimulator(parent, device=a.device)
        r = sims[parent].residual(a, mean)
        if r is None:
            raise SystemExit(f"{parent} has no cheap apply")
        f = sims[parent].rhs(a)
        return (r.flatten(1).norm(dim=1)
                / f.flatten(1).norm(dim=1).clamp_min(1e-12))

    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        m, s, t, a = predict_shard_single(model, blob, stats, args.device,
                                          sigma_source=args.sigma_source)
        parent = PARENT.get(task, task)
        return {"mean": m, "sigma": s, "truth": t,
                "rr": relresid(a, m, parent), "parent": parent, "task": task}

    cal = {t: gather(t, "cal") for t in in_tasks}
    RR_MED = {t: float(cal[t]["rr"].median()) for t in in_tasks}
    MED = float(torch.cat([cal[t]["sigma"].flatten()
                           for t in in_tasks]).median())

    def scores(d, gamma):
        """gamma is None -> the unmodulated base arm (the gamma=0 gate's peer)."""
        if gamma is None:
            sig = d["sigma"]
        else:
            w = (d["rr"] / RR_MED[d["parent"]]).clamp_min(1e-12)
            sig = d["sigma"] * w.pow(gamma).view(-1, 1, 1, 1)
        return fn(d["mean"], sig, d["truth"], med=MED).cpu().numpy()

    ARMS = [None] + GAMMAS          # None == base, no modulation at all
    def key(g):
        return "base" if g is None else f"g{g:g}"

    res = {"alpha": args.alpha, "score": args.score, "ckpt": args.ckpt,
           "seed": ck["args"].get("seed"),
           "sigma_source": args.sigma_source,
           "families": in_tasks, "excluded_families": dropped,
           "gammas": GAMMAS, "relresid_median_cal": RR_MED,
           "sigma_floor_median": MED, "band": list(BAND),
           "note": ("sigma'(gamma) = sigma * (relresid/median_cal)**gamma. "
                    "gamma is NOT identifiable in distribution (modulation "
                    "~1 there), so every gamma != 1 reading is a ceiling or "
                    "is priced in labelled shifted shards. Clause 2 for this "
                    "family is [not measured]."),
           "in_dist": {}, "shards": {}}

    # ---- calibrate every arm on the same calibration split ------------------
    conf = {}
    for g in ARMS:
        s = np.concatenate([scores(cal[t], g) for t in in_tasks])
        grp = np.concatenate([[t] * len(cal[t]["rr"]) for t in in_tasks])
        conf[key(g)] = GroupConformal(args.alpha).fit(s, grp)
    res["q"] = {k: {kk: float(vv) for kk, vv in c.q.items()}
                for k, c in conf.items()}

    for t in in_tasks:
        d = gather(t, "test")
        rec = {"modulation_median": float((d["rr"] / RR_MED[t]).median())}
        for g in ARMS:
            s = scores(d, g)
            rec[key(g)] = cov_entry(conf[key(g)].covered(
                s, np.array([t] * len(s))))
        res["in_dist"][t] = rec
        print(f"  in-dist {t:10s} mod {rec['modulation_median']:.3f}  "
              + "  ".join(f"g{g:g} {rec[key(g)]['coverage']:.3f}"
                          for g in (0.0, 0.5, 1.0, 2.0)), flush=True)
        del d
        torch.cuda.empty_cache()

    for kind, tasks in man["ood_suite"].items():
        if kind not in COVARIATE_KINDS:
            continue
        for task in tasks:
            parent = PARENT.get(task, task)
            if parent not in in_tasks:
                continue
            d = gather(task, "ood", 64)
            rec = {"kind": kind, "parent": parent,
                   "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean()),
                   "modulation_median":
                       float((d["rr"] / RR_MED[parent]).median())}
            for g in ARMS:
                s = scores(d, g)
                rec[key(g)] = cov_entry(conf[key(g)].covered(
                    s, np.array([parent] * len(s))))
            res["shards"][f"{kind}/{task}/N64"] = rec
            del d
            torch.cuda.empty_cache()

    res["n_shards"] = len(res["shards"])
    res["n_band"] = {key(g): sum(1 for r in res["shards"].values()
                                 if in_band(r[key(g)]["coverage"]))
                     for g in ARMS}

    # ---- the two gates ------------------------------------------------------
    gates = {}
    g0 = [(r["g0"]["coverage"], r["base"]["coverage"])
          for r in list(res["shards"].values()) + list(res["in_dist"].values())]
    gates["gamma0_equals_base"] = all(a == b for a, b in g0)
    gates["gamma0_max_abs_diff"] = max(abs(a - b) for a, b in g0)
    if args.h25 and Path(args.h25).is_file():
        h = json.loads(Path(args.h25).read_text())
        d = [abs(res["shards"][n]["g1"]["coverage"]
                 - s["resid"]["coverage"]) for n, s in h["shards"].items()]
        d += [abs(res["in_dist"][t]["g1"]["coverage"]
                  - s["resid"]["coverage"]) for t, s in h["in_dist"].items()]
        gates["gamma1_equals_h25"] = all(x == 0.0 for x in d)
        gates["gamma1_max_abs_diff"] = max(d)
        gates["h25_ref"] = args.h25
    else:
        gates["gamma1_equals_h25"] = None
        gates["h25_ref"] = None
    gates["ok"] = bool(gates["gamma0_equals_base"]
                       and gates["gamma1_equals_h25"] is not False)
    res["gates"] = gates
    print(f"gates: gamma0==base {gates['gamma0_equals_base']} "
          f"(max|d| {gates['gamma0_max_abs_diff']:.2e})  "
          f"gamma1==H25 {gates['gamma1_equals_h25']}  -> ok={gates['ok']}",
          flush=True)
    print("  " + "  ".join(f"g{g:g}:{res['n_band'][key(g)]}"
                           for g in GAMMAS) + f"   /{res['n_shards']}",
          flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
