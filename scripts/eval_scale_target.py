"""H27: the ceiling of EVERY per-sample scalar modulation of sigma.

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/eval_scale_target.py --ckpt runs/u0/best.pt \
        --sigma-source het --out runs/starget_u0_het.json

H26 measured the ceiling of ONE scalar route -- sigma * (relresid/med)^gamma --
at 7/24 shards in band with an oracle exponent per family. codex proposed three
more scalar predictors (a spectral inverse-gain G(r), the Darcy diagonal
D_jj = N^2 * sum of face coefficients, a pinball-fit log-width model). Testing
them one at a time is one run per ceiling. This bounds all of them together.

Coverage on a shard is monotone decreasing in nothing and monotone INCREASING
in the scale applied to sigma, because the score is |mu-u| / (s*sigma + floor)
and the conformal quantile q is FROZEN at its calibration value. So for every
shard there is a scale s* with coverage exactly 0.90. Bisect for it.

s* is then the TARGET any scalar modulation has to reproduce, and the question
collapses to a regression diagnostic: how well does log s* correlate with
per-shard scalars a deployment can actually see, and how many shards land in
band if we fit the best monotone link from the best of them by
leave-one-shard-out?

That last count is the ceiling of the whole family
    "per-sample scalar modulation of sigma from a deployment-observable
     quantity, with ANY link function",
which contains H25, H26 and codex's routes as special cases.

NOTHING HERE IS CLAUSE-ELIGIBLE. s* is computed FROM THE TRUTH; it is an
oracle target, not a method. `clause_eligible: false` is written into the JSON
so no aggregator can quote it as a result. The conformal quantile is never
refit on shifted data.
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
from uqkit.equivar import predict_equivariant, reference_scale  # noqa: E402
from uqkit.metrics import binom_ci, rel_l2  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402

COVARIATE_KINDS = {"input_shift", "graded_rough"}
BAND = (0.88, 0.92)
FRAC = 0.05
TARGET = 0.90


def cov_entry(covered):
    c = np.asarray(covered, dtype=bool)
    k, n = int(c.sum()), int(len(c))
    lo, hi = binom_ci(k, n)
    return {"coverage": k / n, "n": n, "ci95": [lo, hi]}


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
    ap.add_argument("--equivariant", action="store_true",
                    help="wrap the prediction in F_eq(a) = s*F(a/s), the H17 "
                         "test-time restoration of the scale equivariance the "
                         "frozen input standardisation throws away. The "
                         "reference scale is the CALIBRATION median, so the "
                         "wrapper is near-identity in distribution.")
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
    print(f"families with a cheap apply: {in_tasks}", flush=True)

    def relresid(a, mean, parent):
        if parent not in sims:
            sims[parent] = PDE2DSimulator(parent, device=a.device)
        r = sims[parent].residual(a, mean)
        f = sims[parent].rhs(a)
        return (r.flatten(1).norm(dim=1)
                / f.flatten(1).norm(dim=1).clamp_min(1e-12))

    def predict_raw(blob):
        return predict_shard_single(model, blob, stats, args.device,
                                    sigma_source=args.sigma_source)

    # The reference scale is the median per-sample scale on the CALIBRATION
    # split, so s ~ 1 in distribution and the wrapper is near-identity there.
    # Computed before any evaluation shard is touched, exactly as
    # eval_width_tolerance.py does it.
    REF = {}
    if args.equivariant:
        for t in in_tasks:
            REF[t] = reference_scale(load_shard(args.root, t, "cal")["a"], t)

    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        parent = PARENT.get(task, task)
        if args.equivariant:
            m, s, t, a = predict_equivariant(predict_raw, blob, parent,
                                             REF[parent])
        else:
            m, s, t, a = predict_raw(blob)
        return {"mean": m, "sigma": s, "truth": t, "a": a,
                "rr": relresid(a, m, parent), "parent": parent, "task": task}

    cal = {t: gather(t, "cal") for t in in_tasks}
    RR_MED = {t: float(cal[t]["rr"].median()) for t in in_tasks}
    MED = float(torch.cat([cal[t]["sigma"].flatten()
                           for t in in_tasks]).median())
    FLOOR = FRAC * MED

    # the frozen calibration quantile, per family: the unmodulated base arm
    s_cal = np.concatenate([fn(cal[t]["mean"], cal[t]["sigma"],
                               cal[t]["truth"], med=MED).cpu().numpy()
                            for t in in_tasks])
    g_cal = np.concatenate([[t] * len(cal[t]["rr"]) for t in in_tasks])
    conf = GroupConformal(args.alpha).fit(s_cal, g_cal)

    def coverage_at(d, s):
        """Coverage with sigma scaled by the scalar `s`. q stays frozen.

        `med=MED` is passed unchanged so the additive floor stays at its frozen
        calibration value: the denominator is `s*sigma + 0.05*MED`, which is
        what a scalar modulation of sigma actually produces in this kit. That
        makes coverage monotone increasing in `s` but bounded BELOW by the
        coverage the floor alone provides -- so `s*` need not exist, and a
        shard where it does not is a shard no scalar modulation can fix
        downward. That case is reported, not clamped.
        """
        sc = fn(d["mean"], d["sigma"] * s, d["truth"], med=MED).cpu().numpy()
        return conf.covered(sc, np.array([d["parent"]] * len(sc))).mean()

    def find_sstar(d, lo=1e-4, hi=1e5, iters=60):
        """Bisect the scale giving coverage == 0.90. Monotone in s."""
        c_lo, c_hi = coverage_at(d, lo), coverage_at(d, hi)
        if c_lo > TARGET:
            return {"s_star": None, "reason": "coverage above 0.90 even at "
                    f"s={lo:g} (cov {c_lo:.4f}) -- no scale can reduce it",
                    "cov_lo": float(c_lo), "cov_hi": float(c_hi)}
        if c_hi < TARGET:
            return {"s_star": None, "reason": "coverage below 0.90 even at "
                    f"s={hi:g} (cov {c_hi:.4f}) -- sigma's SHAPE is wrong, "
                    "not just its scale", "cov_lo": float(c_lo),
                    "cov_hi": float(c_hi)}
        if c_lo > c_hi:
            raise SystemExit("coverage is not increasing in the scale; the "
                             "bisection's monotonicity assumption is broken "
                             f"(cov {c_lo:.4f} at s={lo:g} vs {c_hi:.4f} at "
                             f"s={hi:g}) and no s* here is interpretable")
        for _ in range(iters):
            mid = float(np.sqrt(lo * hi))          # bisect in log space
            if coverage_at(d, mid) < TARGET:
                lo = mid
            else:
                hi = mid
        s = float(np.sqrt(lo * hi))
        return {"s_star": s, "cov_at_s_star": float(coverage_at(d, s)),
                "reason": None, "cov_lo": float(c_lo), "cov_hi": float(c_hi)}

    def find_cross(d, target, lo=1e-4, hi=1e5, iters=60):
        """The scale at which coverage crosses `target`. None if it never does."""
        if coverage_at(d, lo) > target or coverage_at(d, hi) < target:
            return None
        for _ in range(iters):
            mid = float(np.sqrt(lo * hi))
            if coverage_at(d, mid) < target:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))

    def sband(d):
        """The multiplicative window on the sigma scale keeping coverage in
        [0.88, 0.92].

        This is what turns `s*` from a target into a PASS/FAIL criterion for a
        predicted scale, so it is what the link-function ceiling is scored
        against. The width ratio is reported separately because a wide window
        is an easy shard and a narrow one is a hard shard independently of
        where `s*` sits.
        """
        lo = find_cross(d, BAND[0])
        hi = find_cross(d, BAND[1])
        return {"s_band": [lo, hi],
                "s_band_width_ratio": (hi / lo) if (lo and hi) else None}

    def observables(d):
        """Per-shard scalars a DEPLOYMENT can compute: no truth, no solve.

        `diag_mean` is codex's Darcy diagonal, N^2 * sum of the four face
        coefficients, approximated here by 4*|k| since the face averages are
        inside the simulator. It is MEANINGFUL ONLY FOR DARCY: for the other
        two families channel 0 of `a` is not a coefficient field, so the
        column is recorded for uniformity and must be read per-family.
        """
        sig = d["sigma"]
        a = d["a"]
        k = a[:, :1]
        N = a.shape[-1]
        return {
            "relresid_ratio_med": float((d["rr"] / RR_MED[d["parent"]]).median()),
            "relresid_ratio_p90": float(torch.quantile(
                d["rr"] / RR_MED[d["parent"]], 0.90)),
            "sigma_med": float(sig.median()),
            "sigma_fieldmax_med": float(sig.flatten(1).max(dim=1)
                                        .values.median()),
            "sigma_ratio_to_cal": float(sig.median() / MED),
            "sigma_cv": float(sig.flatten(1).std(dim=1).median()
                              / sig.flatten(1).mean(dim=1).median()),
            "input_amp": float(a.flatten(1).abs().max(dim=1).values.median()),
            "input_rough": float((a[..., 1:, :] - a[..., :-1, :]).abs()
                                 .flatten(1).mean(dim=1).median()),
            "diag_mean": float((N ** 2 * 4 * k.abs()).flatten(1)
                               .mean(dim=1).median()),
            "pred_fieldmax_med": float(d["mean"].flatten(1).abs()
                                       .max(dim=1).values.median()),
        }

    res = {"alpha": args.alpha, "score": args.score, "ckpt": args.ckpt,
           "seed": ck["args"].get("seed"), "sigma_source": args.sigma_source,
           "families": in_tasks, "band": list(BAND), "target": TARGET,
           "clause_eligible": False,
           "equivariant": bool(args.equivariant),
           "reference_scale": {k: float(v) for k, v in REF.items()},
           "note": ("s* is the sigma scale making a shard's coverage exactly "
                    "0.90, found by bisection with the calibration quantile "
                    "FROZEN. It is computed from the truth, so it is an "
                    "ORACLE TARGET and no number here may be quoted as a "
                    "method result. Its purpose is to bound the ceiling of "
                    "every per-sample scalar modulation of sigma at once."),
           "sigma_floor_median": MED, "floor_abs": FLOOR,
           "relresid_median_cal": RR_MED,
           "q_cal": {k: float(v) for k, v in conf.q.items()},
           "in_dist": {}, "shards": {}}

    for t in in_tasks:
        d = gather(t, "test")
        res["in_dist"][t] = {"base": cov_entry(conf.covered(
            fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy(),
            np.array([t] * len(d["rr"])))), **find_sstar(d), **sband(d),
            "observables": observables(d)}
        print(f"  in-dist {t:10s} base "
              f"{res['in_dist'][t]['base']['coverage']:.4f}  s* "
              f"{res['in_dist'][t]['s_star']}", flush=True)
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
            base = cov_entry(conf.covered(
                fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy(),
                np.array([parent] * len(d["rr"]))))
            rec = {"kind": kind, "parent": parent, "base": base,
                   "rel_l2_mean": float(rel_l2(d["mean"], d["truth"]).mean()),
                   **find_sstar(d), **sband(d),
                   "observables": observables(d)}
            res["shards"][f"{kind}/{task}/N64"] = rec
            ss = rec["s_star"]
            print(f"  {kind}/{task:22s} base {base['coverage']:.4f}  s* "
                  + (f"{ss:.4g}" if ss else f"NONE ({rec['reason'][:40]})"),
                  flush=True)
            del d
            torch.cuda.empty_cache()

    res["n_shards"] = len(res["shards"])
    res["n_sstar_found"] = sum(1 for r in res["shards"].values()
                               if r["s_star"] is not None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"s* found on {res['n_sstar_found']}/{res['n_shards']} shards")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
