"""H15: fit the width predictor on development shifts, score the held-out 32.

    CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/pdeno/bin/python \
        scripts/fit_scale.py --ckpt runs/u0/best.pt --sigma-source het \
        --out runs/scale_u0_het.json

Protocol, registered in `critique_log.md` (H15) before this ran:

* h is fitted on the **development** shift suite (`uqkit.devshift`, seed block
  40000+, values disjoint from every evaluation shard, checked by
  `assert_disjoint()`) plus one half of the in-distribution `cal` split. The
  dev shards are generated in memory and never written to disk.
* q is calibrated on the **other half** of `cal`, so no row is used both to fit
  the difficulty model and to calibrate the quantile. The ungated `group`
  baseline is recomputed on that same half, so the comparison is like-for-like
  rather than against a baseline with twice the calibration data.
* The 32 evaluation shards are untouched by the fit -- not for coefficients,
  not for feature standardization, not for anything.
* **Two readings, both reported.** `--fold all` is the generous one: h has seen
  every shift mechanism at strengths that bracket the evaluation values, so it
  is interpolating. `--fold <mechanism>` holds an entire mechanism out of the
  fit, and each evaluation shard is scored only by the fold that never saw its
  mechanism. The leave-one-mechanism-out reading is the headline.
* **The quantile on T = S/h is per family**, exactly as the `group` baseline
  it is compared against is per family. A single pooled quantile cannot serve
  five families whose ungated quantiles span 2.89 (helmholtz) to 21.46 (darcy);
  the first run of this script used one and it destroyed the in-distribution
  pass (helmholtz 0.588, advdiff 1.000) while flattering the graded ladder.
  That was a bug in the comparison, not a result. The pooled reading is still
  reported, labelled, because it is the strict one.
* `--equivariant` (H18) composes this with the H17 test-time wrapper. The
  composition is the only difference between the two arms -- same dev suite,
  same seed block, same folds, same per-family quantile -- so they pair by
  checkpoint and the sign-flip test against the H15 arm is exact.
* Width travels with every coverage. h can buy coverage by inflating every
  interval, which would be H13's abstention trap in a third disguise, so
  `width_mult_median` and its ratio to the ungated interval are in every
  record and a shard that enters the band on a >3x wider interval is flagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit import devshift as D  # noqa: E402
from uqkit.conformal import GroupConformal, _floor, get_score  # noqa: E402
from uqkit.equivar import (predict_equivariant,  # noqa: E402
                           reference_scale)
from uqkit.features import spectral_features  # noqa: E402
from uqkit.metrics import binom_ci, rel_l2  # noqa: E402
from uqkit.ood import consistency_score  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.scale import (GroupScaleConformal,  # noqa: E402
                         PerFamilyScale,
                         QuantileScale, ScaleConformal)
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

COVARIATE_KINDS = {"input_shift", "graded_rough"}
BAND = (0.88, 0.92)
WIDTH_FLAG = 3.0


def cov_entry(covered):
    c = np.asarray(covered, dtype=bool)
    k, n = int(c.sum()), int(len(c))
    if n == 0:
        return {"coverage": None, "n": 0, "ci95": [None, None]}
    lo, hi = binom_ci(k, n)
    return {"coverage": k / n, "n": n, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--dev-n", type=int, default=256,
                    help="samples per development shard, generated in memory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sigma-source", default=None,
                    choices=["het", "cqr", "const"])
    ap.add_argument("--drop-features", default="",
                    help="H19: comma-separated feature names removed from z "
                         "before fitting. Used to ablate the two amplitude "
                         "features of input channel 0 (`a_spec9`, `a_spec10`) "
                         "and so separate 'nothing left to learn' from 'lost "
                         "dynamic range in the fitting target'. The names are "
                         "checked against FEAT_NAMES and an unknown one is an "
                         "error, not a silent no-op.")
    ap.add_argument("--restrict-families", default="",
                    help="comma-separated families to keep. Exists so the H22 "
                         "residual arm has a CONTROL: --residual-features "
                         "forces a restriction to families with a cheap "
                         "operator apply, and comparing that against the "
                         "5-family H15 arm would change two things at once "
                         "(the features and the fitting set). Running this "
                         "flag alone reproduces the restriction without the "
                         "features, so the pair differs in exactly the two "
                         "residual columns.")
    ap.add_argument("--residual-features", action="store_true",
                    help="H22: add two scalars per sample derived from ONE "
                         "residual evaluation r = L(mu) - f: log10 of the "
                         "dimensionless consistency score ||r||/(||L mu||+||f||) "
                         "and log10 of ||r||/||f||. Both are exactly invariant "
                         "to rescaling the linear channel, so any gain they "
                         "produce CANNOT be the amplitude effect that H19 "
                         "showed carries every other arm. "
                         "`PDE2DSimulator.residual` is None for the two "
                         "time-stepped families, so this restricts the "
                         "evaluation to poisson/helmholtz/darcy -- 24 of the "
                         "32 covariate shards. The baseline must be re-read on "
                         "the same 24; `scripts/agg_scale.py` does that from "
                         "per-shard cells already on disk.")
    ap.add_argument("--nonneg-features", default="",
                    help="H24: comma-separated features whose width "
                         "coefficient is constrained to be >= 0 by projection "
                         "after each step. Motivated by measurement, not "
                         "taste: H23 found the residual predicts the "
                         "conformity score with a POSITIVE marginal rank "
                         "correlation on 8/8 seeds while H22's unconstrained "
                         "fit gave it a negative partial coefficient on 8/8, "
                         "which is the signature of suppression. Names are "
                         "checked against the feature vector.")
    ap.add_argument("--per-family-h", action="store_true",
                    help="H19: fit a separate h per family instead of one "
                         "pooled h with family one-hots. H18 measured that "
                         "the pooled h is driven by whichever rows carry the "
                         "most pinball loss mass: removing the amplitude "
                         "effect (H17) changed what h learned on the Darcy "
                         "axis and cost 4/32 -> 1/32. A per-family fit makes "
                         "that contamination impossible by construction.")
    ap.add_argument("--equivariant", action="store_true",
                    help="H18: fit and evaluate h on top of the H17 test-time "
                         "scale-equivariant predictor F_eq(a) = s*F(a/s). "
                         "Composition only -- every other element of the H15 "
                         "protocol is unchanged, so the arms are paired by "
                         "checkpoint and the sign-flip test is exact. "
                         "`darcy_amp2` remains the control that must not "
                         "improve, since Darcy's channel 0 is log-permeability.")
    args = ap.parse_args()
    single = args.sigma_source is not None
    if single and len(args.ckpt) != 1:
        ap.error("--sigma-source is a single-network mode; pass one checkpoint")

    D.assert_disjoint()
    dev_specs = D.register()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    # H22: which families have a cheap operator apply is a property of the
    # problem, not a choice. Ask the simulator rather than hardcoding a list,
    # so the restriction cannot drift away from what `residual()` can do.
    if args.restrict_families:
        want = [t.strip() for t in args.restrict_families.split(",")
                if t.strip()]
        unknown = [t for t in want if t not in in_tasks]
        if unknown:
            raise SystemExit(f"--restrict-families: unknown {unknown}; "
                             f"have {in_tasks}")
        print(f"--restrict-families: {want}", flush=True)
        in_tasks = want
    if args.residual_features:
        import torch as _t
        probe = _t.zeros(1, 2, 64, 64)
        keep = []
        for t in in_tasks:
            sim = PDE2DSimulator(t, device="cpu")
            if sim.residual(probe, _t.zeros(1, 1, 64, 64)) is not None:
                keep.append(t)
        dropped = [t for t in in_tasks if t not in keep]
        if not keep:
            raise SystemExit("--residual-features: no family has a cheap apply")
        print(f"--residual-features: restricted to {keep}; "
              f"{dropped} have no cheap operator apply, so they are EXCLUDED "
              f"rather than fed a placeholder. The baseline must be re-read on "
              f"the same subset.", flush=True)
        in_tasks = keep
    if single:
        model, ck = load_model(args.ckpt[0], args.device)
        if not ck["args"].get("uq"):
            raise SystemExit(f"{args.ckpt[0]} is not a UQ checkpoint")
        models = [model]
    else:
        models, ck = load_members(args.ckpt, args.device)
    stats = ck["stats"]
    fn, _ = get_score(args.score)
    FAM_IDX = {t: i for i, t in enumerate(in_tasks)}

    def predict_raw(blob):
        if single:
            return predict_shard_single(models[0], blob, stats, args.device,
                                        sigma_source=args.sigma_source)
        return predict_shard(models, blob, stats, args.device)

    # REF is the median per-sample scale of the linear channels on the
    # CALIBRATION split, so s ~ 1 in distribution and the wrapper is
    # near-identity there. Filled below, before any evaluation shard is read.
    REF = {}

    def predict(blob):
        """The prediction path, optionally wrapped in scale equivariance.

        Only the prediction is rescaled: `truth` and the returned inputs are
        the originals, so every score, calibrator and feature downstream sees
        exactly the data it saw in the H15 arm.
        """
        if not args.equivariant:
            return predict_raw(blob)
        parent = PARENT.get(blob["task"], blob["task"])
        return predict_equivariant(predict_raw, blob, parent, REF[parent])

    # ---- features: strictly deployment-observable ---------------------------
    FEAT_NAMES = None
    #: column indices kept after --drop-features, fixed on the first call so
    #: every later call uses an identical layout
    KEEP: list[int] = []

    _sims: dict = {}

    def resid_feats(a, mean, parent):
        """(n, 2) log residual features, or None where there is no cheap apply.

        One `residual()` call; `L(mu)` is recovered as `r + f` rather than by a
        second apply, so this costs a single evaluation of the operator and no
        solve. Ground truth is never touched.
        """
        if parent not in _sims:
            _sims[parent] = PDE2DSimulator(parent, device=a.device)
        sim = _sims[parent]
        r = sim.residual(a, mean)
        if r is None:
            return None
        f = sim.rhs(a)
        eps = 1e-12
        c = consistency_score(r, r + f, f)
        rn = (r.flatten(1).norm(dim=1)
              / f.flatten(1).norm(dim=1).clamp_min(eps))
        return torch.stack([torch.log10(c.clamp_min(eps)),
                            torch.log10(rn.clamp_min(eps))], dim=-1)

    def features(a, mean, sigma, parent):
        """(n, d) from the input, the prediction and its spread. No truth."""
        nonlocal FEAT_NAMES
        fa = spectral_features(a)
        fm = spectral_features(mean)
        eps = 1e-12
        sn = sigma.flatten(1).norm(dim=1)
        mn = mean.flatten(1).norm(dim=1)
        scal = torch.stack([
            torch.log10(sn / mn.clamp_min(eps) + eps),
            torch.log10(sn + eps),
            torch.log10(sigma.flatten(1).max(dim=1).values + eps),
            torch.log10(sigma.flatten(1).median(dim=1).values + eps),
            torch.log10(mn + eps),
            torch.log10(mean.flatten(1).abs().max(dim=1).values + eps),
        ], dim=-1)
        oh = torch.zeros(len(a), len(in_tasks), device=a.device)
        oh[:, FAM_IDX[parent]] = 1.0
        parts, extra_names = [fa, fm, scal, oh], []
        if args.residual_features:
            rf = resid_feats(a, mean, parent)
            if rf is None:
                raise SystemExit(
                    f"--residual-features: {parent} has no cheap operator "
                    f"apply, so it must be excluded from this arm rather "
                    f"than fed zeros. This is a bug in the shard filter.")
            parts.append(rf)
            extra_names = ["log_consist", "log_resid"]
        z = torch.cat(parts, dim=1)
        if FEAT_NAMES is None:
            FEAT_NAMES = ([f"a_spec{i}" for i in range(fa.shape[1])]
                          + [f"mu_spec{i}" for i in range(fm.shape[1])]
                          + ["log_sigrel", "log_signorm", "log_sigmax",
                             "log_sigmed", "log_munorm", "log_mumax"]
                          + [f"fam_{t}" for t in in_tasks] + extra_names)
            drop = [d for d in args.drop_features.split(",") if d.strip()]
            unknown = [d for d in drop if d not in FEAT_NAMES]
            if unknown:
                raise SystemExit(f"--drop-features names not in the feature "
                                 f"vector: {unknown}; have {FEAT_NAMES}")
            KEEP.extend(i for i, n in enumerate(FEAT_NAMES) if n not in drop)
            FEAT_NAMES = [n for n in FEAT_NAMES if n not in drop]
        return z.double().cpu()[:, KEEP]

    # ---- in-distribution calibration, split in two --------------------------
    import time as _t
    _t0 = _t.time()

    def _tick(msg):
        print(f"  [{_t.time() - _t0:7.1f}s] {msg}", flush=True)

    cal_rows = {}
    med_parts = []
    _tick("loading in-distribution calibration shards")
    cal_blobs = {t: load_shard(args.root, t, "cal") for t in in_tasks}
    if args.equivariant:
        # computed from calibration inputs alone, before anything else is read
        for t in in_tasks:
            REF[t] = reference_scale(cal_blobs[t]["a"], t)
        _tick(f"equivariant REF from cal only: "
              + ", ".join(f"{t}={REF[t]:.4g}" for t in in_tasks))
    for t in in_tasks:
        m, s, tr, a = predict(cal_blobs[t])
        med_parts.append(s.flatten())
        cal_rows[t] = {"mean": m, "sigma": s, "truth": tr, "a": a}
    del cal_blobs
    MED = float(torch.cat(med_parts).median())
    del med_parts

    _tick("cal predictions done; splitting cal in two")
    rng = np.random.default_rng(0)
    fit_z, fit_s, q_scores, q_groups, fit_fam = [], [], [], [], []
    for t in in_tasks:
        d = cal_rows[t]
        sc = fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy()
        z = features(d["a"], d["mean"], d["sigma"], t)
        n = len(sc)
        perm = rng.permutation(n)
        half = n // 2
        i_fit, i_q = perm[:half], perm[half:]
        fit_z.append(z[i_fit])
        fit_s.append(sc[i_fit])
        fit_fam.append(np.array([t] * len(i_fit)))
        q_scores.append(sc[i_q])
        q_groups.append(np.array([t] * len(i_q)))
        cal_rows[t] = {"z_q": z[i_q], "s_q": sc[i_q]}   # keep only what is used

    # ---- development shards, generated in memory and reduced to rows --------
    _tick(f"generating {len(dev_specs)} development shards in memory")
    dev = {}
    for i, (task, parent, mech, _over) in enumerate(dev_specs):
        if parent not in in_tasks:
            continue    # H22: family excluded for lack of a cheap apply
        if i % 5 == 0:
            _tick(f"  dev shard {i}/{len(dev_specs)} {task}")
        a, u, _extra = D.generate(task, args.dev_n, N=64,
                                  device=args.device, index=i)
        blob = {"a": a.cpu(), "u": u.cpu(), "task": task, "N": 64,
                "split": "dev", "parent": parent}
        m, s, tr, ain = predict(blob)
        sc = fn(m, s, tr, med=MED).cpu().numpy()
        dev.setdefault(mech, {"z": [], "s": [], "tasks": [], "fam": []})
        dev[mech]["z"].append(features(ain, m, s, parent))
        dev[mech]["s"].append(sc)
        dev[mech]["fam"].append(np.array([parent] * len(sc)))
        dev[mech]["tasks"].append(task)
        del a, u, m, s, tr, ain, blob
        torch.cuda.empty_cache()
    for mech in dev:
        dev[mech]["z"] = torch.cat(dev[mech]["z"])
        dev[mech]["s"] = np.concatenate(dev[mech]["s"])
        dev[mech]["fam"] = np.concatenate(dev[mech]["fam"])

    _tick("development suite reduced to rows")
    cal_fit_z, cal_fit_s = torch.cat(fit_z), np.concatenate(fit_s)
    cal_fit_fam = np.concatenate(fit_fam)
    q_all = np.concatenate(q_scores)
    q_grp = np.concatenate(q_groups)
    # the like-for-like ungated baseline: same calibration half, same score
    ungated = GroupConformal(args.alpha).fit(q_all, q_grp)

    res = {"alpha": args.alpha, "score": args.score, "ckpt": args.ckpt,
           "seed": ck["args"].get("seed"),
           "sigma_source": args.sigma_source or "ensemble_spread",
           "forward_passes_per_interval": 1 if single else len(models),
           "sigma_floor_median": MED, "sigma_floor_frac": 0.05,
           "equivariant": bool(args.equivariant),
           "per_family_h": bool(args.per_family_h),
           "residual_features": bool(args.residual_features),
           # H24's defining parameter. It was omitted from the first eight
           # runs, so those files could only be identified by their
           # filename; a run JSON has to say what protocol produced it.
           "nonneg_features": [f for f in args.nonneg_features.split(",")
                               if f.strip()],
           # which families this arm ran on. With --residual-features the set
           # is FORCED by whether the simulator has a cheap operator apply, so
           # it is recorded per run: a 24-shard arm and a 32-shard arm are not
           # comparable and the JSON has to say which one it is.
           "families": list(in_tasks),
           "dropped_features": [d for d in args.drop_features.split(",")
                                if d.strip()],
           "n_features": len(KEEP),
           "equivariant_ref": {k: float(v) for k, v in REF.items()},
           "dev": {"n_shards": len(dev_specs), "n_per_shard": args.dev_n,
                   "seed_base": D.SEED_BASE,
                   "by_mechanism": {k: {"n_rows": int(len(v["s"])),
                                        "n_shards": len(v["tasks"])}
                                    for k, v in dev.items()},
                   "disjoint_checked": True},
           "n_cal_fit": int(len(cal_fit_s)), "n_cal_q": int(len(q_all)),
           "band": list(BAND), "width_flag": WIDTH_FLAG,
           "q_ungated": {k: float(v) for k, v in ungated.q.items()},
           "folds": {}}

    # ---- the evaluation shards ---------------------------------------------
    specs = []
    for kind, tasks in man["ood_suite"].items():
        if kind in COVARIATE_KINDS:
            specs += [(kind, t, 64) for t in tasks
                      if PARENT.get(t, t) in in_tasks]

    # recorded only now that `specs` exists: a 24-shard arm and a 32-shard arm
    # are not comparable, so the count travels in the JSON with the arm.
    res["n_covariate_shards_evaluated"] = len(specs)
    _tick(f"scoring {len(specs)} evaluation shards")
    eval_cache = {}
    for kind, task, N in specs:
        blob = load_shard(args.root, task, "ood", N)
        m, s, tr, a = predict(blob)
        parent = PARENT.get(task, task)
        eval_cache[f"{kind}/{task}/N{N}"] = {
            "kind": kind, "task": task, "parent": parent, "N": N,
            "z": features(a, m, s, parent),
            "s": fn(m, s, tr, med=MED).cpu().numpy(),
            "rel_l2": float(rel_l2(m, tr).mean()),
            "mech": D.eval_mechanism(task),
            # the ungated interval, for the width ratio
            "w_ungated": (_floor(s, res["sigma_floor_frac"], MED)
                          .flatten(1).norm(dim=1)
                          / tr.flatten(1).norm(dim=1)).cpu().numpy(),
        }
        del m, s, tr, a, blob
        torch.cuda.empty_cache()

    _tick("evaluation shards cached; fitting h")
    # `insample_leak` fits h on the evaluation shards themselves. It is NOT a
    # result -- it is the in-sample ceiling of this feature set and model
    # class, and its only job is to separate "h cannot fit the shift" from "h
    # cannot generalize to an unseen mechanism". Flagged wherever it appears
    # and excluded from the headline by construction.
    LEAK_FOLD = "insample_leak"
    FOLDS = ["all"] + list(D.MECHANISMS) + [LEAK_FOLD]
    for fold in FOLDS:
        leak = fold == LEAK_FOLD
        mechs = [k for k in dev if fold in ("all", LEAK_FOLD) or k != fold]
        zs = [cal_fit_z] + [dev[k]["z"] for k in mechs]
        ss = [cal_fit_s] + [dev[k]["s"] for k in mechs]
        if leak:
            zs += [e["z"] for e in eval_cache.values()]
            ss += [e["s"] for e in eval_cache.values()]
        z = torch.cat(zs)
        s = np.concatenate(ss)
        fam = np.concatenate([cal_fit_fam] + [dev[k]["fam"] for k in mechs]
                             + ([np.concatenate(
                                 [[e["parent"]] * len(e["s"])
                                  for e in eval_cache.values()])]
                                if leak else []))
        _tick(f"  fold {fold}: fitting h on {len(s)} rows x {z.shape[1]} "
              f"feats{' per family' if args.per_family_h else ''}")
        nn_idx = []
        if args.nonneg_features:
            want = [t.strip() for t in args.nonneg_features.split(",")
                    if t.strip()]
            unknown = [t for t in want if t not in FEAT_NAMES]
            if unknown:
                raise SystemExit(f"--nonneg-features not in the feature "
                                 f"vector: {unknown}; have {FEAT_NAMES}")
            nn_idx = [FEAT_NAMES.index(t) for t in want]
        if args.per_family_h:
            h = PerFamilyScale(alpha=args.alpha, seed=0,
                               nonneg=nn_idx).fit(z, s, fam)
        else:
            h = QuantileScale(alpha=args.alpha, seed=0,
                              nonneg=nn_idx).fit(z, s)
        _tick(f"  fold {fold}: h fitted")
        # q on the untouched calibration half, one quantile PER FAMILY -- the
        # calibrator the `group` baseline uses. Pooled is recorded beside it.
        def H(zz, parent):
            """h(z) for rows all belonging to `parent`. One call shape for
            both the pooled and the per-family model, so no call site has to
            know which one is in use."""
            return (h(zz, np.array([parent] * len(zz)))
                    if args.per_family_h else h(zz)).cpu().numpy()

        cal_h = np.concatenate([H(cal_rows[t]["z_q"], t) for t in in_tasks])
        cal_s = np.concatenate([cal_rows[t]["s_q"] for t in in_tasks])
        cal_g = np.concatenate([[t] * len(cal_rows[t]["s_q"])
                                for t in in_tasks])
        sc_conf = GroupScaleConformal(args.alpha).fit(cal_s, cal_h, cal_g)
        pooled_conf = ScaleConformal(args.alpha).fit(cal_s, cal_h)

        out = {"held_out_mechanism": (None if fold in ("all", LEAK_FOLD)
                                      else fold),
               "leak": leak,
               "leak_note": ("h was fitted ON the evaluation shards; this is "
                             "an in-sample ceiling, never a result"
                             if leak else None),
               "dev_mechanisms_used": sorted(mechs),
               "n_fit_rows": int(len(s)),
               "q": sc_conf.q, "q_pooled": sc_conf.pooled_q,
               "n_cal_q": sc_conf.n_cal,
               "coef_top": h.coef_table(FEAT_NAMES)[:12],
               "per_family_h": bool(args.per_family_h),
           "residual_features": bool(args.residual_features),
               "loss_curve": h.loss_curve,
               "in_dist": {}, "shards": {}}

        for t in in_tasks:
            hh = H(cal_rows[t]["z_q"], t)
            sv = cal_rows[t]["s_q"]
            g = np.array([t] * len(sv))
            wm = sc_conf.width_multiplier(hh, g)
            out["in_dist"][t] = {
                "scaled": cov_entry(sc_conf.covered(sv, hh, g)),
                "scaled_pooled_q": cov_entry(pooled_conf.covered(sv, hh)),
                "ungated": cov_entry(ungated.covered(sv, g)),
                "width_mult_median": float(np.median(wm)),
                "width_ratio_median": float(np.median(wm) / ungated.q[t]),
                "note": "this is the CALIBRATION half, not the test split",
            }

        for name, e in eval_cache.items():
            hh = H(e["z"], e["parent"])
            g = np.array([e["parent"]] * len(e["s"]))
            wm = sc_conf.width_multiplier(hh, g)
            cov = cov_entry(sc_conf.covered(e["s"], hh, g))
            qu = ungated.q.get(e["parent"], ungated.q_pooled)
            wr = float(np.median(wm) / qu)
            # Diagnostic, uses ground truth: the quantile h SHOULD have
            # predicted on this shard, against what it did predict. The ratio
            # is the under-prediction factor and it is the number that says
            # whether h is weak or the shift is unlearnable.
            realized = float(np.quantile(e["s"], 1.0 - args.alpha))
            want = realized / max(float(np.median(wm)), 1e-12)
            # this fold is only *valid* for a shard whose mechanism it held out
            valid = (fold != "all" and e["mech"] == fold)
            out["shards"][name] = {
                "kind": e["kind"], "parent": e["parent"], "N": e["N"],
                "mechanism": e["mech"], "rel_l2_mean": e["rel_l2"],
                "scaled": cov,
                "scaled_pooled_q": cov_entry(
                    pooled_conf.covered(e["s"], hh)),
                "ungated": cov_entry(ungated.covered(e["s"], g)),
                "h_median": float(np.median(hh)),
                "realized_q90_of_score__uses_truth": realized,
                "underprediction_factor__uses_truth": want,
                "width_mult_median": float(np.median(wm)),
                "width_ratio_median": wr,
                "width_inflated": bool(wr > WIDTH_FLAG),
                "in_band": bool(cov["coverage"] is not None
                                and BAND[0] <= cov["coverage"] <= BAND[1]),
                "in_band_at_deployable_width": bool(
                    cov["coverage"] is not None
                    and BAND[0] <= cov["coverage"] <= BAND[1]
                    and wr <= WIDTH_FLAG),
                "held_out_for_this_shard": valid,
            }
        n_band = sum(1 for v in out["shards"].values() if v["in_band"])
        n_ok = sum(1 for v in out["shards"].values()
                   if v["in_band_at_deployable_width"])
        out["in_band_all_shards"] = n_band
        out["in_band_at_deployable_width_all_shards"] = n_ok
        if fold != "all":
            hv = [v for v in out["shards"].values()
                  if v["held_out_for_this_shard"]]
            out["n_held_out_shards"] = len(hv)
            out["in_band_held_out"] = sum(1 for v in hv if v["in_band"])
            out["in_band_held_out_deployable"] = sum(
                1 for v in hv if v["in_band_at_deployable_width"])
        res["folds"][fold] = out
        print(f"[fold={fold:13s}]{' LEAK' if leak else '     '} "
              f"fit rows {len(s):6d}  "
              f"in band (all 32) {n_band}, at deployable width {n_ok}"
              + ("" if fold == "all" else
                 f"  |  held-out {out['in_band_held_out']}"
                 f"/{out['n_held_out_shards']}"), flush=True)

    # the headline: each shard scored by the fold that never saw its mechanism
    lomo = {}
    for name, e in eval_cache.items():
        f = e["mech"]
        if f in res["folds"] and not res["folds"][f]["leak"]:
            lomo[name] = res["folds"][f]["shards"][name]
    res["leave_one_mechanism_out"] = {
        "note": ("each shard scored only by the fold that never saw its shift "
                 "mechanism; the insample_leak fold is excluded by "
                 "construction"),
        "n_shards": len(lomo),
        "in_band": sum(1 for v in lomo.values() if v["in_band"]),
        "in_band_at_deployable_width": sum(
            1 for v in lomo.values() if v["in_band_at_deployable_width"]),
        "shards": lomo,
    }
    print(f"[LOMO headline] in band "
          f"{res['leave_one_mechanism_out']['in_band']}/{len(lomo)}, at "
          f"deployable width "
          f"{res['leave_one_mechanism_out']['in_band_at_deployable_width']}",
          flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
