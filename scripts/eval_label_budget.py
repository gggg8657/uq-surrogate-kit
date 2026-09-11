"""H7: how many labels does it take to re-certify the interval after a shift?

    CUDA_VISIBLE_DEVICES=2 python scripts/eval_label_budget.py \
        --ckpts runs/m0/best.pt --out runs/label_budget.json

Two-sided 90 +- 2% coverage under arbitrary covariate shift, without labels, is
not attainable: weighted conformal gives one-sided conservative validity only,
and when the calibration and test inputs are separable it returns `q = +inf`.
That is not a theoretical worry here -- `runs/conformal.json` has probe AUC
1.000 on every shard, 30/32 shards with at least one infinite quantile and 14/32
infinite for all 512 test points. An infinite interval covers everything and
certifies nothing, which is why this file reports **width** next to every
coverage number and refuses to report one without it.

So the honest question is not "can it be done unlabelled" but "what does it
cost". This measures that, in labels:

* **k-label recalibration** -- draw k labelled samples from the shifted shard,
  take the split-conformal quantile of their scores, and test coverage on the
  samples that were NOT drawn. Repeated over `--repeats` independent draws.
  `conformal_quantile` returns +inf whenever `ceil((k+1)(1-alpha)) > k`, i.e.
  for every k < 9 at alpha = 0.1, so the curve is *forced* to start at infinite
  width and the k at which it stops being infinite is arithmetic, not a
  finding. What is measured is where coverage enters [88, 92] and how wide the
  interval is when it does.
* **the oracle target split** (`k = n/2`) -- the same thing at the largest
  budget the shard can pay. It separates two explanations of the unlabelled
  failure: if the oracle lands in band, `field_max` is a sound score under
  shift and the whole failure is calibration transfer; if it does not, the
  score distribution itself is pathological under shift and no reweighting can
  fix it.

**H30 adds a second, cheaper estimator on the same draws.** H27/H29 measured
that the correction the coverage clause needs is ONE SCALAR -- `s*` exists on
24/24 shards, is 1.00 in distribution, and hitting it to +/-2.52% suffices.
Estimating one scale is a cheaper statistical problem than estimating a 90th
percentile, and the estimator can be a MEDIAN, which is finite at k = 1:

    arm A (H7)   q_hat = conformal_quantile(S_k, alpha)             k >= 9 only
    arm B (H30)  q_hat = q_cal[family] * median(S_k)/median_cal(S)  finite at k=1
    arm C (H30)  the same ratio computed on ALL the shard's samples

Arm C is arm B's k -> infinity limit and is the diagnostic that decides whether
the route has a ceiling: it tests SHAPE PRESERVATION. If the shifted score
distribution is the calibration distribution times a constant, the median ratio
and the 0.90-quantile ratio are the same number and arm C lands in band. If it
is not, arm B converges to the wrong constant and plateaus outside the band at
every k -- a cap no label budget can buy past.

**All three are oracles. None is a deployable method** -- they consume labels
from the shard they certify, which is exactly what a surrogate exists to avoid.
They are reported as an upper bound on what any calibrator could achieve on
this data, and as a price list. Every table this writes says so in its own row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import _floor, conformal_quantile, get_score  # noqa: E402
from uqkit.metrics import binom_ci  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.checkpoint import load_model  # noqa: E402
from uqkit.sims.predict import (load_members, load_shard,  # noqa: E402
                                predict_shard, predict_shard_single)

BAND = (0.88, 0.92)
KS = (1, 2, 3, 5, 8, 9, 12, 16, 24, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/label_budget.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--score", default="field_max")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-source", default=None,
                    choices=["het", "cqr", "const"])
    args = ap.parse_args()
    single = args.sigma_source is not None
    if single and len(args.ckpts) != 1:
        ap.error("--sigma-source is a single-network mode; pass one checkpoint")

    man = json.loads(Path(args.root, "manifest.json").read_text())
    in_tasks = man["in_tasks"]
    if single:
        model, ck = load_model(args.ckpts[0], args.device)
        if not ck["args"].get("uq"):
            raise SystemExit(f"{args.ckpts[0]} is not a UQ checkpoint")
        models = [model]
    else:
        models, ck = load_members(args.ckpts, args.device)
    stats = ck["stats"]
    fn, _per_sample = get_score(args.score)

    def gather(task, split, N=64):
        blob = load_shard(args.root, task, split, N)
        if single:
            m, s, t, a = predict_shard_single(models[0], blob, stats,
                                              args.device,
                                              sigma_source=args.sigma_source)
        else:
            m, s, t, a = predict_shard(models, blob, stats, args.device)
        return {"mean": m, "sigma": s, "truth": t, "parent": PARENT.get(task, task)}

    # the sigma floor is frozen on the pooled in-distribution calibration split
    # and passed to every score call, exactly as `eval_conformal.py` does it.
    # Letting each shard set its own floor would move every band width in the
    # direction of apparent coverage.
    cal = {t: gather(t, "cal") for t in in_tasks}
    MED = float(torch.cat([cal[t]["sigma"].flatten() for t in in_tasks]).median())
    FRAC = 0.05

    # Arm B rescales the FROZEN per-family calibration quantile by a median
    # ratio, so both constants come from the calibration split and neither ever
    # sees a shifted sample. This is the same quantile eval_conformal.py ships.
    CAL_S = {t: fn(cal[t]["mean"], cal[t]["sigma"], cal[t]["truth"],
                   med=MED).cpu().numpy() for t in in_tasks}
    Q_CAL = {t: float(conformal_quantile(CAL_S[t], args.alpha))
             for t in in_tasks}
    MED_CAL = {t: float(np.median(CAL_S[t])) for t in in_tasks}
    if not all(np.isfinite(v) for v in Q_CAL.values()):
        raise SystemExit(f"a frozen calibration quantile is not finite: "
                         f"{Q_CAL}; arm B would be vacuous")

    ood_specs = []
    for kind, tasks in man["ood_suite"].items():
        ood_specs += [(kind, t, 64) for t in tasks]
    for N in (128, 256):
        for t in man["res_tasks"]:
            if Path(args.root, f"{t}_ood_N{N}.pt").exists():
                ood_specs.append((f"resolution_{N}", t, N))

    rng = np.random.default_rng(args.seed)
    res = {"alpha": args.alpha, "score": args.score, "band": list(BAND),
           "ks": list(KS), "repeats": args.repeats, "ckpts": args.ckpts,
           "n_members": len(models), "sigma_floor_median": MED,
           "sigma_floor_frac": FRAC,
           "sigma_source": args.sigma_source or "ensemble_spread",
           "k_min_finite": int(np.ceil((1 - args.alpha) / args.alpha)),
           "q_cal_frozen": Q_CAL, "median_cal_score": MED_CAL,
           "arms": {
               "A": "conformal_quantile(S_k, alpha) -- recalibrates the "
                    "quantile from k labels; infinite for every k < "
                    "k_min_finite by arithmetic",
               "B": "q_cal[family] * median(S_k)/median_cal(S) -- rescales the "
                    "frozen quantile by a median ratio; ONE parameter, finite "
                    "at k=1. Reported as scale_* keys.",
               "C": "arm B's k->infinity limit, computed on all samples and "
                    "scored on them. Tests shape preservation. Key "
                    "scale_limit."},
           "reading_warning":
               "`coverage` and `in_band` are averages over `repeats` "
               "independent labelled draws. Split conformal is exactly valid "
               "marginally, so the AVERAGE sits at 1-alpha whenever the "
               "quantile is finite and `in_band` on the mean is nearly "
               "vacuous. A deployment gets ONE draw: read "
               "`frac_draws_in_band`, whose distribution-free prediction is "
               "Beta(k+1-l, l), l = floor((k+1)*alpha) -- 0.156 at k=9.",
           "oracle_warning":
               "Every number in this file consumes labels from the shard it "
               "certifies. It is an upper bound on what a calibrator could do "
               "on this data and a price list in labels -- NOT a deployable "
               "method, and not evidence for the unlabelled KPI clause.",
           "shards": {}}

    def curve(s, half_width, q_cal, med_cal):
        """Coverage and width vs k, averaged over independent labelled draws.

        Both arms see THE SAME draw at every repeat, so the comparison between
        them is paired and the difference is the estimator, not the sample.
        """
        s = np.asarray(s, dtype=np.float64)
        hw = np.asarray(half_width, dtype=np.float64)
        n = len(s)
        out = {}
        for k in list(KS) + [n // 2]:
            if k >= n:
                continue
            covs, qs, n_inf = [], [], 0
            covs_b, qs_b = [], []
            for _ in range(args.repeats):
                idx = rng.permutation(n)
                drawn, rest = idx[:k], idx[k:]
                # --- arm A: recalibrate the quantile from k labels -----------
                q = conformal_quantile(s[drawn], args.alpha)
                if not np.isfinite(q):
                    n_inf += 1
                    covs.append(1.0)          # an infinite interval covers all
                else:
                    qs.append(float(q))
                    covs.append(float((s[rest] <= q).mean()))
                # --- arm B: rescale the FROZEN quantile by a median ratio ----
                # One parameter, estimated by a median, so it is finite at k=1.
                qb = q_cal * float(np.median(s[drawn])) / med_cal
                qs_b.append(qb)
                covs_b.append(float((s[rest] <= qb).mean()))
            key = "oracle_half" if k == n // 2 else f"k{k}"
            mean_cov = float(np.mean(covs))
            mean_cov_b = float(np.mean(covs_b))
            lo, hi = binom_ci(int(round(mean_cov * (n - k))), n - k)
            lob, hib = binom_ci(int(round(mean_cov_b * (n - k))), n - k)
            # THE NUMBER THAT MATTERS, and the one an earlier version of this
            # file did not report. Split conformal is exactly valid MARGINALLY,
            # so the mean over many labelled draws sits at 1-alpha by
            # construction and "the mean is in band" is close to a tautology.
            # A deployment gets ONE draw. At k calibration points the coverage
            # of a single draw is Beta(k+1-l, l) with l = floor((k+1)*alpha) --
            # sd 0.0905 at k=9 -- so the fraction of INDIVIDUAL draws landing
            # in band is the real price, and it is 0.156 at k=9, not 1.0.
            frac_a = float(np.mean([BAND[0] <= c <= BAND[1] for c in covs]))
            frac_b = float(np.mean([BAND[0] <= c <= BAND[1] for c in covs_b]))
            out[key] = {
                "k": int(k), "coverage": mean_cov,
                "frac_draws_in_band": frac_a,
                "scale_frac_draws_in_band": frac_b,
                "coverage_sd_over_draws": float(np.std(covs)),
                "scale_coverage_sd_over_draws": float(np.std(covs_b)),
                "coverage_ci95": [lo, hi],
                "frac_draws_infinite": n_inf / args.repeats,
                "q_median": float(np.median(qs)) if qs else None,
                # the band's own half-width in the field's units, so a 100%
                # coverage row that is infinitely wide cannot read as a pass
                "mean_half_width": (float(np.median(qs) * hw.mean())
                                    if qs else float("inf")),
                "in_band_of_mean": bool(BAND[0] <= mean_cov <= BAND[1]),
                "in_band": bool(BAND[0] <= mean_cov <= BAND[1]),
                # arm B, on the same draws
                "scale_coverage": mean_cov_b,
                "scale_coverage_ci95": [lob, hib],
                "scale_q_median": float(np.median(qs_b)),
                "scale_mean_half_width": float(np.median(qs_b) * hw.mean()),
                "scale_in_band": bool(BAND[0] <= mean_cov_b <= BAND[1])}
        # --- arm C: arm B's k -> infinity limit, the shape-preservation test --
        # Uses every sample on the shard to form the ratio, then scores the same
        # samples: it is the most generous possible reading of arm B and it
        # cannot be improved by more labels.
        qc = q_cal * float(np.median(s)) / med_cal
        cov_c = float((s <= qc).mean())
        loc, hic = binom_ci(int(round(cov_c * n)), n)
        out["scale_limit"] = {
            "k": int(n), "coverage": cov_c, "coverage_ci95": [loc, hic],
            "q": qc, "median_ratio": float(np.median(s)) / med_cal,
            "in_band": bool(BAND[0] <= cov_c <= BAND[1]),
            "note": "arm C: the median-ratio scale computed on ALL samples and "
                    "scored on the same samples. Arm B's ceiling. If this is "
                    "out of band the shift is not shape-preserving and no "
                    "label budget reaches the band by this estimator."}
        return out

    def half_width_unit(d):
        """Mean per-sample sigma after the floor -- the band is q * this."""
        return _floor(d["sigma"], FRAC, MED).flatten(1).mean(1).cpu().numpy()

    todo = [("in_dist", t, t, 64) for t in in_tasks] + \
           [(kind, t, PARENT.get(t, t), N) for kind, t, N in ood_specs]
    for kind, task, parent, N in todo:
        d = gather(task, "test" if kind == "in_dist" else "ood", N)
        s = fn(d["mean"], d["sigma"], d["truth"], med=MED).cpu().numpy()
        c = curve(s, half_width_unit(d), Q_CAL[parent], MED_CAL[parent])
        res["shards"][f"{kind}/{task}/N{N}"] = {
            "kind": kind, "parent": parent, "N": N, "n": int(len(s)),
            "curve": c,
            "k_min_in_band": next((c[f"k{k}"]["k"] for k in KS
                                   if f"k{k}" in c and c[f"k{k}"]["in_band"]),
                                  None),
            "scale_k_min_in_band": next((c[f"k{k}"]["k"] for k in KS
                                         if f"k{k}" in c
                                         and c[f"k{k}"]["scale_in_band"]),
                                        None),
            "scale_limit_in_band": bool(c["scale_limit"]["in_band"]),
            "scale_limit_coverage": c["scale_limit"]["coverage"],
            "scale_median_ratio": c["scale_limit"]["median_ratio"]}
        o = c.get("oracle_half", {})
        k9 = c.get("k9", {})
        rr = res["shards"][f"{kind}/{task}/N{N}"]
        print(f"{kind[:14]:14s} {task:16s} N{N:<4d} n={len(s):4d}  "
              f"A: k9 {k9.get('coverage', float('nan')):.3f} "
              f"oracle {o.get('coverage', float('nan')):.3f} "
              f"kmin {str(rr['k_min_in_band']):>4s}  |  "
              f"B: k1 {c.get('k1', {}).get('scale_coverage', float('nan')):.3f} "
              f"k8 {c.get('k8', {}).get('scale_coverage', float('nan')):.3f} "
              f"kmin {str(rr['scale_k_min_in_band']):>4s}  |  "
              f"C: {c['scale_limit']['coverage']:.3f} "
              f"x{c['scale_limit']['median_ratio']:.3g} "
              f"{'IN BAND' if c['scale_limit']['in_band'] else '       '}",
              flush=True)
        del d
        torch.cuda.empty_cache()

    ood = [v for k, v in res["shards"].items() if v["kind"] != "in_dist"]
    res["summary"] = {
        "n_ood_shards": len(ood),
        "oracle_half_in_band": sum(1 for v in ood
                                   if v["curve"].get("oracle_half", {}).get("in_band")),
        "k9_in_band": sum(1 for v in ood
                          if v["curve"].get("k9", {}).get("in_band")),
        "median_k_min_in_band": float(np.median(
            [v["k_min_in_band"] for v in ood if v["k_min_in_band"] is not None]))
        if any(v["k_min_in_band"] is not None for v in ood) else None,
        "n_never_in_band": sum(1 for v in ood if v["k_min_in_band"] is None),
        # arm B and its ceiling
        "scale_limit_in_band": sum(1 for v in ood if v["scale_limit_in_band"]),
        "scale_k9_in_band": sum(1 for v in ood
                                if v["curve"].get("k9", {})
                                .get("scale_in_band")),
        "scale_k1_in_band": sum(1 for v in ood
                                if v["curve"].get("k1", {})
                                .get("scale_in_band")),
        "scale_median_k_min_in_band": float(np.median(
            [v["scale_k_min_in_band"] for v in ood
             if v["scale_k_min_in_band"] is not None]))
        if any(v["scale_k_min_in_band"] is not None for v in ood) else None,
        "scale_n_never_in_band": sum(1 for v in ood
                                     if v["scale_k_min_in_band"] is None),
        # the per-draw reading: what one deployment with k labels actually gets
        "frac_draws_in_band_by_k": {
            f"k{k}": float(np.median([v["curve"][f"k{k}"]["frac_draws_in_band"]
                                      for v in ood if f"k{k}" in v["curve"]]))
            for k in KS
            if any(f"k{k}" in v["curve"] for v in ood)},
        "scale_frac_draws_in_band_by_k": {
            f"k{k}": float(np.median(
                [v["curve"][f"k{k}"]["scale_frac_draws_in_band"]
                 for v in ood if f"k{k}" in v["curve"]]))
            for k in KS
            if any(f"k{k}" in v["curve"] for v in ood)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
