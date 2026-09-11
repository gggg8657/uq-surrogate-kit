"""H32, rung 3: the shift gate's error rates, replicated over INPUTS.

    ~/miniforge3/envs/pdeno/bin/python scripts/eval_gate_only.py \
        --out runs/gate_reps.json

Why this script exists at all. In `runs/gteq_u*_het.json` the gate's threshold
and every probe AUC are identical on all 8 seeds to four decimals, because the
gate is a function of the *inputs* alone: spectral features of `a` against the
calibration inputs. Varying the model seed does not vary the gate. So the
"0/3 false alarms, 24/24 fired" in that file is **one input realization
replicated eight times**, which by this repo's own seed rule is a screen and
not a verdict.

This replicates over what actually varies: repeated disjoint subsamples of the
calibration inputs and of the batch being judged, at a matched batch size. It
loads no checkpoint and touches no GPU -- which is the cleanest possible
demonstration that the gate is model-independent, and means the gate's error
rates can be quoted without a surrogate at all.

Protocol, unchanged from the in-run gate:

* features are `spectral_features(a)`, inputs only, no truth and no solve;
* theta is the MAXIMUM held-out AUC over in-distribution null probes built from
  disjoint halves of the CALIBRATION split at the matched batch size. No
  evaluation shard and no development shard contributes to it;
* the in-distribution TEST split is held out from the null, so its fire rate is
  an honest false-alarm rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import LikelihoodRatioProbe  # noqa: E402
from uqkit.features import spectral_features  # noqa: E402
from uqkit.sims.pde2d import PARENT  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402
from uqkit.sims.predict import load_shard  # noqa: E402

COVARIATE_KINDS = {"input_shift", "graded_rough"}


def feats(root, task, split, N=64):
    blob = load_shard(root, task, split, N)
    return spectral_features(blob["a"]).cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-n", type=int, default=256,
                    help="batch size on both sides of every probe")
    ap.add_argument("--reps", type=int, default=8,
                    help="input replicates per decision")
    ap.add_argument("--null-reps", type=int, default=8,
                    help="null probes per family used to estimate theta")
    ap.add_argument("--delta", type=float, default=None,
                    help="H32b: set theta to the (1-delta) quantile of the "
                         "per-family null AUC distribution instead of its "
                         "sample maximum. A sample maximum over R draws has "
                         "no false-alarm guarantee -- it is the largest of R "
                         "samples, and a fresh draw exceeds it with "
                         "probability about 1/(R+1), which is how the 1/24 "
                         "false alarm in the first run happened. With --delta "
                         "the false-alarm rate is a design parameter.")
    args = ap.parse_args()

    man = json.loads(Path(args.root, "manifest.json").read_text())
    probe_a, probe_u = torch.zeros(1, 2, 64, 64), torch.zeros(1, 1, 64, 64)
    in_tasks = [t for t in man["in_tasks"]
                if PDE2DSimulator(t, device="cpu").residual(probe_a, probe_u)
                is not None]
    print(f"families with a cheap apply: {in_tasks}", flush=True)

    CAL = {t: feats(args.root, t, "cal") for t in in_tasks}
    n = args.probe_n

    def probe_auc(fa, fb, seed):
        rng = np.random.default_rng(seed)
        ia = rng.choice(len(fa), size=min(n, len(fa)), replace=False)
        ib = rng.choice(len(fb), size=min(n, len(fb)), replace=False)
        return float(LikelihoodRatioProbe(seed=seed).fit(fa[ia], fb[ib]).auc)

    # ---- theta: in-distribution nulls on disjoint halves of `cal` ----------
    # per family, because the configured task is known at deployment and the
    # conformal quantile in this kit is already per family
    nulls_by_fam = {}
    for t in in_tasks:
        F = CAL[t]
        half = len(F) // 2
        A, B = F[:half], F[half:]
        nulls_by_fam[t] = [probe_auc(A, B, 1000 + rep)
                           for rep in range(args.null_reps)]
    nulls = [x for v in nulls_by_fam.values() for x in v]
    if args.delta is None:
        theta_by_fam = {t: float(max(nulls))    # the pooled sample maximum
                        for t in in_tasks}
        theta_rule = f"pooled sample maximum of {len(nulls)} nulls"
    else:
        theta_by_fam = {t: float(np.quantile(v, 1.0 - args.delta))
                        for t, v in nulls_by_fam.items()}
        theta_rule = (f"per-family {100 * (1 - args.delta):g}th percentile of "
                      f"{args.null_reps} nulls (delta = {args.delta})")
    theta = float(max(theta_by_fam.values()))
    print(f"theta rule: {theta_rule}", flush=True)
    for t in in_tasks:
        print(f"  theta[{t}] {theta_by_fam[t]:.4f}  "
              f"(null median {np.median(nulls_by_fam[t]):.4f}, "
              f"max {max(nulls_by_fam[t]):.4f})", flush=True)

    res = {"hypothesis": "H32", "reading": "gate error rates, replicated over "
           "inputs rather than over model seeds",
           "probe_n": n, "reps": args.reps, "families": in_tasks,
           "theta": theta, "theta_by_family": theta_by_fam,
           "theta_rule": theta_rule, "delta": args.delta,
           "null_reps_per_family": args.null_reps,
           "null_aucs": nulls, "null_aucs_by_family": nulls_by_fam,
           "null_auc_median": float(np.median(nulls)),
           "model_free": True,
           "note": ("the gate is a function of the inputs alone, so it has no "
                    "model seed. These replicates resample the calibration "
                    "batch and the judged batch; the in-run gate in "
                    "runs/gt*_u*_het.json is ONE realization of this."),
           "in_dist": {}, "shards": {}}

    # ---- false alarms on the in-distribution TEST split -------------------
    for t in in_tasks:
        F = feats(args.root, t, "test")
        aucs = [probe_auc(CAL[t], F, 2000 + r) for r in range(args.reps)]
        fired = [a > theta_by_fam[t] for a in aucs]
        res["in_dist"][t] = {"n_test": int(len(F)), "aucs": aucs,
                             "auc_median": float(np.median(aucs)),
                             "fired": fired,
                             "false_alarm_rate": float(np.mean(fired))}
        print(f"  in-dist {t:10s} auc median {np.median(aucs):.4f}  "
              f"false alarms {sum(fired)}/{args.reps}", flush=True)

    # ---- fire rate on every covariate-shift shard ------------------------
    for kind, tasks in man["ood_suite"].items():
        if kind not in COVARIATE_KINDS:
            continue
        for task in tasks:
            parent = PARENT.get(task, task)
            if parent not in in_tasks:
                continue
            F = feats(args.root, task, "ood")
            aucs = [probe_auc(CAL[parent], F, 3000 + r)
                    for r in range(args.reps)]
            fired = [a > theta_by_fam[parent] for a in aucs]
            res["shards"][f"{kind}/{task}/N64"] = {
                "parent": parent, "n_ood": int(len(F)), "aucs": aucs,
                "auc_median": float(np.median(aucs)), "fired": fired,
                "fire_rate": float(np.mean(fired))}
            print(f"  {kind}/{task:22s} auc median {np.median(aucs):.4f}  "
                  f"fired {sum(fired)}/{args.reps}", flush=True)

    res["n_shards"] = len(res["shards"])
    res["shards_fired_every_rep"] = sum(
        1 for r in res["shards"].values() if all(r["fired"]))
    res["shards_fired_no_rep"] = sum(
        1 for r in res["shards"].values() if not any(r["fired"]))
    res["in_dist_false_alarm_rate_pooled"] = float(np.mean(
        [x for r in res["in_dist"].values() for x in r["fired"]]))
    res["min_shard_auc_median"] = min(r["auc_median"]
                                      for r in res["shards"].values())
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n{res['shards_fired_every_rep']}/{res['n_shards']} shards fired "
          f"on every replicate, {res['shards_fired_no_rep']} on none; pooled "
          f"in-distribution false-alarm rate "
          f"{res['in_dist_false_alarm_rate_pooled']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
