"""H32: aggregate the regime-gated method over seeds.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h32.py \
        --out runs/h32_gate.json

Reads `runs/gteq_u*_het.json` / `runs/gtbase_u*_het.json`, written by
`scripts/eval_scale_target.py --onesided --gate`.

The gated method is `c = 1` when the shift gate is quiet and `c = c_dev_p100`
when it fires, so it is reconstructed here from two cells that are already in
each file -- no third run, and no opportunity to choose the constant after
seeing the coverage.

The two error rates that decide whether the gate is usable are reported
together and neither is allowed to stand alone:

* **false alarm** -- the gate firing on the in-distribution TEST split, which
  is held out from the null that set the threshold. A false alarm inflates the
  in-distribution interval by the conservative constant, which is *worse than
  doing nothing* on the case a plant is in almost all the time.
* **miss** -- the gate staying quiet on a shifted shard, which leaves that
  shard at c = 1 and therefore at its uncorrected coverage.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BAND, TARGET = (0.88, 0.92), 0.90
ARMS = {"eq": "runs/gteq_u*_het.json", "base": "runs/gtbase_u*_het.json"}
CONST = "c_dev_p100"


def spread(v):
    v = [float(x) for x in v]
    return {"min": min(v), "median": float(np.median(v)), "max": max(v),
            "n_seeds": len(v), "per_seed": v}


def arm(files):
    seeds = [json.load(open(f)) for f in files]
    for d in seeds:
        if not d.get("gate", {}).get("enabled"):
            raise SystemExit("a file in this arm has no gate block")
    names = sorted(seeds[0]["shards"])
    out = {"n_seeds": len(seeds), "n_shards": len(names),
           "files": [Path(f).name for f in files], "constant": CONST}

    out["theta"] = spread([d["gate"]["theta"] for d in seeds])
    out["null_auc_median"] = spread([d["gate"]["null_auc_median"]
                                     for d in seeds])
    out["probe_n"] = seeds[0]["gate"]["probe_n"]
    out["c"] = spread([d["onesided"]["c_ladder"][CONST] for d in seeds])

    # ---- in distribution: the false-alarm rate and what it costs -----------
    fam = sorted(seeds[0]["in_dist"])
    fa, id_r2, id_cov, id_w = [], [], {t: [] for t in fam}, []
    for d in seeds:
        fired = [d["in_dist"][t]["gate"]["fired"] for t in fam]
        fa.append(int(sum(fired)))
        cov, w = [], []
        for t, f in zip(fam, fired):
            cell = d["in_dist"][t]["onesided"][CONST if f else "c_fixed_1"]
            cov.append(cell["coverage"])
            w.append(cell["width_mult_median"])
            id_cov[t].append(cell["coverage"])
        id_r2.append(int(sum(1 for v in cov if BAND[0] <= v <= BAND[1])))
        id_w.append(float(np.median(w)))
    out["in_dist"] = {
        "n_families": len(fam),
        "false_alarms_of_3": spread(fa),
        "gated_R2_in_band_of_3": spread(id_r2),
        "gated_coverage": {t: spread(v) for t, v in id_cov.items()},
        "gated_width_mult_median": spread(id_w),
        "probe_auc": {t: spread([d["in_dist"][t]["gate"]["probe_auc"]
                                 for d in seeds]) for t in fam}}

    # ---- under shift: fire rate, gated coverage, gated width --------------
    fired_n, r1, r2, wmed, miss_names = [], [], [], [], []
    auc_min = []
    for d in seeds:
        cov, w, nf = [], [], 0
        for n in names:
            f = d["shards"][n]["gate"]["fired"]
            nf += f
            cell = d["shards"][n]["onesided"][CONST if f else "c_fixed_1"]
            cov.append(cell["coverage"])
            w.append(cell["width_mult_median"])
            if not f:
                miss_names.append(n)
        cov, w = np.array(cov), np.array(w)
        fired_n.append(nf)
        r1.append(int((cov >= TARGET).sum()))
        r2.append(int(((cov >= BAND[0]) & (cov <= BAND[1])).sum()))
        wmed.append(float(np.median(w)))
        auc_min.append(min(d["shards"][n]["gate"]["probe_auc"] for n in names))
    out["shift"] = {
        "gate_fired_of_24": spread(fired_n),
        "gated_R1_one_sided_ge_0p90": spread(r1),
        "gated_R2_two_sided_in_band": spread(r2),
        "gated_width_mult_median": spread(wmed),
        "min_probe_auc_over_shards": spread(auc_min),
        "missed_shards": sorted(set(miss_names))}

    # ---- the joint statement, per seed, with nothing averaged away --------
    joint = [{"seed": d.get("seed"),
              "in_dist_R2_of_3": r2i, "shift_R1_of_24": r1i}
             for d, r2i, r1i in zip(seeds, id_r2, r1)]
    out["joint_per_seed"] = joint
    out["joint_all_seeds_pass"] = bool(
        all(j["in_dist_R2_of_3"] == len(fam) for j in joint)
        and all(j["shift_R1_of_24"] == len(names) for j in joint))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/h32_gate.json")
    args = ap.parse_args()
    res = {"hypothesis": "H32", "band": list(BAND), "target": TARGET,
           "method": ("c = 1 when the shift gate is quiet, c = c_dev_p100 "
                      "when it fires. The gate is a two-sample "
                      "LikelihoodRatioProbe on spectral input features, "
                      "thresholded at the maximum held-out AUC over "
                      "in-distribution null probes built from disjoint halves "
                      "of the CALIBRATION split. No evaluation shard "
                      "contributes to the threshold or to the constant."),
           "reading": ("R2 (two-sided, the clause) in distribution and R1 "
                       "(one-sided >= 0.90 with a quoted width) under shift. "
                       "R1 is NOT the clause and is never reported as it."),
           "arms": {}}
    for a, pat in ARMS.items():
        fs = sorted(glob.glob(pat))
        res["arms"][a] = arm(fs) if fs else {"status": "[not measured]",
                                             "glob": pat}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")
    for a, A in res["arms"].items():
        if "shift" not in A:
            print(f"{a}: {A['status']}")
            continue
        print(f"--- {a}: {A['n_seeds']} seeds, theta "
              f"{A['theta']['median']:.4f} (null median "
              f"{A['null_auc_median']['median']:.4f}), c = "
              f"{A['c']['median']:.4g}")
        print(f"  in-dist  false alarms {A['in_dist']['false_alarms_of_3']['min']:.0f}"
              f"-{A['in_dist']['false_alarms_of_3']['max']:.0f}/3   "
              f"gated R2 {A['in_dist']['gated_R2_in_band_of_3']['min']:.0f}"
              f"-{A['in_dist']['gated_R2_in_band_of_3']['max']:.0f}/3   "
              f"width x{A['in_dist']['gated_width_mult_median']['median']:.3g}")
        print(f"  shift    fired {A['shift']['gate_fired_of_24']['min']:.0f}"
              f"-{A['shift']['gate_fired_of_24']['max']:.0f}/24   "
              f"gated R1 {A['shift']['gated_R1_one_sided_ge_0p90']['min']:.0f}"
              f"-{A['shift']['gated_R1_one_sided_ge_0p90']['max']:.0f}/24   "
              f"gated R2 {A['shift']['gated_R2_two_sided_in_band']['min']:.0f}"
              f"-{A['shift']['gated_R2_two_sided_in_band']['max']:.0f}/24   "
              f"width x{A['shift']['gated_width_mult_median']['median']:.4g}")
        print(f"  min probe AUC over shards "
              f"{A['shift']['min_probe_auc_over_shards']['min']:.4f}; "
              f"joint pass on all seeds: {A['joint_all_seeds_pass']}")


if __name__ == "__main__":
    main()
