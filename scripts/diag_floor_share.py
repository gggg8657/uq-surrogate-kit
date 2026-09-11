"""How much of the modulated denominator is the frozen sigma floor?

The kit's score divides by `_floor(sigma) = sigma + 0.05 * med_cal`, an
ADDITIVE constant frozen on calibration. H26 modulates only the sigma term:

    denominator = sigma * w**gamma + 0.05 * med_cal ,  w = relresid/median_cal

so the modulation's leverage is diluted wherever the floor is a large share of
the denominator, and the dilution is worst exactly where sigma is small. codex
raised this as the reason the exponent sweep may not be testing the model it
claims to test. This measures the share before anything is rebuilt around it.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.sims.checkpoint import load_model            # noqa: E402
from uqkit.sims.pde2d import PARENT                     # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator         # noqa: E402
from uqkit.sims.predict import load_shard, predict_shard_single  # noqa: E402

TASKS = ["poisson", "helmholtz", "darcy"]
GAMMAS = (0.375, 1.0, 3.0)
sims = {}


def relresid(a, m, p):
    sims.setdefault(p, PDE2DSimulator(p, device=a.device))
    r = sims[p].residual(a, m)
    f = sims[p].rhs(a)
    return r.flatten(1).norm(dim=1) / f.flatten(1).norm(dim=1).clamp_min(1e-12)


def gather(t, split, model, stats, N=64):
    b = load_shard("data", t, split, N)
    m, s, tr, a = predict_shard_single(model, b, stats, "cuda",
                                       sigma_source="het")
    p = PARENT.get(t, t)
    return m, s, tr, relresid(a, m, p), p


def main():
    man = json.loads(Path("data/manifest.json").read_text())
    model, ck = load_model("runs/u0/best.pt", "cuda")
    stats = ck["stats"]
    cal = {t: gather(t, "cal", model, stats) for t in TASKS}
    RRMED = {t: float(cal[t][3].median()) for t in TASKS}
    MED = float(torch.cat([cal[t][1].flatten() for t in TASKS]).median())
    FL = 0.05 * MED
    out = {"ckpt": "runs/u0/best.pt", "seed": ck["args"].get("seed"),
           "floor_abs": FL, "sigma_median_cal": MED,
           "relresid_median_cal": RRMED, "gammas": list(GAMMAS),
           "note": "share = floor / (sigma*w**gamma + floor), median over "
                   "pixels x samples. 1.0 means the modulation has no "
                   "leverage at all; 0.0 means the floor is irrelevant.",
           "rows": {}}
    print(f"frozen floor 0.05*med_cal = {FL:.6g}\n")
    print(f"{'shard':32s} {'gamma':>6s} {'floor share':>12s} "
          f"{'sigma*w^g (med)':>16s}")

    def row(label, t, split, N=64):
        m, s, tr, r, p = gather(t, split, model, stats, N)
        rec = {}
        for g in GAMMAS:
            w = (r / RRMED[p]).clamp_min(1e-12).pow(g).view(-1, 1, 1, 1)
            sw = s * w
            share = float((FL / (sw + FL)).median())
            rec[f"g{g:g}"] = {"floor_share_median": share,
                              "sigma_mod_median": float(sw.median())}
            print(f"{label:32s} {g:6.3g} {share:12.4f} "
                  f"{float(sw.median()):16.4g}")
        out["rows"][label] = rec
        del m, s, tr, r
        torch.cuda.empty_cache()

    for t in TASKS:
        row(f"in-dist/{t}", t, "test")
    for kind in ("input_shift", "graded_rough"):
        for t in man["ood_suite"][kind]:
            p = PARENT.get(t, t)
            if p not in TASKS:
                continue
            row(f"{kind}/{t}", t, "ood")

    Path("runs/floor_share.json").write_text(json.dumps(out, indent=2))
    print("\nwrote runs/floor_share.json")


if __name__ == "__main__":
    main()
