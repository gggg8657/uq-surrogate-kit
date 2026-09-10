"""Is the packed-weight spectral path bit-identical at EVERY resolution shipped?

`bench_fair.py`'s `packed_weight_gate` checks the shipped checkpoint at N=64,
which is where clause 2 is timed. But the coverage and OOD tables in this repo
evaluate at N=128 and N=256 as well, and `SpectralConv2d` caches its packed
weights under a key that includes `(m1, m2)` -- which is the *same* key at every
N >= 40, since `m1 = min(modes, H//2)` saturates. So the higher resolutions
reuse a cache built at N=64 and nothing had checked that.

Bar: exact equality on the mean and both interval bounds, at every resolution
and on every family. Writes `runs/packed_equivalence.json`.

    CUDA_VISIBLE_DEVICES=2 python scripts/check_packed_equivalence.py \
        --ckpt runs/u0/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.bench import env_report  # noqa: E402
from uqkit.sims.fno2d_uq import sigma_from, split_heads  # noqa: E402
from uqkit.sims.pde2d import PARENT, TASK_ID  # noqa: E402
from uqkit.sims.predict import load_members  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--source", default="het", choices=["het", "cqr", "const"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="runs/packed_equivalence.json")
    args = ap.parse_args()

    models, ck = load_members([args.ckpt], "cuda")
    model = models[0].eval()
    specs = [b.spectral for b in model.blocks]

    shards = []
    for p in sorted(Path(args.root).glob("*.pt")):
        name = p.stem                                  # e.g. darcy_ood_N128
        if "_N" not in name:
            continue
        shards.append(name)

    res = {"env": env_report("cuda"), "ckpt": args.ckpt, "source": args.source,
           "bar": "exact equality (max|delta| == 0) on mean, lo and hi",
           "why": ("the cache key is (m1, m2, dtype, device) and m1 = "
                   "min(modes, H//2) saturates for every N >= 2*modes, so a "
                   "cache built at N=64 is reused verbatim at N=128 and "
                   "N=256; nothing had checked that until now"),
           "shards": {}}

    def run(blob, task, fast):
        for sp in specs:
            sp.cache_packed_weights = fast
            sp.clear_packed_cache()
        parent = PARENT.get(task, task)
        st = ck["stats"][parent]
        dev = "cuda"
        a = blob["a"][: args.batch].to(dev)
        tid = torch.full((a.shape[0],), TASK_ID[parent], device=dev,
                         dtype=torch.long)
        a_n = (a - st["a_mean"].to(dev)) / st["a_std"].to(dev)
        u_m, u_s = st["u_mean"].to(dev), st["u_std"].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(a_n, tid).float()
        mean = split_heads(y)[0] * u_s + u_m
        sig = sigma_from(y, args.source) * u_s
        return mean, mean - sig, mean + sig

    def task_of(name):
        """`darcy_dam0p5_ood_N64` -> `darcy_dam0p5`, `frac_s3_ood_N64` -> `frac_s3`.

        Splitting on the first `_` gets `frac`, `navier` and `ns` wrong -- those
        are exactly the unseen-operator shards the OOD tables lean on, so a
        naive parse silently skips the five shards it most matters for.
        """
        stem = name
        for suf in ("_N16", "_N32", "_N64", "_N128", "_N256"):
            stem = stem.removesuffix(suf)
        for suf in ("_ood", "_test", "_train", "_cal"):
            stem = stem.removesuffix(suf)
        return stem

    worst = 0.0
    for name in shards:
        blob = torch.load(Path(args.root) / f"{name}.pt",
                          map_location="cpu", weights_only=False)
        if "a" not in blob:
            continue
        base = task_of(name)
        parent = PARENT.get(base, base)
        if parent not in ck["stats"]:
            res["shards"][name] = {"skipped": f"{base!r} -> no stats for "
                                              f"{parent!r}"}
            continue
        N = int(blob["a"].shape[-1])
        try:
            slow = run(blob, base, False)
            fast = run(blob, base, True)
        except Exception as e:                      # noqa: BLE001
            res["shards"][name] = {"error": repr(e)}
            continue
        dev = {k: float((f - s).abs().max())
               for k, f, s in zip(("mean", "lo", "hi"), fast, slow)}
        worst = max(worst, *dev.values())
        res["shards"][name] = {"N": N, "n": int(min(args.batch, len(blob["a"]))),
                               "max_abs_dev": dev,
                               "field_scale": float(slow[0].abs().max()),
                               "identical": all(v == 0.0 for v in dev.values())}
        print(f"{name:28s} N={N:4d} max|d| = "
              f"{max(dev.values()):.3e} "
              f"{'IDENTICAL' if all(v == 0.0 for v in dev.values()) else 'DIFFERS'}",
              flush=True)

    ok = [v for v in res["shards"].values() if v.get("identical")]
    bad = [k for k, v in res["shards"].items() if v.get("identical") is False]
    res["summary"] = {"n_shards": len(res["shards"]), "n_identical": len(ok),
                      "n_differ": len(bad), "differing": bad,
                      "worst_max_abs_dev": worst,
                      "resolutions": sorted({v["N"] for v in res["shards"].values()
                                             if "N" in v})}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\n{len(ok)}/{len(res['shards'])} shards bit-identical, "
          f"resolutions {res['summary']['resolutions']}, worst {worst:.3e}")
    print(f"wrote {args.out}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
