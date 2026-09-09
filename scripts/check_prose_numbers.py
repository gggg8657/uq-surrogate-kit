"""Verify that the headline numbers in the prose files still match the run JSONs.

    python scripts/check_prose_numbers.py

`RESULTS.md` is generated, so it cannot drift. `WEEKEND.md`, `paper_draft.md`
and `README.md` are written by hand, and a hand-written number that no longer
matches its JSON is the single failure mode this workspace has paid for most
often. This recomputes each claim from `runs/*.json` and checks that the exact
rendered string is present in the file that claims it.

Exit code 1 if any claim has drifted, so CI catches it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / "runs" / name
    return json.loads(p.read_text()) if p.exists() else None


def claims():
    """(file, rendered string, description) for every hand-written headline."""
    out = []
    c, b, o = load("conformal.json"), load("bench.json"), load("ood.json")
    n, i, m = load("bench_noise.json"), load("isoaccuracy.json"), load("members.json")
    lp = load("label_probe.json")

    if c:
        e = c["scores"]["field_max"]["in_dist"]["pooled_group"]
        s = f"{100*e['coverage']:.1f}%"
        for f in ("WEEKEND.md", "paper_draft.md"):
            out.append((f, s, "in-distribution pooled coverage, field_max/group"))
        ood = c["scores"]["field_max"]["ood"]
        elig = [r for r in ood.values()
                if "weighted" in r and not r.get("assumption_violated")]
        n_split = sum(1 for r in elig if 0.88 <= r["split"]["coverage"] <= 0.92)
        n_w = sum(1 for r in elig if 0.88 <= r["weighted"]["coverage"] <= 0.92)
        out.append(("WEEKEND.md", f"{n_split}/{len(elig)} shards in band (split), "
                                  f"{n_w}/{len(elig)} (weighted)",
                    "covariate-shift coverage counts"))
        over = sum(1 for r in elig if r["split"]["coverage"] >= 0.99)
        out.append(("paper_draft.md", f"{over} of {len(elig)} covariate-shift "
                                      "shards sit at 99–100%", "over-coverage count"))

    if b:
        gpu = [r for r in b["rows"] if r["device"] == "cuda" and r["trained"]]
        sp = [r["surrogate"]["ensemble"]["speedup"] for r in gpu]
        n_ge = sum(1 for x in sp if x >= 100)
        out.append(("WEEKEND.md", f"**{n_ge}/{len(sp)} rows ≥100×**",
                    "trained GPU rows clearing 100x"))
        d1 = next(r for r in gpu if r["task"] == "darcy" and r["batch"] == 1)
        d64 = next(r for r in gpu if r["task"] == "darcy" and r["batch"] == 64)
        out.append(("paper_draft.md",
                    f"{d1['surrogate']['ensemble']['speedup']:.1f}× to "
                    f"{d64['surrogate']['ensemble']['speedup']:.1f}×",
                    "darcy latency-vs-throughput speedup"))

    if n:
        v = n["headline_speedup"]
        out.append(("WEEKEND.md", f"**{v['median']:.1f}×** [{v['min']:.1f}, "
                                  f"{v['max']:.1f}]", "repeated headline speedup"))
        out.append(("paper_draft.md", f"{v['median']:.1f}× [{v['min']:.1f}, "
                                      f"{v['max']:.1f}]", "repeated headline speedup"))

    if i and i["families"]["darcy"].get("iso_accuracy"):
        a = i["families"]["darcy"]["iso_accuracy"]
        out.append(("WEEKEND.md", f"**{a['speedup']:.1f}×**", "iso-accuracy speedup"))
        out.append(("paper_draft.md", f"**{a['speedup']:.1f}×**",
                    "iso-accuracy speedup"))

    if m:
        for r in m["rows"]:
            if r["M"] in (1, 2, 5):
                out.append(("WEEKEND.md", f"{r['darcy_speedup']:.1f}×",
                            f"member ablation M={r['M']}"))

    if o:
        sd = o["shift_detection"]
        best, bn = None, -1
        for det in ("spread", "mahalanobis", "residual"):
            k = sum(1 for r in sd.values()
                    if r["auroc"].get(det) and r["auroc"][det]["auroc"] >= 0.9)
            if k > bn:
                best, bn = det, k
        out.append(("WEEKEND.md", f"{bn}/{len(sd)} (`{best}`)",
                    "best shift-detection shard count"))
        for shard in ("unseen_operator/biharmonic/N64", "param_oor/frac_s3/N64"):
            r = sd[shard]
            for det in ("spread", "mahalanobis", "residual"):
                s = f"{r['auroc'][det]['auroc']:.3f}"
                out.append(("WEEKEND.md", s, f"{shard} {det} AUROC")
                           if shard.endswith("biharmonic/N64") or det != "x"
                           else None)
        ed = o["error_detection"]["detectors"]["spread"]
        out.append(("WEEKEND.md", f"**{ed['auroc']:.3f}**",
                    "pooled spread error-detection AUROC"))

    if lp:
        blind = [k for k, r in lp["shifts"].items()
                 if r["k_for_95pct_power"] == 1 and r["kind"] in
                 ("unseen_operator", "param_oor")]
        if blind:
            out.append(("WEEKEND.md", "**k = 1**", "labelled probes needed"))
    return [c for c in out if c]


def main():
    bad = []
    text = {}
    for f in ("WEEKEND.md", "paper_draft.md", "README.md"):
        p = ROOT / f
        text[f] = p.read_text() if p.exists() else ""
    for f, s, why in claims():
        if s not in text.get(f, ""):
            bad.append((f, s, why))
    for f, s, why in bad:
        print(f"DRIFT  {f}: expected {s!r} ({why})")
    n = len(claims())
    print(f"{n - len(bad)}/{n} prose claims match runs/*.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
