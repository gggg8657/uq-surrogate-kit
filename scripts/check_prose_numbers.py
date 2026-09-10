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
    selj, scj = load("selective.json"), load("scale.json")
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

    # H18: the composition arm, and the withdrawal it forces. These are the
    # newest hand-written numbers in WEEKEND.md, so they are pinned here --
    # the whole point of this script is that a prose number cannot outlive the
    # run that produced it.
    if scj and scj.get("h18") and scj["h18"].get("vs_h15"):
        v = scj["h18"]["vs_h15"]
        L = scj["h18"]["lomo"]
        out.append(("WEEKEND.md", f"**{L['in_band_median']:g}/32** median",
                    "H18 leave-one-mechanism-out median"))
        out.append(("WEEKEND.md",
                    f"p = {v['exact_sign_flip_p']:.4f}",
                    "H18 vs H15 paired exact sign-flip p"))
        # the coefficient that carried H15, read out of the per-seed runs
        import glob as _g
        import statistics as _st
        co = []
        for f in sorted(_g.glob("runs/scale_u*_het.json")):
            d = json.loads(Path(f).read_text())
            for cf in d["folds"]["all"]["coef_top"]:
                if cf["feature"] == "a_spec9":
                    co.append(cf["coef"])
        if len(co) >= 2:
            out.append(("WEEKEND.md", f"+{_st.median(co):.3f}",
                        "H15 amplitude coefficient (a_spec9), median over seeds"))

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

    # ---- the M=1 single-network results (the reopened clauses) ----
    u, csu = load("uq_seeds.json"), load("consistency_uq.json")
    # The H12 run (packed spectral weights) supersedes the H11 one as the
    # clause reading; the H11 file is kept for the before/after in RESULTS.md
    # but no hand-written prose should still be quoting it.
    fa = load("bench_fair_h12.json") or load("bench_fair.json")
    pe = load("packed_equivalence.json")
    h13 = load("h13_clip.json")
    if u:
        arms = {k: v for k, v in u["arms"].items() if k != "const"}
        best = min(arms, key=lambda k: sum(
            r["sharpness_rel"] for r in arms[k]["per_seed"]) / arms[k]["n_seeds"])
        a = arms[best]
        for f in ("WEEKEND.md", "paper_draft.md", "README.md"):
            out.append((f, f"{a['coverage_in_dist']['mean']:.4f}",
                        f"M=1 {best}-head coverage mean over "
                        f"{a['n_seeds']} seeds"))
        ctl = u["arms"].get("const")
        if ctl:
            def ms(x):
                return sum(r["sharpness_rel"] for r in x["per_seed"]) / x["n_seeds"]
            gain = 100 * (ms(ctl) - ms(a)) / ms(ctl)
            out.append(("WEEKEND.md", f"{gain:.1f}% sharper",
                        "sigma head's only real effect vs the constant control"))
    if fa:
        b1 = fa["batches"].get("1", {})
        for arm in ("graph", "eager"):
            r = b1.get("ratios", {}).get(arm)
            if r:
                for f in ("WEEKEND.md", "paper_draft.md", "README.md"):
                    out.append((f, f"{r['ratio_conservative']:.1f}×",
                                f"batch-1 {arm} ratio, fair denominator"))
        eag = b1.get("ratios", {}).get("eager")
        if eag:
            out.append(("WEEKEND.md",
                        f"{eag['ratio_vs_check_every_1_median']:.1f}×",
                        "the subsidized eager reading that would have passed"))
        b64 = fa["batches"].get("64", {}).get("ratios", {})
        if b64:
            bst = max(b64, key=lambda k: b64[k]["ratio_conservative"])
            for f in ("WEEKEND.md", "README.md"):
                out.append((f, f"{b64[bst]['ratio_conservative']:.1f}×",
                            "batch-64 batched reading, which does NOT meet 100x"))
        sw = fa.get("sample_sweep")
        if sw:
            g = sw["ratio_graph_conservative"]
            for f in ("WEEKEND.md", "README.md"):
                out.append((f, f"{g['n_ge_100x']}/{g['n']}",
                            "distinct batch-1 samples clearing 100x"))
                out.append((f, f"{g['min']:.1f}×",
                            "worst single-sample batch-1 ratio"))
                out.append((f, f"{g['median']:.1f}×",
                            "median single-sample batch-1 ratio"))
            # The pass is conditional on CUDA-graph capture and every prose
            # file that claims the pass must also carry what it is conditional
            # on -- that is the whole reason this row exists.
            ng = sw.get("ratio_nograd_conservative")
            if ng:
                for f in ("WEEKEND.md", "README.md"):
                    out.append((f, f"{ng['n_ge_100x']}/{ng['n']}",
                                "the same fields WITHOUT CUDA-graph capture"))
    if h13:
        sel = h13["selection"]
        for f in ("WEEKEND.md", "README.md"):
            out.append((f, f"{h13['no_inf_bound']:.2f}",
                        "the no-infinity bound on the density-ratio clip"))
            out.append((f, f"{h13['summary']['20']['mean_abstention_rate']*100:.1f}%",
                        "abstention rate at the shipped clip of 20"))
            out.append((f, f"{sel['held_out']['in_band_equivalent_of_n']:.2f}/"
                           f"{h13['n_eligible_shards']}",
                        "shifted shards in band, held-out clip selection"))
            out.append((f, f"{sel['failure_mode_by_clip'][sel['tuned_clip']]['under_0p88']}/"
                           f"{sel['failure_mode_by_clip'][sel['tuned_clip']]['n_cells']}",
                        "cells UNDER-covering once the abstention is removed"))
            out.append((f, f"p = {sel['vs_group_calibrator']['exact_sign_flip_p']:.4f}"
                        if False else
                        f"{sel['vs_group_calibrator']['exact_sign_flip_p']:.4f}",
                        "exact sign-flip p, weighted@clip vs the group calibrator"))
    if pe:
        su = pe["summary"]
        for f in ("WEEKEND.md", "README.md"):
            out.append((f, f"{su['n_identical']}/{su['n_shards']}",
                        "shards where the packed spectral path is "
                        "bit-identical to einsum"))
    if csu:
        rows = []
        for n_, v in csu["shards"].items():
            a_ = v["auroc"].get("combo")
            a_ = a_["auroc"] if isinstance(a_, dict) else a_
            if a_ is not None and v.get("rel_l2_in_dist"):
                rows.append((v["rel_l2_mean"] / v["rel_l2_in_dist"], a_))
        if rows:
            allc = [x for _, x in rows]
            n_ok = sum(1 for x in allc if x >= 0.9)
            out.append(("WEEKEND.md", f"{n_ok}/{len(allc)}",
                        "strict OOD reading, every shard"))
            miss = [d for d, x in rows if x < 0.9]
            if miss:
                thr = max(miss)
                above = [x for d, x in rows if d > thr]
                out.append(("WEEKEND.md",
                            f"{sum(1 for x in above if x >= 0.9)}/{len(above)}",
                            "conditional OOD reading, shifts that degrade the "
                            "model"))

    if lp:
        blind = [k for k, r in lp["shifts"].items()
                 if r["k_for_95pct_power"] == 1 and r["kind"] in
                 ("unseen_operator", "param_oor")]
        if blind:
            out.append(("WEEKEND.md", "**k = 1**", "labelled probes needed"))
    # --- H14: the gated certificate, and the oracle that also fails -------
    if selj:
        hs = selj["headline_score"]
        sg = selj["by_gate"]["sigma"]["by_score"][hs]
        out.append(("WEEKEND.md", f"0/{sg['n_covariate_shards']}",
                    "shards in band under the gated certificate"))
        out.append(("WEEKEND.md",
                    f"{sg['abstention_median_over_shards']:.3f}",
                    "median abstention of the shipped gate"))

    # --- H15/H16/H17 -------------------------------------------------------
    if scj and scj.get("h15"):
        L = scj["h15"]["lomo"]
        out.append(("WEEKEND.md", f"{L['in_band_median']:g}/{L['n_shards']}",
                    "H15 leave-one-mechanism-out shards in band, median"))
        out.append(("WEEKEND.md",
                    f"{L['in_band_range'][0]}\u2013{L['in_band_range'][1]}",
                    "H15 seed range, in-band shards"))
        out.append(("WEEKEND.md",
                    f"{L['vs_ungated']['exact_sign_flip_p']:.4f}",
                    "H15 sign-flip p against the ungated arm"))
        out.append(("WEEKEND.md",
                    f"{L['vs_insample_ceiling']['exact_sign_flip_p']:.4f}",
                    "H15 sign-flip p against its in-sample ceiling"))
    if scj and scj.get("wtol"):
        fm = scj["wtol"]["by_score"]["field_max"]
        nr = scj["wtol"]["by_score"]["norm_ratio"]
        out.append(("WEEKEND.md",
                    f"\u00b1{fm['tol_rel_median_over_shards']*50:.1f}%",
                    "H16 required width accuracy (half the tolerance)"))
        out.append(("WEEKEND.md",
                    f"{nr['tol_rel_median_over_shards']*100:.2f}%",
                    "H16 tolerance for norm_ratio"))
        e = fm.get("equivariant")
        if e:
            out.append(("WEEKEND.md", f"{e['required_range_base']:.1f}\u00d7",
                        "H17 required range before the repair"))
            out.append(("WEEKEND.md", f"{e['required_range_eq']:.1f}\u00d7",
                        "H17 required range after the repair"))
            out.append(("WEEKEND.md",
                        f"{e['vs_base']['exact_sign_flip_p']:.4f}",
                        "H17 sign-flip p, ungated in-band vs baseline"))
            # The registered control is the claim most damaging if it silently
            # flipped, so it is asserted here rather than merely rendered.
            c_ = e.get("control_darcy_amp2")
            if c_ and c_["change"] < 0.95:
                raise SystemExit(
                    f"H17 CONTROL VIOLATED: darcy_amp2's required width "
                    f"factor changed by {c_['change']:.3f} (< 0.95). The "
                    f"wrapper is improving a shard where no "
                    f"scale-equivariance exists, so it is doing something "
                    f"other than what H17 claims. Fix or withdraw the H17 "
                    f"result before publishing it.")

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
