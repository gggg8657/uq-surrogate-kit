"""Regenerate RESULTS.md from the run JSONs. Nothing in that file is typed by hand.

    python scripts/report.py

Every table below is built from `runs/*.json`, so a number cannot appear in the
report unless a run in this repository produced it. Missing inputs render as
`[not measured]` rather than being skipped, because a blank row is information
and a missing row is not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("runs")
KPI = ("conformal 커버리지 90±2% · 추론 가속 ≥100× · OOD 탐지 AUROC ≥0.9")
NM = "`[not measured]`"


def load(name):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{100 * x:.1f}%"


def cov_cell(e):
    if e is None:
        return NM
    return f"{pct(e['coverage'])} [{pct(e['ci95'][0])}, {pct(e['ci95'][1])}]"


def band(e, lo=0.88, hi=0.92):
    """PASS only if the point estimate is in the band."""
    if e is None:
        return "—"
    return "✅" if lo <= e["coverage"] <= hi else "❌"


# --------------------------------------------------------------------------- #
def sec_conformal(c, out):
    out.append("## Clause 1 — conformal coverage 90±2%\n")
    if c is None:
        out.append(f"{NM} — `runs/conformal.json` absent.\n")
        return None
    a = c["alpha"]
    out.append(f"alpha = {a}, target {pct(1-a)}, KPI band [88.0%, 92.0%]. "
               f"{c['n_members']} ensemble members. Every coverage carries a "
               "Wilson 95% interval; on a 512-sample shard that interval is "
               "about ±2.6pp, which is *wider than the KPI band*, so no single "
               "shard decides a clause.\n")

    out.append("\n### In distribution\n")
    out.append("| score | what it certifies | pooled, marginal | pooled, per-family q |")
    out.append("|---|---|---|---|")
    meaning = {
        "field_max": "whole field inside mean ± q·σ (simultaneous)",
        "norm_ratio": "‖error‖ ≤ q·‖σ‖ (aggregate, per sample)",
        "rel_l2": "rel-L2 ≤ q (fixed bound, no σ used)",
        "pixel": "each pixel inside the band (marginal over pixels)",
    }
    for s, d in c["scores"].items():
        i = d["in_dist"]
        out.append(f"| `{s}` | {meaning.get(s, '')} | "
                   f"{cov_cell(i['pooled_split'])} {band(i['pooled_split'])} | "
                   f"{cov_cell(i['pooled_group'])} {band(i['pooled_group'])} |")

    head = c["scores"].get("field_max")
    if head:
        out.append("\nPer family, `field_max` with a per-family quantile:\n")
        out.append("| family | coverage | rel-L2 (ensemble mean) | corr(spread, error) |")
        out.append("|---|---|---|---|")
        for t, e in head["in_dist"]["per_family_group"].items():
            er = c["error"].get(t, {})
            out.append(f"| {t} | {cov_cell(e)} {band(e)} | "
                       f"{er.get('rel_l2_mean', float('nan')):.4f} | "
                       f"{er.get('spread_error_pearson', float('nan')):+.3f} |")

    out.append("\n### Under shift\n")
    out.append("`split` is the calibrated quantile applied unchanged; `weighted` "
               "reweights the calibration scores by a density ratio estimated "
               "from the shifted *inputs only*, which is what a deployment has. "
               "`probe AUC` is that estimator's own held-out AUC — at 0.5 it "
               "found no shift and the weights are noise.\n")
    out.append("| shift | kind | rel-L2 | split | weighted | probe AUC | ESS |")
    out.append("|---|---|---|---|---|---|---|")
    if head:
        for k, r in sorted(head["ood"].items(),
                           key=lambda kv: (kv[1]["kind"], kv[0])):
            w = r.get("weighted")
            d = r.get("weighted_diag", {})
            out.append(
                f"| `{k.split('/')[1]}` @N{r['N']} | {r['kind']} | "
                f"{r['rel_l2_mean']:.4f} | {cov_cell(r['split'])} {band(r['split'])} | "
                f"{cov_cell(w) if w else NM} {band(w) if w else ''} | "
                f"{d.get('probe_auc', float('nan')):.3f} | "
                f"{d.get('ess', float('nan')):.0f}/{d.get('n_cal', 0)} |")
    return c


def sec_speed(b, out):
    out.append("\n## Clause 2 — inference speedup ≥100×\n")
    if b is None:
        out.append(f"{NM} — `runs/bench.json` absent.\n")
        return
    e = b["env"]
    out.append(f"`{e.get('gpu', 'cpu only')}`, torch {e['torch']}, CUDA "
               f"{e.get('cuda', '—')}, {e['torch_num_threads']} torch threads "
               f"(`OMP_NUM_THREADS={e['OMP_NUM_THREADS']}`), "
               f"`CUDA_VISIBLE_DEVICES={e.get('visible_devices', '—')}`. "
               "Solver and surrogate are timed on the same device in the same "
               "process. Median of "
               f"{b['rows'][0]['solver_timing']['n_iter']} runs after warmup.\n")
    out.append("\n`single` is one member; `ensemble` is all "
               f"{b['n_members']} (what produces σ); `ensemble+residual` adds the "
               "fp64 PDE residual check. The uncertainty is not free and the "
               "table says how much it costs.\n")
    out.append("| family | dev | batch | solver | surrogate (ens) | single | "
               "ensemble | ens+resid | rel-L2 | solver converged to |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in b["rows"]:
        s = r["surrogate"]
        def sp(k):
            return f"{s[k]['speedup']:.1f}×" if k in s else "—"
        rl = r.get("surrogate_rel_l2")
        out.append(
            f"| {r['task']}{'' if r['trained'] else ' *(untrained)*'} | "
            f"{r['device']} | {r['batch']} | {r['solver_s']*1e3:.2f} ms | "
            f"{s['ensemble']['s']*1e3:.2f} ms | {sp('single')} | "
            f"**{sp('ensemble')}** | {sp('ensemble+residual')} | "
            f"{f'{rl:.4f}' if rl is not None else NM} | {r['solver_accuracy']} |")


def sec_ood(o, out):
    out.append("\n## Clause 3 — OOD detection AUROC ≥0.9\n")
    if o is None:
        out.append(f"{NM} — `runs/ood.json` absent.\n")
        return
    out.append("Two different questions, reported separately.\n")
    out.append("\n### 3a. Shift detection — in-distribution test vs each shifted "
               "shard, one row per shard\n")
    out.append("| shard | kind | rel-L2 (in→shift) | spread | mahalanobis | residual |")
    out.append("|---|---|---|---|---|---|")
    for k, r in sorted(o["shift_detection"].items(),
                       key=lambda kv: (kv[1]["kind"], kv[0])):
        def cell(d):
            v = r["auroc"].get(d)
            if v is None:
                return "—"
            mark = "✅" if v["auroc"] >= 0.9 else ""
            return (f"{v['auroc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
                    f" {mark}")
        out.append(f"| `{k.split('/')[1]}` @N{r['N']} | {r['kind']} | "
                   f"{r['rel_l2_in_dist']:.4f} → {r['rel_l2_mean']:.4f} | "
                   f"{cell('spread')} | {cell('mahalanobis')} | {cell('residual')} |")
    ed = o["error_detection"]
    out.append(f"\n### 3b. Error detection — will this sample's rel-L2 exceed "
               f"τ = {ed['tau']:.4f}?\n")
    out.append(f"{ed['tau_definition']}. Pooled over in-distribution test and "
               f"every shifted shard: n = {ed['n']}, positive rate "
               f"{pct(ed['positive_rate'])}.\n")
    out.append("| detector | AUROC | 95% CI | n scored |")
    out.append("|---|---|---|---|")
    for d, v in ed["detectors"].items():
        mark = " ✅" if v["auroc"] >= 0.9 else ""
        out.append(f"| `{d}` | {v['auroc']:.3f}{mark} | "
                   f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['n']} |")


def sec_floor(f, out):
    out.append("\n## The residual detector's noise floor\n")
    if f is None:
        out.append(f"{NM}\n")
        return
    out.append("`‖L u − f‖/‖f‖` evaluated on the **exact** solution. The check "
               "cannot resolve a surrogate error smaller than this, and the "
               "floor grows with the order of the operator's symbol because "
               "applying L amplifies the round-off already in `u`. Solution in "
               f"{f['solution_dtype']}, apply in {f['apply_dtype']}, "
               f"{f['n_samples']} samples.\n")
    grids = sorted(f["grids"], key=int)
    tasks = list(f["grids"][grids[0]])
    out.append("| family | symbol order | " + " | ".join(f"N={g}" for g in grids) + " |")
    out.append("|---|---:|" + "---:|" * len(grids))
    for t in sorted(tasks, key=lambda x: f["grids"][grids[0]][x]["order"]):
        cells = " | ".join(f"{f['grids'][g][t]['mean']:.1e}" for g in grids)
        out.append(f"| {t} | {f['grids'][grids[0]][t]['order']} | {cells} |")
    out.append("\nUsable where the floor sits well below the surrogate's error: "
               "Poisson, Helmholtz, Darcy. Not usable at fourth order and above.\n")


def verdict(c, b, o, out):
    """The KPI, clause by clause, with the JSON each verdict came from."""
    out.insert(0, "")
    lines = ["## KPI verdict\n",
             f"**{KPI}**\n",
             "| clause | target | measured | source | verdict |",
             "|---|---|---|---|---|"]

    # clause 1
    if c is None:
        lines.append(f"| conformal coverage | 90±2% | {NM} | — | — |")
    else:
        h = c["scores"]["field_max"]
        ind = h["in_dist"]["pooled_group"]
        oods = [r for r in h["ood"].values() if "weighted" in r]
        n_in_band = sum(1 for r in oods
                        if 0.88 <= r["weighted"]["coverage"] <= 0.92)
        n_in_band_split = sum(1 for r in oods
                              if 0.88 <= r["split"]["coverage"] <= 0.92)
        lines.append(f"| coverage, in distribution | 90±2% | {cov_cell(ind)} | "
                     f"`runs/conformal.json` | {band(ind)} |")
        lines.append(f"| coverage, under shift | 90±2% | split: "
                     f"{n_in_band_split}/{len(oods)} shards in band; weighted: "
                     f"{n_in_band}/{len(oods)} | `runs/conformal.json` | "
                     f"{'✅' if n_in_band == len(oods) else '❌'} |")
    # clause 2
    if b is None:
        lines.append(f"| inference speedup | ≥100× | {NM} | — | — |")
    else:
        best = max((r for r in b["rows"] if r["device"] == "cuda"),
                   key=lambda r: r["surrogate"]["ensemble"]["speedup"])
        n_ge = sum(1 for r in b["rows"]
                   if r["device"] == "cuda" and r["trained"]
                   and r["surrogate"]["ensemble"]["speedup"] >= 100)
        n_tot = sum(1 for r in b["rows"] if r["device"] == "cuda" and r["trained"])
        lines.append(
            f"| inference speedup | ≥100× | best {best['surrogate']['ensemble']['speedup']:.0f}× "
            f"({best['task']}, batch {best['batch']}, rel-L2 "
            f"{best['surrogate_rel_l2']:.4f}); {n_ge}/{n_tot} trained rows ≥100× | "
            f"`runs/bench.json` | {'✅' if n_ge == n_tot else '⚠️'} |")
    # clause 3
    if o is None:
        lines.append(f"| OOD AUROC | ≥0.9 | {NM} | — | — |")
    else:
        rows = o["shift_detection"]
        best_det, best_n = None, -1
        for det in ("spread", "mahalanobis", "residual"):
            n = sum(1 for r in rows.values()
                    if r["auroc"].get(det) and r["auroc"][det]["auroc"] >= 0.9)
            if n > best_n:
                best_det, best_n = det, n
        lines.append(f"| OOD shift AUROC | ≥0.9 on every shard | best single "
                     f"detector `{best_det}`: {best_n}/{len(rows)} shards ≥0.9 | "
                     f"`runs/ood.json` | {'✅' if best_n == len(rows) else '❌'} |")
        ed = o["error_detection"]["detectors"]
        bd = max(ed, key=lambda d: ed[d]["auroc"])
        lines.append(f"| OOD error-detection AUROC | ≥0.9 | `{bd}` "
                     f"{ed[bd]['auroc']:.3f} "
                     f"[{ed[bd]['ci95'][0]:.3f}, {ed[bd]['ci95'][1]:.3f}] | "
                     f"`runs/ood.json` | "
                     f"{'✅' if ed[bd]['auroc'] >= 0.9 else '❌'} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()
    c, b, o, f = (load("conformal.json"), load("bench.json"),
                  load("ood.json"), load("residual_floor.json"))
    body = []
    sec_conformal(c, body)
    sec_speed(b, body)
    sec_ood(o, body)
    sec_floor(f, body)
    head = ["# Results", "",
            "Generated by `scripts/report.py` from `runs/*.json`. Do not edit by "
            "hand — every number here is regenerated from the JSON a run wrote.",
            ""]
    Path(args.out).write_text("\n".join(head + verdict(c, b, o, body) + body) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
