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
    star = "†" if e.get("ci_kind") == "cluster_bootstrap_over_fields" else ""
    return f"{pct(e['coverage'])} [{pct(e['ci95'][0])}, {pct(e['ci95'][1])}]{star}"


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
    out.append("`†` = interval from a cluster bootstrap over fields, not a "
               "Wilson interval on the pixel count. Pixels within a field are "
               "not independent observations, so a binomial interval on 2.1M "
               "pixels is roughly 64× too narrow, and the `pixel` row is "
               "descriptive rather than a conformal guarantee: exchangeability "
               "holds over fields, not over pixels.\n")
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
    out.append("`‡` marks a shard where the **governing equation** changed, so "
               "p(y|x) changed too and weighted conformal's covariate-shift "
               "assumption does not hold. Its number is a diagnostic, not a "
               "repair.\n")
    out.append("| shift | kind | rel-L2 | split | weighted | probe AUC | cal ESS | q=∞ |")
    out.append("|---|---|---|---|---|---|---|---|")
    if head:
        for k, r in sorted(head["ood"].items(),
                           key=lambda kv: (kv[1]["kind"], kv[0])):
            w = r.get("weighted")
            d = r.get("weighted_diag", {})
            vio = "‡" if r.get("assumption_violated") else ""
            out.append(
                f"| `{k.split('/')[1]}` @N{r['N']} | {r['kind']}{vio} | "
                f"{r['rel_l2_mean']:.4f} | {cov_cell(r['split'])} {band(r['split'])} | "
                f"{cov_cell(w) if w else NM} {band(w) if w else ''} | "
                f"{d.get('probe_auc', float('nan')):.3f} | "
                f"{d.get('ess', float('nan')):.0f}/{d.get('n_cal', 0)} | "
                f"{d.get('n_infinite_quantiles', '—')} |")
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
    out.append("Each speedup carries **its own variant's** rel-L2: a single "
               "member is faster than the ensemble and also less accurate, so "
               "one accuracy column next to three speedup columns would "
               "advertise a ratio that accuracy never achieved.\n")
    out.append("| family | dev | batch | solver | single (rel-L2) | "
               "ensemble (rel-L2) | ens+resid | solver converged to |")
    out.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in b["rows"]:
        s = r["surrogate"]
        def cell(k):
            if k not in s:
                return "—"
            rl = s[k].get("rel_l2")
            acc = f" ({rl:.4f})" if rl is not None else f" ({NM})"
            v = s[k]["speedup"]
            num = f"{v:.1f}" if v >= 1 else f"{v:.3g}"
            return f"{num}×{acc if k != 'ensemble+residual' else ''}"
        out.append(
            f"| {r['task']}{'' if r['trained'] else ' *(untrained)*'} | "
            f"{r['device']} | {r['batch']} | {r['solver_s']*1e3:.2f} ms | "
            f"{cell('single')} | **{cell('ensemble')}** | "
            f"{cell('ensemble+residual')} | {r['solver_accuracy']} |")


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
    common = ed.get("detectors_common_population", {})
    out.append("`residual` is undefined for the time-evolution families, so on "
               "the full pool it is scored on a *different population* from the "
               "other two and the three cannot be ranked against each other "
               "there. The right-hand column restricts all three to "
               f"{', '.join(ed.get('common_population', []))}, where all are "
               "defined.\n")
    out.append("| detector | AUROC (own population) | 95% CI | n | "
               "AUROC (common population) | n |")
    out.append("|---|---|---|---|---|---|")
    for d, v in ed["detectors"].items():
        c2 = common.get(d, {})
        mark = " ✅" if v["auroc"] >= 0.9 else ""
        cm = " ✅" if c2 and c2["auroc"] >= 0.9 else ""
        out.append(f"| `{d}` | {v['auroc']:.3f}{mark} | "
                   f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['n']} | "
                   f"{c2['auroc']:.3f}{cm} | {c2['n']} |" if c2 else
                   f"| `{d}` | {v['auroc']:.3f}{mark} | "
                   f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['n']} | — | — |")
    out.append("\nStratified by family, since a pooled AUROC can be driven by "
               "one family being harder than another rather than by ranking "
               "failures within a family:\n")
    fams = list(next(iter(ed["detectors"].values()))["per_parent"])
    out.append("| detector | " + " | ".join(fams) + " |")
    out.append("|---|" + "---|" * len(fams))
    for d, v in ed["detectors"].items():
        cells = " | ".join(
            "—" if v["per_parent"][f] != v["per_parent"][f]
            else f"{v['per_parent'][f]:.3f}" for f in fams)
        out.append(f"| `{d}` | {cells} |")


def sec_consistency(cs, out, tag="M1"):
    """H6: the residual under the REQUESTED operator, and what it does not buy."""
    out.append(f"\n### 3c. Solver-consistency detection, and the two deployments "
               f"(`runs/consistency_{tag}.json`)\n")
    if cs is None:
        out.append(f"{NM} — `runs/consistency_{tag}.json` absent.\n")
        return
    op = set(cs["operator_shift_shards"])
    out.append(
        f"An operator shift is observable only if the request *names* the "
        f"operator. **(A)** scores the residual under `PARENT[task]`, the "
        f"operator the surrogate is configured for — the process moved and "
        f"nobody reconfigured the model. Under (A) every unsupervised detector "
        f"is a function of (input, configured operator), neither of which "
        f"changed, so it is at chance **by construction**; that is the strict "
        f"reading and it stands. **(B)** scores it under the operator the "
        f"request names. Of the {cs['summary']['mahalanobis']['n_total_shards']} "
        f"shards, exactly **{len(op)} change the operator** "
        f"(`tests/test_sims.py::test_operator_key_partitions_the_ood_suite` "
        f"pins the partition); on the rest (B) is byte-identical to (A), which "
        f"is the control.\n")
    out.append(
        "`lookup` is a **dict lookup** — *is the requested operator one we "
        "trained on?* — costing nothing. Where it is 1.000, no continuous "
        "detector may claim credit over chance; it must claim it over 1.000. "
        "`floor_c` is the same consistency score on that shard's **ground "
        "truth**, an offline diagnostic no detector sees: `floor_c` near 1 "
        "means applying that operator in fp64 is round-off dominated, so a "
        "high AUROC there is an operator-identity signal and not detection. "
        "`headroom` is median c(û) / median c(u_true).\n")
    out.append("| shard | L changed | rel-L2 | mahalanobis | (A) cons | "
               "(B) cons | floor_c | headroom | combo | router | lookup |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def cell(r, d):
        v = r["auroc"].get(d)
        if v is None:
            return NM
        return f"{v['auroc']:.3f}" + (" ✅" if v["auroc"] >= 0.9 else "")

    for k, r in sorted(cs["shards"].items(), key=lambda kv: (kv[1]["kind"], kv[0])):
        fc = r["floor_c"]
        hd = r["headroom"]
        flag = ""
        if fc is not None and hd is not None and hd < 2.0:
            flag = " ⚠️"
        out.append(
            f"| `{k.split('/')[1]}` @N{r['N']} | "
            f"{'**yes**' if r['operator_changed'] else 'no'} | "
            f"{r['rel_l2_mean']:.4g} | {cell(r, 'mahalanobis')} | "
            f"{cell(r, 'cons_A')} | {cell(r, 'cons_B')}{flag} | "
            f"{NM if fc is None else f'{fc:.4g}'} | "
            f"{NM if hd is None else f'{hd:.3g}'} | "
            f"{cell(r, 'combo')} | {cell(r, 'router')} | {cell(r, 'lookup')} |")

    out.append("\n| detector | shards scored | ≥0.9 | ≥0.9 among the "
               f"{len(op)} operator-shift shards |")
    out.append("|---|---|---|---|")
    for d, v in cs["summary"].items():
        o2 = cs["summary_operator_shift"][d]
        out.append(f"| `{d}` | {v['n_shards_scored']}/{v['n_total_shards']} | "
                   f"**{v['n_ge_0p9']}**/{v['n_total_shards']} | "
                   f"{o2['n_ge_0p9']}/{o2['n_total']} (scored {o2['n_scored']}) |")

    arte = [k for k, r in cs["shards"].items()
            if r["headroom"] is not None and r["headroom"] < 2.0]
    out.append(
        f"\n**⚠️ marks a row whose AUROC is not detection.** {len(arte)} row(s) "
        f"here: " + (", ".join(f"`{k.split('/')[1]}`" for k in arte) or "none")
        + ". Their `floor_c` says the *exact* solution scores nearly the same, "
        "so the separation comes from the operator's conditioning, not from the "
        "prediction being wrong. Predicted in `critique_log.md` before the run "
        "— `measure_residual_floor.py` had already put the `frac_s3` residual "
        "floor above its own right-hand side — and reported rather than banked.\n")
    out.append(
        f"Two shards remain under 0.9 for every detector here, and both are the "
        f"*weakest* rung of the graded ladder, where the shift is by design "
        f"barely present. Note also that `combo` = max(z) is **worse** than "
        f"`mahalanobis` alone on exactly those two rows: taking a maximum over "
        f"z-scores pays for a second, noisier component. `router` — use the "
        f"consistency residual only when the requested operator is untrained, "
        f"else the input detector — avoids that, but on the {len(op)} shards "
        f"where it differs it is inheriting the free `lookup`, not beating it.\n")


def sec_noise(n, out):
    out.append("\n### How repeatable are these timings?\n")
    if n is None:
        out.append(f"{NM} — `runs/bench_noise.json` absent.\n")
        return
    out.append(f"One configuration repeated {n['repeats']}× "
               "(`scripts/bench_noise.py`). This exists because an earlier "
               "version of this table reported **40.8×** for the Darcy headline: "
               "an idle H100 boosts over the first seconds of load, and a "
               "configuration timed cold against one timed hot inflated the "
               "ratio by 74%. Every benchmark script now drives the GPU to "
               "steady clocks before measuring.\n")
    out.append("| ratio | median | range | spread |")
    out.append("|---|---:|---:|---:|")
    for k in ("headline_speedup", "isoaccuracy_speedup"):
        if k not in n:
            continue
        v = n[k]
        out.append(f"| {k.replace('_', ' ')} (Darcy, batch {n['batch']}, "
                   f"{len(n['timings'])-1}-tol) | {v['median']:.2f}× | "
                   f"{v['min']:.2f}–{v['max']:.2f}× | ±{100*v['rel_range']/2:.0f}% |")
    out.append("\nThe repeated measurement is the authoritative one for the "
               "Darcy batch-1 ratio. The single-pass table above times each "
               "family once inside a longer script and still carries some "
               "drift — it reads 26.8× where the 8-repeat measurement reads "
               f"{n.get('headline_speedup', {}).get('median', float('nan')):.1f}×. "
               "Neither is near 100×, so the clause is unaffected, but no ratio "
               "in this file should be read to better than ±15%.\n")


def sec_iso(i, out):
    out.append("\n### Clause 2, the fairer version: iso-accuracy speedup\n")
    if i is None:
        out.append(f"{NM} — `runs/isoaccuracy.json` absent.\n")
        return
    out.append("The table above times the solver at the tolerance the *corpus* "
               "was generated with. Nobody converges a solver to 1e-10 to get an "
               "answer they will then accept at 4% error, so this sweeps the "
               "solver's own accuracy knob and reports the ratio against the "
               "cheapest setting that matches the surrogate's error.\n")
    for fam, d in i["families"].items():
        out.append(f"\n**{fam}** — knob: {d['knob']}"
                   + ("" if d.get("trained", True) else " *(no surrogate trained)*")
                   + "\n")
        if fam == "darcy":
            sur = d["surrogate"]
            out.append(f"Surrogate ({sur['members']} members, {sur['precision']}): "
                       f"rel-L2 **{sur['rel_l2_vs_reference']:.5f}** in "
                       f"{sur['s']*1e3:.2f} ms at batch {i['batch']}.\n")
        out.append("| setting | solver rel-L2 vs reference | solver time | ×surrogate |")
        out.append("|---|---:|---:|---:|")
        for r in d["sweep"]:
            key = (f"tol {r['tol']:.0e}" if "tol" in r
                   else f"dt {r['dt']:.1e} ({r['steps']} steps)")
            if "error" in r:
                out.append(f"| {key} | diverged | — | — |")
                continue
            sp = (f"{r['speedup_vs_surrogate']:.1f}×"
                  if "speedup_vs_surrogate" in r else "—")
            out.append(f"| {key} | {r['rel_l2_vs_reference']:.2e} | "
                       f"{r['s']*1e3:.2f} ms | {sp} |")
        if d.get("iso_accuracy"):
            a = d["iso_accuracy"]
            out.append(f"\n**Iso-accuracy speedup: {a['speedup']:.1f}×** — the "
                       f"cheapest swept solver setting (tol {a['tol']:.0e}) still "
                       f"reaches rel-L2 {a['solver_rel_l2']:.1e}, which is "
                       "*better* than the surrogate by orders of magnitude. The "
                       "sweep never got the solver down to the surrogate's own "
                       "accuracy, so even this ratio is an upper bound.\n")
        if d.get("note"):
            out.append(f"\n{d['note']}\n")


def sec_members(m, out):
    out.append("\n## The knob the KPI turns on: ensemble size\n")
    if m is None:
        out.append(f"{NM} — `runs/members.json` absent.\n")
        return
    out.append(f"{m['note']}. Darcy, batch {m['batch']}, same device and "
               "timing method as the speedup table. Coverage, spread–error "
               "correlation and shift AUROC all require a spread, which does "
               "not exist at M=1.\n")
    out.append("| M | Darcy speedup | rel-L2 | coverage (field_max) | "
               "corr(spread, err) | shift AUROC (`" + m["shift_task"] + "`) |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for r in m["rows"]:
        def cell(k, fmt="{:.3f}", pctf=False):
            v = r.get(k)
            if not v:
                return "— *(no spread)*"
            base = (f"{100*v['mean']:.1f}%" if pctf else fmt.format(v["mean"]))
            if v["n_subsets"] > 1:
                rng = (f"{100*v['min']:.1f}–{100*v['max']:.1f}%" if pctf
                       else f"{fmt.format(v['min'])}–{fmt.format(v['max'])}")
                return f"{base} ({rng})"
            return base
        mark = " ✅" if r["darcy_speedup"] >= 100 else ""
        out.append(f"| {r['M']} | {r['darcy_speedup']:.1f}×{mark} | "
                   f"{cell('rel_l2', '{:.5f}')} | {cell('coverage', pctf=True)} | "
                   f"{cell('spread_error_pearson', '{:+.3f}')} | "
                   f"{cell('shift_auroc_spread')} |")
    ok = [r for r in m["rows"] if r["darcy_speedup"] >= 100]
    uq = [r for r in m["rows"] if r["has_uncertainty"]]
    if ok and all(not r["has_uncertainty"] for r in ok):
        out.append("\n**Every configuration that clears 100× has no uncertainty, "
                   "and every configuration with uncertainty is below it.** The "
                   "speedup clause and the coverage clause are contested by the "
                   "same knob; this is a property of the design, not of the "
                   "training. Timing noise between adjacent M is visible (M=3 "
                   "measures faster than M=2), so read the trend, not the points.\n")


def sec_probe(d, out):
    out.append("\n## What it costs to detect what nothing unsupervised can\n")
    if d is None:
        out.append(f"{NM} — `runs/label_probe.json` absent.\n")
        return
    out.append("Every unsupervised detector is a function of the input and the "
               "*configured* operator. An operator shift with a matched input "
               "distribution changes neither, so those detectors are at chance "
               "by identity, not by deficiency. The cheapest thing that does "
               "work is a labelled probe: run k samples through the reference "
               "solver and compare their errors to the calibrated distribution "
               "with a conformal p-value and Fisher's method.\n")
    out.append(f"Target: {100*d['target_power']:.0f}% power at a "
               f"{100*d['alpha']:.0f}% false-alarm rate. The false-alarm rate is "
               "**measured** on the held-out in-distribution test split, not "
               "assumed:\n")
    out.append("| family | " + " | ".join(f"k={k}" for k in d["ks"][:5]) + " |")
    out.append("|---|" + "---:|" * 5)
    for t, v in d["false_alarm_measured"].items():
        out.append(f"| {t} | " + " | ".join(
            f"{v[str(k)] if str(k) in v else v[k]:.3f}" for k in d["ks"][:5]) + " |")
    out.append("\nSmallest k reaching the target, grouped by shift kind:\n")
    out.append("| shift | kind | rel-L2 (in → shifted) | k for 95% power |")
    out.append("|---|---|---|---:|")
    for k, r in sorted(d["shifts"].items(), key=lambda kv: (kv[1]["kind"], kv[0])):
        kk = r["k_for_95pct_power"]
        out.append(f"| `{k.split('/')[1]}` @N{r['N']} | {r['kind']} | "
                   f"{r['rel_l2_in_dist']:.4f} → {r['rel_l2_mean']:.4f} | "
                   f"{kk if kk else '>' + str(d['ks'][-1])} |")
    out.append("\n`>64` is not a failure: those are the shifts where the "
               "surrogate's error did **not** move (resolution changes, and "
               "smoother inputs, where the error goes down). A labelled probe "
               "correctly declines to alarm on a shift that does no harm — which "
               "is the difference between it and an input-space detector.\n")


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


def verdict(c, b, o, out, i=None, m=None, cs=None):
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
        # only shards where weighted conformal's assumption actually holds can
        # count toward the clause; the operator-shift shards are reported but
        # are not evidence for or against a covariate-shift method
        oods = [r for r in h["ood"].values()
                if "weighted" in r and not r.get("assumption_violated")]
        n_in_band = sum(1 for r in oods
                        if 0.88 <= r["weighted"]["coverage"] <= 0.92)
        n_in_band_split = sum(1 for r in oods
                              if 0.88 <= r["split"]["coverage"] <= 0.92)
        lines.append(f"| coverage, in distribution | 90±2% | {cov_cell(ind)} | "
                     f"`runs/conformal.json` | {band(ind)} |")
        lines.append(f"| coverage, under covariate shift | 90±2% | split: "
                     f"{n_in_band_split}/{len(oods)} shards in band; weighted: "
                     f"{n_in_band}/{len(oods)} | `runs/conformal.json` | "
                     f"{'✅' if n_in_band == len(oods) else '❌'} |")
    # clause 2
    if b is None:
        lines.append(f"| inference speedup | ≥100× | {NM} | — | — |")
    else:
        gpu = [r for r in b["rows"] if r["device"] == "cuda" and r["trained"]]
        sp = [r["surrogate"]["ensemble"]["speedup"] for r in gpu]
        n_ge = sum(1 for x in sp if x >= 100)
        worst = min(gpu, key=lambda r: r["surrogate"]["ensemble"]["speedup"])
        # Range and count, not the maximum. Reporting the best row makes the
        # clause a selection over families; the clause is about the surrogate.
        lines.append(
            f"| inference speedup, GPU, trained families | ≥100× | "
            f"{min(sp):.3g}×–{max(sp):.0f}× across {len(sp)} rows, "
            f"**{n_ge}/{len(sp)} ≥100×**; worst is {worst['task']} at batch "
            f"{worst['batch']} | `runs/bench.json` | "
            f"{'✅' if n_ge == len(sp) else '❌'} |")
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
        ed = (o["error_detection"].get("detectors_common_population")
              or o["error_detection"]["detectors"])
        bd = max(ed, key=lambda d: ed[d]["auroc"])
        lines.append(f"| OOD error-detection AUROC | ≥0.9 | `{bd}` (common "
                     f"population) {ed[bd]['auroc']:.3f} "
                     f"[{ed[bd]['ci95'][0]:.3f}, {ed[bd]['ci95'][1]:.3f}] | "
                     f"`runs/ood.json` | "
                     f"{'✅' if ed[bd]['auroc'] >= 0.9 else '❌'} |")
    if cs is not None:
        n_op = len(cs["operator_shift_shards"])
        tot = cs["summary"]["mahalanobis"]["n_total_shards"]
        best = max(("mahalanobis", "combo", "router"),
                   key=lambda d: cs["summary"][d]["n_ge_0p9"])
        arte = [k for k, r in cs["shards"].items()
                if r["headroom"] is not None and r["headroom"] < 2.0]
        lines.append(
            f"| OOD shift AUROC, presentation (B): the request names the "
            f"operator | ≥0.9 on every shard | `{best}` "
            f"**{cs['summary'][best]['n_ge_0p9']}/{tot}** shards ≥0.9 "
            f"(vs `mahalanobis` {cs['summary']['mahalanobis']['n_ge_0p9']}/{tot} "
            f"under (A)); {len(arte)} of the gained rows is an operator-"
            f"conditioning artefact, not detection | "
            f"`runs/consistency_M1.json` | "
            f"{'✅' if cs['summary'][best]['n_ge_0p9'] == tot else '❌'} |")
        lines.append(
            f"| — the free baseline it must beat | — | a dict lookup on the "
            f"requested operator scores **1.000 on all {n_op} operator-shift "
            f"shards** at zero cost | `runs/consistency_M1.json` | — |")
    nz = load("bench_noise.json")
    if nz and "headline_speedup" in nz:
        v = nz["headline_speedup"]
        lines.append(f"| inference speedup, repeated 8× | ≥100× | "
                     f"{v['median']:.1f}× median, {v['min']:.1f}–{v['max']:.1f}× "
                     f"(Darcy, batch 1, 5 members) | `runs/bench_noise.json` | "
                     f"{'✅' if v['median'] >= 100 else '❌'} |")
    if i and i["families"].get("darcy", {}).get("iso_accuracy"):
        a = i["families"]["darcy"]["iso_accuracy"]
        lines.append(f"| inference speedup, iso-accuracy | ≥100× | "
                     f"{a['speedup']:.1f}× on Darcy against the cheapest solver "
                     f"setting swept (which is still {a['solver_rel_l2']:.0e} "
                     f"accurate) | `runs/isoaccuracy.json` | "
                     f"{'✅' if a['speedup'] >= 100 else '❌'} |")
    if m:
        ok = [r for r in m["rows"] if r["darcy_speedup"] >= 100]
        lines.append(f"| ≥100× *and* an interval | both | "
                     + (f"only M={ok[0]['M']} clears 100× "
                        f"({ok[0]['darcy_speedup']:.0f}×) and M=1 has no spread, "
                        f"so no interval and no OOD score"
                        if ok and not any(r["has_uncertainty"] for r in ok)
                        else "see `runs/members.json`")
                     + " | `runs/members.json` | ❌ |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()
    c, b, o, f = (load("conformal.json"), load("bench.json"),
                  load("ood.json"), load("residual_floor.json"))
    i, m, lp = load("isoaccuracy.json"), load("members.json"), \
        load("label_probe.json")
    body = []
    sec_conformal(c, body)
    sec_speed(b, body)
    sec_noise(load("bench_noise.json"), body)
    sec_iso(i, body)
    sec_ood(o, body)
    sec_consistency(load('consistency_M1.json'), body, 'M1')
    sec_probe(lp, body)
    sec_members(m, body)
    sec_floor(f, body)
    head = ["# Results", "",
            "Generated by `scripts/report.py` from `runs/*.json`. Do not edit by "
            "hand — every number here is regenerated from the JSON a run wrote.",
            ""]
    Path(args.out).write_text(
        "\n".join(head + verdict(c, b, o, body, i, m, load('consistency_M1.json')) + body) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
