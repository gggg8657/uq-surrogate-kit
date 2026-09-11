"""What the H17 equivariance wrapper costs clause 2, with the guard that makes
the number quotable.

    ~/miniforge3/envs/pdeno/bin/python scripts/agg_h17_cost.py \
        --base runs/bench_fair_h17_base.json \
        --eq   runs/bench_fair_h17_eq.json \
        --out  runs/h17_cost.json

This repository has already published, and withdrawn, one version of this
number. The withdrawn version measured the wrapper's overhead on the **eager**
`predict_shard_single` path and divided the clause-2 figure -- measured under
**CUDA-graph replay** -- by (1 + overhead). That is arithmetic across two
protocols, and it produced a confident answer that meant nothing. The cell has
read `[not measured]` since.

So this script's job is as much refusal as arithmetic. It emits a delta only
when the two arms are the same measurement in every respect except the wrapper,
and it names the mismatch when they are not:

* same GPU (a ratio measured across two devices is not a ratio),
* same task, checkpoint, uncertainty head, batch list, trial count, tolerance,
* same **fair denominator** configuration -- the solver setting the ratio is
  taken against. If the two arms chose different solver settings, their ratios
  are against different denominators and the delta between them is not the
  wrapper's cost.

The overhead is reported **per execution path**. `eager` and `graph` are
different protocols and their overheads differ; combining one arm's path with
the other's is the exact error above. Each arm's clause-2 ratio is recomputed
against **its own** denominator, never against the other's and never against
the published figure from a different run.

`quotable_against_published` is always `false` and carries its reason, because
the published 117.0x came from a different run on a different device: the
comparison this file supports is eq-vs-base *within this pair*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: fields that must agree, or the two arms are not the same measurement
MUST_MATCH = ("task", "uq_source", "trials", "tol", "seed")
DENOM_KEYS = ("check_every", "fast_apply", "precond")
#: how much the two arms' surrogate accuracy may differ before the comparison
#: is refused. The wrapper is near-identity on in-distribution inputs, so this
#: is tight on purpose: anything larger means the arms are not solving the same
#: problem, and the brief's rule is that a speedup is quoted with the error it
#: was achieved at.
ACC_TOL = 0.01


def _paths(entry):
    """{path: median seconds} for whichever surrogate paths this arm recorded."""
    return {k: v["median_s"] for k, v in entry.get("surrogate", {}).items()
            if isinstance(v, dict) and "median_s" in v}


def compare(base, eq):
    mismatches = []
    for k in MUST_MATCH:
        if base.get(k) != eq.get(k):
            mismatches.append(f"{k}: base={base.get(k)!r} eq={eq.get(k)!r}")
    bg = base.get("env", {}).get("gpu")
    eg = eq.get("env", {}).get("gpu")
    if bg != eg:
        mismatches.append(f"env.gpu: base={bg!r} eq={eg!r}")
    if sorted(base.get("batches", {})) != sorted(eq.get("batches", {})):
        mismatches.append("batch lists differ")
    # A speedup at a different accuracy is not the same speedup. On this task
    # the wrapper is near-identity (its reference scale is the calibration
    # median), so a material change here means the arms are not solving the
    # same problem and the ratio is not the wrapper's cost.
    br, er = base.get("surrogate_rel_l2"), eq.get("surrogate_rel_l2")
    if br and er and abs(er / br - 1.0) > ACC_TOL:
        mismatches.append(
            f"surrogate_rel_l2 differs by {100 * (er / br - 1.0):+.2f}% "
            f"(base={br:.6g}, eq={er:.6g}); a ratio at a different accuracy "
            f"is not a ratio at the same accuracy")

    out = {"mismatches": mismatches,
           "comparable": not mismatches,
           "gpu": bg, "task": base.get("task"),
           "base_rel_l2": base.get("surrogate_rel_l2"),
           "eq_rel_l2": eq.get("surrogate_rel_l2"),
           "rel_l2_ratio_eq_over_base": (
               (eq.get("surrogate_rel_l2") / base.get("surrogate_rel_l2"))
               if base.get("surrogate_rel_l2") else None),
           "accuracy_tol": ACC_TOL,
           "quotable_against_published": False,
           "quotable_note": (
               "false by construction. The published clause-2 figure comes "
               "from a different run on a different device; this pair supports "
               "eq-vs-base WITHIN itself. Multiplying the overhead here into "
               "that figure would repeat the withdrawn error of combining an "
               "eager-path overhead with a graph-path speedup."),
           "by_batch": {}}
    if mismatches:
        return out

    for b in sorted(base["batches"], key=int):
        bb, be = base["batches"][b], eq["batches"][b]
        bd = {k: bb.get("fair_denominator", {}).get(k) for k in DENOM_KEYS}
        ed = {k: be.get("fair_denominator", {}).get(k) for k in DENOM_KEYS}
        rec = {"denominator_base": bd, "denominator_eq": ed,
               "denominator_matches": bd == ed,
               "solver_median_s_base":
                   bb.get("fair_denominator", {}).get("median_s"),
               "solver_median_s_eq":
                   be.get("fair_denominator", {}).get("median_s"),
               "by_path": {}}
        pb, pe = _paths(bb), _paths(be)
        for path in sorted(set(pb) & set(pe)):
            sb, se = pb[path], pe[path]
            cell = {
                "surrogate_median_s_base": sb,
                "surrogate_median_s_eq": se,
                "overhead_factor": (se / sb) if sb else None,
                "overhead_pct": ((se / sb - 1.0) * 100.0) if sb else None,
            }
            # each arm's ratio against ITS OWN denominator, never the other's
            for tag, arm, s in (("base", bb, sb), ("eq", be, se)):
                den = arm.get("fair_denominator", {}).get("median_s")
                cell[f"clause2_ratio_{tag}"] = (den / s) if (den and s) else None
            if not rec["denominator_matches"]:
                cell["ratio_delta_withheld"] = (
                    "the two arms took their ratios against different solver "
                    "settings, so the difference between the ratios is not "
                    "the wrapper's cost")
            rec["by_path"][path] = cell
        out["by_batch"][b] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/bench_fair_h17_base.json")
    ap.add_argument("--eq", default="runs/bench_fair_h17_eq.json")
    ap.add_argument("--out", default="runs/h17_cost.json")
    args = ap.parse_args()

    missing = [p for p in (args.base, args.eq) if not Path(p).exists()]
    if missing:
        res = {"status": "[not measured]", "missing": missing,
               "note": ("both arms must exist. A one-armed cost figure is not "
                        "a cost figure.")}
        print(f"[not measured] missing: {missing}")
    else:
        base = json.loads(Path(args.base).read_text())
        eq = json.loads(Path(args.eq).read_text())
        res = {"status": "measured", "base": args.base, "eq": args.eq,
               **compare(base, eq)}
        if not res["comparable"]:
            print("REFUSED: arms are not the same measurement")
            for m in res["mismatches"]:
                print(f"  mismatch  {m}")
        else:
            for b, rec in res["by_batch"].items():
                for path, c in rec["by_path"].items():
                    o = c["overhead_pct"]
                    print(f"  batch {b:>3s} {path:<7s} overhead "
                          f"{o:+7.1f}%   clause-2 ratio "
                          f"{c['clause2_ratio_base'] or float('nan'):8.1f}x -> "
                          f"{c['clause2_ratio_eq'] or float('nan'):8.1f}x"
                          + ("" if rec["denominator_matches"]
                             else "   [denominators differ: delta withheld]"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
