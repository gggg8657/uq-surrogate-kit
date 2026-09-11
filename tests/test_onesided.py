"""H31: the one-sided reading, pinned so it cannot quietly replace the clause.

Three things have to stay true for a one-sided coverage number in this repo to
mean anything:

1. **Its constant came from the development suite, not from the test shards.**
   Every `c_dev_p*` is re-derived here from the dev s* values in the same file,
   so a future edit that fits c on the evaluation shards fails here.
2. **Every coverage cell carries a finite width multiplier.** H13's weighted
   arm read 100% coverage with 30/32 shards abstaining on an infinite interval;
   a one-sided reading with no width column is the same trap.
3. **The two readings stay separate.** R2 (the clause, two-sided) and R1 (the
   conventional one-sided reading) are both counted, and the documents may not
   quote an R1 count without saying it is one-sided.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BAND, TARGET = (0.88, 0.92), 0.90
ARMS = {"eq": "runs/oseq_u*_het.json", "base": "runs/osbase_u*_het.json"}


def _files():
    out = []
    for pat in ARMS.values():
        out += sorted(glob.glob(str(ROOT / pat)))
    return out


def test_dev_chosen_constants_rederive_from_the_dev_suite_alone():
    fs = _files()
    if not fs:
        print("skip: no --onesided runs present")
        return
    for f in fs:
        d = json.loads(Path(f).read_text())
        o = d["onesided"]
        assert o["enabled"] and o["clause_eligible"], f
        got = sorted(r["s_star"] for r in o["dev_shards"].values()
                     if r["s_star"] is not None)
        assert len(got) >= 20, f"{f}: only {len(got)} dev shards with an s*"
        for name, c in o["c_ladder"].items():
            if not name.startswith("c_dev_p"):
                continue
            pc = float(name[len("c_dev_p"):])
            want = float(np.percentile(got, pc))
            assert abs(c - want) < 1e-9 * max(1.0, want), \
                f"{f}: {name}={c} is not the {pc}th percentile of the dev " \
                f"s* values ({want}) -- it was fitted on something else"
        # every dev task name must be from the dev block, never an eval shard
        for task in o["dev_shards"]:
            assert task.startswith("dev_"), f"{f}: {task} is not a dev shard"
    print(f"ok  every c_dev_p* in {len(fs)} files re-derives from that file's "
          f"own dev s* values; no dev shard name is an evaluation shard")


def test_every_coverage_cell_carries_a_finite_width():
    fs = _files()
    if not fs:
        print("skip: no --onesided runs present")
        return
    n = 0
    for f in fs:
        d = json.loads(Path(f).read_text())
        recs = list(d["shards"].values()) + list(d["in_dist"].values())
        for r in recs:
            for name, cell in r["onesided"].items():
                w, c = cell["width_mult_median"], cell["c"]
                assert np.isfinite(w) and w > 0, (f, name, w)
                # width = median of (c*sigma+floor)/(sigma+floor): between 1
                # and c for c >= 1, and the floor keeps it strictly below c
                if c >= 1.0:
                    assert 1.0 - 1e-9 <= w <= c + 1e-9, (f, name, c, w)
                assert "coverage" in cell and "ci95" in cell
                n += 1
    print(f"ok  {n} one-sided cells, every one with a finite width multiplier "
          f"in [1, c]")


def test_the_protocol_is_the_same_one_that_produced_the_sstar_files():
    """oseq_u{k} must reproduce stargeq_u{k}'s s* exactly, and osbase/starget.

    The --onesided flag adds a dev-suite pass before the evaluation shards are
    touched. If it perturbed the frozen quantile, the calibration median or the
    floor, every s* would move. This is the regression test that says it did
    not.
    """
    pairs = [("runs/oseq_u%d_het.json", "runs/stargeq_u%d_het.json"),
             ("runs/osbase_u%d_het.json", "runs/starget_u%d_het.json")]
    checked = 0
    for new, old in pairs:
        for k in range(8):
            fn, fo = ROOT / (new % k), ROOT / (old % k)
            if not (fn.is_file() and fo.is_file()):
                continue
            a, b = json.loads(fn.read_text()), json.loads(fo.read_text())
            assert sorted(a["shards"]) == sorted(b["shards"]), (fn, fo)
            for name in a["shards"]:
                sa, sb = a["shards"][name]["s_star"], b["shards"][name]["s_star"]
                if sa is None or sb is None:
                    assert sa is sb is None, (fn, name)
                    continue
                assert abs(sa - sb) <= 1e-6 * max(sa, sb), \
                    f"{fn.name} vs {fo.name}: {name} s* {sa} vs {sb} -- the " \
                    f"--onesided pass changed the protocol"
            checked += 1
    if not checked:
        print("skip: no matching (onesided, starget) seed pairs on disk")
        return
    print(f"ok  {checked} seed pairs reproduce their s* files to 1e-6; the "
          f"dev-suite pass does not touch the frozen quantile")


def test_r1_and_r2_are_counted_separately_and_r1_is_never_the_clause():
    p = ROOT / "runs/h31_onesided.json"
    if not p.is_file():
        print("skip runs/h31_onesided.json absent")
        return
    d = json.loads(p.read_text())
    assert "R1" in d["readings"] and "R2" in d["readings"]
    assert "THE CLAUSE" in d["readings"]["R2"]
    for arm, A in d["arms"].items():
        if "c" not in A:
            continue
        assert A["consistency_sstar_vs_coverage"]["ok"], \
            f"{arm}: 'coverage >= 0.90 at c' disagrees with 's* <= c' by " \
            f"more than one shard: " \
            f"{A['consistency_sstar_vs_coverage']['disagreements']}"
        for cn, r in A["c"].items():
            assert r["R1_one_sided_ge_0p90"]["min"] >= \
                r["R2_two_sided_in_band"]["min"] - A["n_shards"], (arm, cn)
            assert r["dev_chosen"] == cn.startswith("c_dev_")
    print("ok  R1 and R2 counted separately in every cell; s*-vs-coverage "
          "consistent on every arm")


def test_documents_never_quote_an_r1_count_without_saying_one_sided():
    """A bare '23/24 shards in band' would read as the clause. It is not.

    Restricted to lines that are about coverage or shards: `x/24` also counts
    the 24 distinct coefficient fields in clause 2's timing table, and those
    are a different denominator entirely.
    """
    bad = []
    for doc in ("RESULTS.md", "README.md", "WEEKEND.md", "paper_draft.md"):
        p = ROOT / doc
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            low = line.lower()
            if not re.search(r"\b2[0-4]\s*/\s*24\b", line):
                continue
            if not ("shard" in low or "coverag" in low):
                continue        # x/24 in a timing row is a different count
            if ("one-sided" in low or "r1" in low or "conservative" in low
                    or "strict" in low or "two-sided" in low
                    or "≥ 0.90" in line or ">= 0.90" in line
                    or "oracle" in low or "s\\*" in line or "s*" in line):
                continue
            bad.append(f"{doc}:{i}: {line.strip()[:120]}")
    assert not bad, ("a 2x/24 count appears with no mark that it is the "
                     "one-sided reading:\n" + "\n".join(bad))
    print("ok  no document quotes a 2x/24 count without labelling the reading")


if __name__ == "__main__":
    test_dev_chosen_constants_rederive_from_the_dev_suite_alone()
    test_every_coverage_cell_carries_a_finite_width()
    test_the_protocol_is_the_same_one_that_produced_the_sstar_files()
    test_r1_and_r2_are_counted_separately_and_r1_is_never_the_clause()
    test_documents_never_quote_an_r1_count_without_saying_one_sided()
    print("all one-sided tests passed")
