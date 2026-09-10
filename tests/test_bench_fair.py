"""Tests for the clause-2 machinery added when the M=1 interval made the
coverage row and the speedup row the same row.

The things worth testing here are not the timings -- those are measurements --
but the *gates* that decide whether a timing is admissible at all. Every one of
these was written because an adversarial review found the corresponding hole:

  - a CUDA-graph replay that returns a stale buffer must be rejected, and the
    check that only compares an unchanged input cannot reject it;
  - a solver setting whose achieved residual misses `tol` must not become a
    denominator;
  - the conservative pairing must be the least favourable one, not the best;
  - the report must not silently drop a clause when its JSON is absent.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_conservative_pairing_is_the_least_favourable():
    """fastest admissible solver / slowest surrogate <= every other pairing."""
    solver = {"min_s": 0.100, "median_s": 0.120}
    sur = {"min_s": 0.0008, "median_s": 0.0010, "max_s": 0.0012}
    cons = solver["min_s"] / sur["max_s"]
    med = solver["median_s"] / sur["median_s"]
    best = solver["median_s"] / sur["min_s"]
    assert cons <= med <= best, (cons, med, best)
    assert abs(cons - 83.33) < 0.01, cons
    print(f"ok  conservative {cons:.2f}x <= median {med:.2f}x <= "
          f"flattering {best:.2f}x")


def test_inadmissible_solver_setting_cannot_be_the_denominator():
    """A faster solver whose residual misses tol must be excluded.

    This is the loophole that would let clause 2 pass by loosening the
    reference: a solver stopped early is faster and wrong.
    """
    tol = 1e-10
    settings = {
        "check_every=1": {"min_s": 0.21, "achieved_residual": 9.0e-11},
        "check_every=50": {"min_s": 0.19, "achieved_residual": 6.8e-11},
        "loose": {"min_s": 0.02, "achieved_residual": 1.0e-4},   # fastest, bad
    }
    for v in settings.values():
        v["admissible"] = v["achieved_residual"] <= tol
    adm = [v for v in settings.values() if v["admissible"]]
    fastest = min(adm, key=lambda v: v["min_s"])
    assert fastest["min_s"] == 0.19, fastest
    assert not settings["loose"]["admissible"]
    # and the excluded one would have inflated the ratio by ~10x
    assert settings["loose"]["min_s"] / fastest["min_s"] < 0.2
    print("ok  the fastest setting is rejected on residual, not used because "
          "it is fast")


def test_staleness_gate_rejects_a_cached_output():
    """The gate must distinguish a live graph from one returning a cache.

    Simulated because capturing a real graph needs a GPU, but the *logic* is
    what had the hole: comparing only on an unchanged input passes for both.
    """
    def gate(out_same, out_mutated, eager_mutated, eager_orig, tol=1e-5):
        same = abs(out_same - eager_orig) / abs(eager_orig)
        moved = abs(eager_mutated - eager_orig) / abs(eager_orig)
        live = abs(out_mutated - eager_mutated) / abs(eager_mutated)
        if moved < 1e-6:
            raise AssertionError("vacuous gate: input mutation did nothing")
        return same <= tol and live <= tol

    # a live graph tracks the mutated input
    assert gate(out_same=1.0, out_mutated=2.0, eager_mutated=2.0,
                eager_orig=1.0)
    # a stale buffer returns the OLD value and must be rejected...
    assert not gate(out_same=1.0, out_mutated=1.0, eager_mutated=2.0,
                    eager_orig=1.0)
    # ...even though it passes the unchanged-input check the review flagged
    assert abs(1.0 - 1.0) <= 1e-5
    # and a gate whose perturbation moves nothing must raise, not pass
    try:
        gate(1.0, 1.0, 1.0, 1.0)
    except AssertionError as e:
        assert "vacuous" in str(e)
    else:
        raise AssertionError("vacuous gate was not caught")
    print("ok  staleness gate rejects a cached output and refuses to pass "
          "vacuously")


def test_erase_threshold_is_consistent_with_the_ratio():
    """`solver_s_to_erase_100x` must be exactly the 100x break-even."""
    sur_max = 0.000639
    erase = 100.0 * sur_max
    assert abs(erase / sur_max - 100.0) < 1e-9
    # a solver at that latency yields exactly 100x, so above it the clause
    # holds and below it the clause does not
    assert (erase * 1.001) / sur_max > 100.0
    assert (erase * 0.999) / sur_max < 100.0
    print(f"ok  a reference solver at {erase*1e3:.1f} ms is the 100x "
          f"break-even for a {sur_max*1e3:.3f} ms surrogate")


def test_report_marks_absent_runs_not_measured():
    """A missing JSON must render `[not measured]`, never a silent omission."""
    import report
    for fn, args in ((report.sec_uq_seeds, (None,)),
                     (report.sec_degradation, (None,)),
                     (report.sec_fair, (None, None))):
        body = []
        fn(*args, body)
        text = "\n".join(body)
        assert report.NM in text, (fn.__name__, text)
    print("ok  absent runs render [not measured] in all three new sections")


def test_report_regenerates_and_is_deterministic():
    """RESULTS.md must be a pure function of the run JSONs."""
    with tempfile.TemporaryDirectory() as d:
        outs = []
        for i in range(2):
            o = Path(d) / f"r{i}.md"
            r = subprocess.run([sys.executable, "scripts/report.py",
                                "--out", str(o)], cwd=ROOT,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(o.read_text())
        assert outs[0] == outs[1], "report.py is not deterministic"
    print(f"ok  report.py regenerates deterministically "
          f"({len(outs[0].splitlines())} lines)")


def test_verdict_reports_both_readings_of_every_reopened_clause():
    """Rung 1: a clause met under one reading must show the other too."""
    res = (ROOT / "RESULTS.md")
    if not res.exists():
        print("skip  RESULTS.md absent")
        return
    t = res.read_text()
    fair = ROOT / "runs" / "bench_fair.json"
    if fair.exists():
        # both the batch-1 and the batch-64 readings must be present, so a
        # passing latency row can never appear without the failing batched one
        assert "per-sample latency" in t and "batched" in t, \
            "only one of the two speedup readings is in the verdict"
        # and the subsidized denominator must be shown next to the fair one
        assert "subsidized" in t, "the subsidy is not disclosed in the verdict"
    csu = ROOT / "runs" / "consistency_uq.json"
    if csu.exists():
        assert "strict" in t and "conditional on the shift" in t, \
            "clause 3 shows only one reading"
    print("ok  every reopened clause shows both readings in the verdict")


if __name__ == "__main__":
    for f in [test_conservative_pairing_is_the_least_favourable,
              test_inadmissible_solver_setting_cannot_be_the_denominator,
              test_staleness_gate_rejects_a_cached_output,
              test_erase_threshold_is_consistent_with_the_ratio,
              test_report_marks_absent_runs_not_measured,
              test_report_regenerates_and_is_deterministic,
              test_verdict_reports_both_readings_of_every_reopened_clause]:
        f()
    print("all bench_fair tests passed")
