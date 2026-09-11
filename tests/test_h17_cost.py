"""The H17 cost aggregator is mostly a refusal, so the refusals are pinned.

A guard whose value is that it declines to emit a number is worthless if it
quietly stops declining. Each test below is one way the two arms can fail to be
the same measurement -- and each corresponds to a mistake this repository has
actually made and withdrawn:

* different device -- a ratio measured across two GPUs is not a ratio;
* different solver denominator -- the ratios are then against different
  baselines and their difference is not the wrapper's cost;
* different accuracy -- a speedup at a different error is not the same speedup,
  which is the brief's rule about quoting the error a speedup was achieved at;
* different protocol fields (task, head, trials, tolerance, seed).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("agg_h17_cost")


def _arm(gpu="NVIDIA H100 NVL", rel_l2=0.05, check_every=50, solver_s=0.137,
         surrogate_s=0.000639, **over):
    d = {"task": "darcy", "uq_source": "het", "trials": 5, "tol": 1e-10,
         "seed": 0, "surrogate_rel_l2": rel_l2, "env": {"gpu": gpu},
         "batches": {"1": {
             "fair_denominator": {"check_every": check_every,
                                  "fast_apply": True, "precond": "discrete",
                                  "median_s": solver_s},
             "surrogate": {"graph": {"median_s": surrogate_s}}}}}
    d.update(over)
    return d


def test_identical_arms_are_comparable_and_the_arithmetic_is_right():
    base = _arm()
    eq = _arm(surrogate_s=0.000639 * 1.10)          # 10% slower wrapper
    r = H.compare(base, eq)
    assert r["comparable"], r["mismatches"]
    cell = r["by_batch"]["1"]["by_path"]["graph"]
    assert abs(cell["overhead_pct"] - 10.0) < 1e-6, cell["overhead_pct"]
    # each ratio is against its OWN denominator
    assert abs(cell["clause2_ratio_base"] - 0.137 / 0.000639) < 1e-6
    assert abs(cell["clause2_ratio_eq"] - 0.137 / (0.000639 * 1.10)) < 1e-6
    assert r["by_batch"]["1"]["denominator_matches"]
    assert "ratio_delta_withheld" not in cell


def test_different_device_is_refused():
    r = H.compare(_arm(), _arm(gpu="NVIDIA A100"))
    assert not r["comparable"]
    assert any("env.gpu" in m for m in r["mismatches"]), r["mismatches"]
    assert r["by_batch"] == {}, "no numbers may be emitted after a refusal"


def test_different_accuracy_is_refused():
    """A 5% accuracy difference is not a 5% detail; it changes the problem."""
    r = H.compare(_arm(rel_l2=0.05), _arm(rel_l2=0.05 * 1.05))
    assert not r["comparable"]
    assert any("surrogate_rel_l2" in m for m in r["mismatches"]), r["mismatches"]
    # and a difference inside the stated tolerance is allowed through
    ok = H.compare(_arm(rel_l2=0.05), _arm(rel_l2=0.05 * 1.001))
    assert ok["comparable"], ok["mismatches"]


def test_different_denominator_withholds_the_delta_but_keeps_the_overhead():
    """Different solver settings: the overhead is still the wrapper's, the
    ratio difference is not."""
    r = H.compare(_arm(check_every=50), _arm(check_every=1, solver_s=0.30))
    assert r["comparable"], r["mismatches"]     # same measurement, different denom
    rec = r["by_batch"]["1"]
    assert not rec["denominator_matches"]
    assert "ratio_delta_withheld" in rec["by_path"]["graph"]


def test_protocol_fields_are_checked():
    for field, bad in (("task", "poisson"), ("uq_source", "cqr"),
                       ("trials", 3), ("seed", 1)):
        r = H.compare(_arm(), _arm(**{field: bad}))
        assert not r["comparable"], field
        assert any(field in m for m in r["mismatches"]), (field, r["mismatches"])


def test_it_never_claims_to_be_quotable_against_the_published_figure():
    """The withdrawn error was multiplying an overhead into a figure from a
    different run. The flag is false unconditionally, including on a clean
    comparison."""
    for a, b in ((_arm(), _arm()), (_arm(), _arm(gpu="other"))):
        assert H.compare(a, b)["quotable_against_published"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all h17-cost guard tests passed")
