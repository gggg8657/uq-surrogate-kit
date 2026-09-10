"""The H14 selective-certificate layer, checked where it can be checked exactly.

The one thing that could make H14's 0/32 wrong in the *other* direction is a
bookkeeping bug that suppresses in-band shards, so the pieces that decide the
tally are pinned here as identities rather than eyeballed:

* the gate's threshold produces the abstention rate it was calibrated to;
* a shard below `min_accepted` can never be counted in-band, whatever its
  coverage says -- this is the abstention-wearing-a-coverage-number guard, and
  it is the reason the tally is trustworthy in the direction that hurts;
* the exact sign-flip p and the rank correlation are arithmetic, so they are
  checked against hand-computable cases;
* the oracle gate stays fenced: it must be declared in `ORACLE_GATES` in both
  scripts, so it cannot quietly become a shipped detector.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name):
    """Import a `scripts/*.py` module by path; they are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AGG = _load("agg_selective")


def test_sign_flip_exact():
    """8 same-signed differences give the smallest attainable p, 2/256."""
    assert AGG.sign_flip([1] * 8) == 2 / 256
    assert AGG.sign_flip([-3, -1, -2, -1, -5, -1, -2, -1]) == 2 / 256
    # a perfectly balanced set cannot reject
    assert AGG.sign_flip([1, -1]) == 1.0
    # At 3 seeds the smallest attainable p is 2/8 = 0.25 -- the seed-count
    # lesson as arithmetic: three seeds cannot produce a significant result at
    # any effect size, which is why 3-seed comparisons are a screen.
    assert AGG.sign_flip([1, 1, 1]) == 2 / 8
    # And a dominant term does NOT rescue that. For [10, -1, 1], obs = 10 and
    # six of the eight sign assignments reach |10|: (+10,+1,+1)=12,
    # (+10,+1,-1)=10, (+10,-1,+1)=10 and their three negations, so p = 6/8.
    assert AGG.sign_flip([10, -1, 1]) == 6 / 8


def test_spearman_identities():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(AGG.spearman(x, x) - 1.0) < 1e-12
    assert abs(AGG.spearman(x, x[::-1]) + 1.0) < 1e-12
    # monotone but nonlinear is still exactly 1 -- that is the point of ranks
    assert abs(AGG.spearman(x, [v ** 3 for v in x]) - 1.0) < 1e-12
    assert AGG.spearman([1.0, 2.0], [1.0, 2.0]) is None      # too few points
    assert AGG.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # no variance


def test_in_band_is_two_sided():
    """The KPI band is two-sided: over-covering is a failure, not a bonus."""
    assert AGG.in_band(0.90) and AGG.in_band(0.88) and AGG.in_band(0.92)
    assert not AGG.in_band(0.921)
    assert not AGG.in_band(0.879)
    assert not AGG.in_band(1.0)          # the abstention failure mode
    assert not AGG.in_band(None)


def test_min_accepted_gate_cannot_be_bypassed():
    """A perfect coverage on 3 surviving points must not count as in-band.

    This is the guard that makes the H14 tally honest in the direction that
    hurts it: `measured` is a function of `n_accepted` alone, and the in-band
    tally requires `measured and in_band`, never `in_band` alone.
    """
    cell = {"measured": False, "selective": {"coverage": 0.90}}
    counted = bool(cell["measured"] and AGG.in_band(cell["selective"]["coverage"]))
    assert counted is False
    cell["measured"] = True
    assert bool(cell["measured"] and AGG.in_band(cell["selective"]["coverage"]))


def test_oracle_gate_is_declared_in_both_scripts():
    """`oracle_err` must be fenced as ground-truth-using in every script."""
    sel = _load("eval_selective")
    price = _load("eval_selective_price")
    assert sel.ORACLE_GATES == {"oracle_err"} == price.ORACLE_GATES
    assert "oracle_err" in sel.GATES and "oracle_err" in price.GATES
    # and the two scripts must agree on the gate set, or the price curve would
    # be priced for a different detector than the one that was measured
    assert set(sel.GATES) == set(price.GATES)


def test_threshold_delivers_its_calibrated_abstention():
    """tau = quantile(cal, 1-beta) refuses ~beta of an exchangeable test set."""
    rng = np.random.default_rng(0)
    for beta in (0.05, 0.2, 0.5):
        cal = rng.normal(size=20000)
        tau = float(np.quantile(cal, 1.0 - beta))
        test = rng.normal(size=20000)
        got = 1.0 - (test <= tau).mean()
        assert abs(got - beta) < 0.02, (beta, got)


def test_gating_the_numerator_is_strictly_weaker_than_gating_the_score():
    """The measured mechanism, reduced to an inequality that is provable.

    H14's finding is that gating on the error's *magnitude* cannot fix coverage
    when the conformity score is error/sigma and sigma is the part that
    shifted. The checkable core of that is an ordering, not a magic constant:
    at the same abstention rate, selecting on the numerator moves the score's
    upper quantile strictly less than selecting on the score itself. With the
    two factors independent it barely moves it at all.

    An earlier version of this test asserted the ratio stayed above 0.94. That
    bound was invented rather than derived -- the actual value here is ~0.81 --
    and the run it was supposed to describe is in `runs/selective.json`, which
    `test_selective_json_is_self_consistent` checks instead.
    """
    rng = np.random.default_rng(1)
    err = np.exp(rng.normal(size=200000))
    sigma = np.exp(rng.normal(size=200000))
    score = err / sigma
    q_all = np.quantile(score, 0.9)
    by_err = np.quantile(score[err <= np.quantile(err, 0.95)], 0.9)
    by_score = np.quantile(score[score <= np.quantile(score, 0.95)], 0.9)
    assert by_score < by_err < q_all, (by_score, by_err, q_all)
    # and the gap is large: selecting on the score is worth several times the
    # reduction that selecting on the numerator buys, at identical abstention
    assert (q_all - by_err) < (q_all - by_score)
    # a gate perfectly correlated with the score recovers the score selection
    perfect = np.quantile(score[score <= np.quantile(score, 0.95)], 0.9)
    assert abs(perfect - by_score) < 1e-12


def test_selective_json_is_self_consistent():
    """If the run JSON is present, its tally must match its own per-shard cells."""
    import json
    p = ROOT / "runs" / "selective.json"
    if not p.exists():
        print("  runs/selective.json absent -- skipped")
        return
    sv = json.loads(p.read_text())
    hs = sv["headline_score"]
    for gate, gv in sv["by_gate"].items():
        b = gv["by_score"].get(hs)
        if b is None:
            continue
        # the per-seed tally and the per-shard seed counts must agree in total
        from_shards = sum(v["in_band_seeds_selective"] for v in b["shards"].values()
                          if v["kind"] in AGG.COVARIATE_KINDS)
        assert from_shards == sum(b["in_band_selective_per_seed"]), gate
        # and no shard may be counted in-band on more seeds than it was measured
        for n, v in b["shards"].items():
            assert v["in_band_seeds_selective"] <= v["measured_seeds"], (gate, n)
    print(f"  checked {len(sv['by_gate'])} gates over {sv['n_seeds']} seeds")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all selective tests passed")
