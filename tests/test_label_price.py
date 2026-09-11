"""The label-budget reading, pinned so the vacuous one cannot come back.

Two instances of this brief independently reported "49 of 49 shifted shards
inside 90+/-2% from 9 labels" and one of them nearly published it. It is an
artefact: split conformal's MEAN coverage over labelled draws is
`(k+1-l)/(k+1)` with `l = floor((k+1)*alpha)`, which is exactly 9/10 at k=9.
A deployment gets ONE draw, whose coverage is Beta(k+1-l, l).

These tests pin both laws and the sawtooth, so any future reading of
`in_band_of_mean` as a result fails here first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALPHA, BAND = 0.1, (0.88, 0.92)


def _l(k):
    return int(np.floor((k + 1) * ALPHA))


def test_mean_coverage_is_a_sawtooth_and_k9_is_exactly_0p9():
    """Why 49/49 at k=9 is integer arithmetic and not a measurement."""
    want = {9: 0.9000, 12: 12 / 13, 16: 16 / 17, 24: 23 / 25,
            32: 30 / 33, 64: 59 / 65, 128: 117 / 129}
    for k, m in want.items():
        l = _l(k)
        got = (k + 1 - l) / (k + 1)
        assert abs(got - m) < 1e-12, (k, got, m)
    assert abs((9 + 1 - _l(9)) / 10 - 0.90) < 1e-15, "k=9 must be exactly 0.9"
    # the sawtooth: adding labels can take the MEAN out of band
    inband = {k: BAND[0] <= (k + 1 - _l(k)) / (k + 1) <= BAND[1]
              for k in (9, 12, 16, 32)}
    assert inband == {9: True, 12: False, 16: False, 32: True}, inband
    print("ok  mean coverage is (k+1-l)/(k+1); exactly 0.9000 at k=9, and "
          "out of band at k=12 and k=16 -- non-monotone in k")


def test_single_draw_law_and_the_price():
    """Beta(k+1-l, l) at k=9 has sd 0.0905, so one draw is nowhere near +/-2%."""
    a, b = 9 + 1 - _l(9), _l(9)
    assert (a, b) == (9, 1)
    assert abs(beta.std(a, b) - 0.0905) < 5e-4, beta.std(a, b)
    p9 = beta.cdf(BAND[1], a, b) - beta.cdf(BAND[0], a, b)
    assert abs(p9 - 0.1557) < 5e-4, p9
    prices = {}
    for k in (256, 512, 1024):
        l = _l(k)
        prices[k] = beta.cdf(BAND[1], k + 1 - l, l) - \
            beta.cdf(BAND[0], k + 1 - l, l)
    assert abs(prices[512] - 0.8711) < 5e-4, prices
    assert abs(prices[1024] - 0.9681) < 5e-4, prices
    assert prices[256] < 0.75 < prices[512], prices
    print(f"ok  one draw at k=9 lands in band with p={p9:.4f}; reaching 0.90 "
          f"needs k~1024 (p={prices[1024]:.4f})")


def test_measured_fractions_track_the_analytic_law():
    """The result: the price is the calibrator's, not this surrogate's."""
    p = ROOT / "runs" / "h30_labels.json"
    if not p.is_file():
        print("skip runs/h30_labels.json absent")
        return
    d = json.loads(p.read_text())
    assert d["clause_eligible"] is False, \
        "label-budget numbers consume labels from the shard they certify"
    checked = 0
    for key, row in d["by_k"].items():
        pred = row["armA_beta_prediction"]
        if pred is None:
            assert row["armA_frac_draws_in_band_median"] == 0.0, \
                f"{key}: quantile is infinite, so no draw can be in band"
            continue
        got = row["armA_frac_draws_in_band_median"]
        assert abs(got - pred) < 0.05, (key, got, pred)
        checked += 1
    assert checked >= 6, f"only {checked} finite budgets compared"
    print(f"ok  measured arm-A fractions track Beta(k+1-l, l) within 0.05 at "
          f"{checked}/{checked} finite budgets")


def test_arm_b_is_biased_not_merely_noisy():
    """Arm B getting WORSE with more labels is the claim; pin it."""
    p = ROOT / "runs" / "h30_labels.json"
    if not p.is_file():
        print("skip runs/h30_labels.json absent")
        return
    d = json.loads(p.read_text())
    b = {r["k"]: r["armB_frac_draws_in_band_median"]
         for r in d["by_k"].values()}
    peak = max(b, key=lambda k: b[k])
    big = max(b)
    assert b[big] < b[peak], \
        f"arm B is not worse at k={big} ({b[big]}) than at its peak k={peak}"
    assert b[big] < 0.10, f"arm B at k={big} is {b[big]}, expected < 0.10"
    assert d["arm_c"]["in_band_median"] < d["n_ood_shards"] / 2, \
        "arm C in band on at least half the shards would mean the shift IS " \
        "shape-preserving, contradicting the recorded finding"
    print(f"ok  arm B peaks at k={peak} ({b[peak]:.3f}) and falls to "
          f"{b[big]:.3f} at k={big}; arm C in band on "
          f"{d['arm_c']['in_band_median']:.0f}/{d['n_ood_shards']}")


if __name__ == "__main__":
    test_mean_coverage_is_a_sawtooth_and_k9_is_exactly_0p9()
    test_single_draw_law_and_the_price()
    test_measured_fractions_track_the_analytic_law()
    test_arm_b_is_biased_not_merely_noisy()
    print("all label-price tests passed")
