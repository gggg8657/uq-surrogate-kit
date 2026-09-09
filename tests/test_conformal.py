"""The calibration layer, checked against what it is supposed to guarantee.

These are not smoke tests. Split conformal has an exact finite-sample coverage
property, so it can be checked as an identity rather than eyeballed, and the
off-by-one in the quantile index -- the mistake that costs about a point of
coverage and hides inside a +/-2pp KPI band -- is checked directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.conformal import (GroupConformal, LikelihoodRatioProbe,  # noqa: E402
                             SplitConformal, WeightedConformal,
                             conformal_quantile, get_score)
from uqkit.metrics import auroc, binom_ci  # noqa: E402


def test_quantile_index():
    """ceil((n+1)(1-alpha)) picks the right order statistic, not the numpy one."""
    s = np.arange(1, 101, dtype=float)          # 1..100
    # ceil(101*0.9) = 91 -> the 91st order statistic = 91.0
    assert conformal_quantile(s, 0.1) == 91.0, conformal_quantile(s, 0.1)
    # too few points to certify: n=5, ceil(6*0.99)=6 > 5
    assert conformal_quantile(np.arange(5.0), 0.01) == float("inf")
    print("ok  quantile index")


def test_marginal_coverage():
    """Exchangeable data: coverage must sit at 1-alpha within sampling error."""
    rng = np.random.default_rng(0)
    covs = []
    for trial in range(200):
        x = rng.standard_normal(1000)
        cal, test = x[:500], x[500:]
        c = SplitConformal(0.1).fit(np.abs(cal))
        covs.append(c.coverage(np.abs(test)))
    m = float(np.mean(covs))
    assert 0.888 < m < 0.912, m
    print(f"ok  marginal coverage over 200 trials: {m:.4f}")


def test_group_conformal_protects_the_hard_group():
    """A pooled quantile abandons a heavy-tailed group; a per-group one does not."""
    rng = np.random.default_rng(1)
    easy = np.abs(rng.standard_normal(2000)) * 0.1
    hard = np.abs(rng.standard_normal(200)) * 5.0
    s = np.concatenate([easy, hard])
    g = np.array(["easy"] * len(easy) + ["hard"] * len(hard))
    cut = np.zeros(len(s), bool)
    cut[::2] = True                                    # interleaved cal/test
    pooled = SplitConformal(0.1).fit(s[cut])
    grouped = GroupConformal(0.1).fit(s[cut], g[cut])
    hard_te = (~cut) & (g == "hard")
    cov_pooled = float(pooled.covered(s[hard_te]).mean())
    cov_group = float(grouped.covered(s[hard_te], g[hard_te]).mean())
    assert cov_pooled < 0.7, cov_pooled
    assert 0.84 < cov_group < 0.96, cov_group
    print(f"ok  hard-group coverage: pooled {cov_pooled:.3f} -> group {cov_group:.3f}")


def test_weighted_conformal_recovers_coverage_under_shift():
    """Under a known covariate shift, weighting restores what split conformal loses.

    The shift is engineered so the score depends on x: calibration x ~ N(0,1),
    test x ~ N(1,1), score = |noise| * (1 + x^2). Unweighted split conformal
    must undercover; the weighted version, given the exact density ratio, must
    come back to the band.
    """
    rng = np.random.default_rng(2)
    n = 20000
    x_cal = rng.standard_normal(n)
    x_te = rng.standard_normal(n) + 1.0
    f = lambda x, e: np.abs(e) * (1 + x**2)
    s_cal = f(x_cal, rng.standard_normal(n))
    s_te = f(x_te, rng.standard_normal(n))
    plain = SplitConformal(0.1).fit(s_cal)
    cov_plain = plain.coverage(s_te)
    # exact ratio N(1,1)/N(0,1) = exp(x - 1/2)
    w_cal = np.exp(x_cal - 0.5)
    w_te = np.exp(x_te - 0.5)
    wc = WeightedConformal(0.1).fit(s_cal, w_cal, w_te)
    cov_w = wc.coverage(s_te)
    assert cov_plain < 0.87, cov_plain
    assert abs(cov_w - 0.9) < 0.02, cov_w
    print(f"ok  under shift: split {cov_plain:.3f} -> weighted {cov_w:.3f} "
          f"(ess {wc.ess:.0f}/{wc.n_cal})")


def test_probe_auc_is_honest():
    """No shift -> the probe reports ~0.5 and its weights are near 1."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2000, 6))
    b = rng.standard_normal((2000, 6))
    p = LikelihoodRatioProbe().fit(a, b)
    assert 0.42 < p.auc < 0.58, p.auc
    w = p.weights(a)
    assert 0.5 < float(np.median(w)) < 2.0, float(np.median(w))
    c = rng.standard_normal((2000, 6)) + 2.0
    p2 = LikelihoodRatioProbe().fit(a, c)
    assert p2.auc > 0.9, p2.auc
    print(f"ok  probe AUC: no shift {p.auc:.3f}, strong shift {p2.auc:.3f}")


def test_scores_shapes_and_scaling():
    torch.manual_seed(0)
    pred = torch.randn(16, 1, 8, 8)
    truth = pred + 0.1 * torch.randn_like(pred)
    sigma = 0.1 * torch.ones_like(pred)
    for name, expect_n in (("field_max", 16), ("norm_ratio", 16),
                           ("rel_l2", 16), ("pixel", 16 * 64)):
        fn, per_sample = get_score(name)
        s = fn(pred, sigma, truth)
        assert s.shape[0] == expect_n, (name, s.shape)
        assert per_sample == (name != "pixel")
    # field_max >= norm_ratio-style aggregate: the simultaneous score is stricter
    fm = get_score("field_max")[0](pred, sigma, truth)
    px = get_score("pixel")[0](pred, sigma, truth).reshape(16, -1)
    assert torch.allclose(fm, px.max(1).values)
    print("ok  score shapes and the field_max = max(pixel) identity")


def test_binom_ci():
    lo, hi = binom_ci(900, 1000)
    assert lo < 0.9 < hi and (hi - lo) < 0.05, (lo, hi)
    print(f"ok  Wilson interval for 900/1000: [{lo:.4f}, {hi:.4f}]")


def test_auroc_matches_brute_force():
    rng = np.random.default_rng(4)
    s = rng.integers(0, 5, 200).astype(float)      # heavy ties on purpose
    y = rng.random(200) < 0.4
    pos, neg = s[y], s[~y]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(auroc(s, y) - brute) < 1e-9, (auroc(s, y), brute)
    print(f"ok  AUROC with ties matches brute force: {auroc(s, y):.6f}")


if __name__ == "__main__":
    for f in [test_quantile_index, test_marginal_coverage,
              test_group_conformal_protects_the_hard_group,
              test_weighted_conformal_recovers_coverage_under_shift,
              test_probe_auc_is_honest, test_scores_shapes_and_scaling,
              test_binom_ci, test_auroc_matches_brute_force]:
        f()
    print("all conformal tests passed")
