"""Split conformal calibration for field-valued surrogates.

The design rule: **a calibrator only ever sees nonconformity scores.** Whatever
produced them -- an FNO on a PDE, a GP on a plant model, a lookup table -- is
somebody else's problem. That is what makes this a kit rather than one model's
eval script.

Three calibrators, in increasing order of what they assume:

| calibrator            | assumes                                    | gives |
|-----------------------|--------------------------------------------|-------|
| `SplitConformal`      | calibration and test exchangeable          | marginal coverage |
| `GroupConformal`      | exchangeable *within* each group           | per-group coverage |
| `WeightedConformal`   | covariate shift with a known density ratio | coverage under shift |

and four score functions, because for a field the question "is the error bar
big enough" has more than one honest reading -- see `SCORES`.
"""
from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# nonconformity scores
#
# `pred`, `sigma`, `truth` are all (n, C, H, W). A score function returns either
# one score per sample (n,) or one per pixel (n*C*H*W,) -- `per_sample` says
# which, because the coverage a user gets differs completely between the two and
# conflating them is how a 90% guarantee turns into a field that is wrong
# somewhere 100% of the time.
# --------------------------------------------------------------------------- #
def _floor(sigma, frac=0.05, med=None):
    """Floor the spread at `frac` of a *fixed* median spread.

    A deep ensemble whose members happen to agree at a pixel reports sigma ~ 0
    there, and |resid| / sigma then diverges, so the 90th percentile of the
    pixelwise ratio is set by numerical accidents rather than by error. The
    floor is a scale, not a tuned constant: a fixed fraction of the median
    spread on the *calibration* split.

    `med` must be passed by anything that calibrates on one set and evaluates on
    another. Defaulting it to `sigma.median()` -- which is what an earlier
    version of this file did -- makes the floor, and therefore the width of the
    interval, depend on the batch it is being tested against: a shifted set with
    a larger spread quietly widens its own band and repairs part of the
    undercoverage the measurement exists to expose. `tests/test_api.py` catches
    it as a disagreement between `interval` and `evaluate`.
    """
    return sigma + frac * (sigma.median() if med is None else med)


def score_field_max(pred, sigma, truth, frac=0.05, med=None):
    """max over pixels of |resid| / sigma. One score per sample.

    Coverage of the resulting band is *simultaneous*: the whole field lies
    inside pred +/- q*sigma. This is the strict reading and the kit's default.
    """
    r = (pred - truth).abs() / _floor(sigma, frac, med)
    return r.flatten(1).max(dim=1).values


def score_pixel(pred, sigma, truth, frac=0.05, med=None):
    """|resid| / sigma at every pixel. One score per pixel (marginal coverage)."""
    return ((pred - truth).abs() / _floor(sigma, frac, med)).flatten()


def score_norm_ratio(pred, sigma, truth, frac=0.05, med=None):
    """||resid||_2 / ||sigma||_2. One score per sample.

    The "is this sample's error bar the right size" reading: it certifies the
    field's aggregate error, not every pixel of it.
    """
    num = (pred - truth).flatten(1).norm(dim=1)
    den = sigma.flatten(1).norm(dim=1).clamp_min(1e-12)
    return num / den


def score_rel_l2(pred, sigma, truth, frac=0.05, med=None):
    """||resid||_2 / ||truth||_2, ignoring sigma. One score per sample.

    A model-free certificate: calibrating this gives "with 90% probability the
    relative error of this surrogate is below q". No uncertainty head required,
    which makes it the fallback for a surrogate that has none -- and the
    baseline that shows what the ensemble spread is actually buying.
    """
    num = (pred - truth).flatten(1).norm(dim=1)
    den = truth.flatten(1).norm(dim=1).clamp_min(1e-12)
    return num / den


SCORES = {
    "field_max": (score_field_max, True),
    "pixel": (score_pixel, False),
    "norm_ratio": (score_norm_ratio, True),
    "rel_l2": (score_rel_l2, True),
}


def get_score(name):
    """(fn, per_sample) for a registered score name."""
    if name not in SCORES:
        raise KeyError(f"unknown score {name!r}; have {sorted(SCORES)}")
    return SCORES[name]


# --------------------------------------------------------------------------- #
# calibrators
# --------------------------------------------------------------------------- #
def conformal_quantile(scores, alpha):
    """The finite-sample split-conformal quantile, ceil((n+1)(1-alpha))/n.

    Not `numpy.quantile(1-alpha)`. The +1 is the whole guarantee: with n=100 and
    alpha=0.1 the correct index is the 91st order statistic, and using the 90th
    undercovers by about a point -- inside the KPI's +/-2pp band, so it would
    never be caught by looking at the answer.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:                      # too few points to certify at this alpha
        return float("inf")
    return float(s[k - 1])


class SplitConformal:
    """Marginal split conformal. `fit` on calibration scores, `q` thereafter."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None
        self.n_cal = 0

    def fit(self, cal_scores):
        cal_scores = _np(cal_scores)
        self.q = conformal_quantile(cal_scores, self.alpha)
        self.n_cal = len(cal_scores)
        return self

    def covered(self, test_scores):
        return _np(test_scores) <= self.q

    def coverage(self, test_scores):
        return float(self.covered(test_scores).mean())


class GroupConformal:
    """One quantile per group -- the regression analogue of class-conditional.

    Marginal coverage on a pooled test set is satisfiable by over-covering the
    easy families and abandoning the hard one, which is exactly the failure a
    multi-PDE surrogate is prone to: four families at 0.3% error and Darcy at
    4.5%. A per-group quantile removes that degree of freedom.

    Groups with fewer than `min_n` calibration points fall back to the pooled
    quantile, and which groups did so is recorded in `fellback`.
    """

    def __init__(self, alpha=0.1, min_n=50):
        self.alpha, self.min_n = alpha, min_n
        self.q = {}
        self.q_pooled = None
        self.fellback = []

    def fit(self, cal_scores, cal_groups):
        s, g = _np(cal_scores), np.asarray(cal_groups)
        self.q_pooled = conformal_quantile(s, self.alpha)
        self.q, self.fellback = {}, []
        for u in np.unique(g):
            m = g == u
            if m.sum() < self.min_n:
                self.q[u] = self.q_pooled
                self.fellback.append(u)
            else:
                self.q[u] = conformal_quantile(s[m], self.alpha)
        return self

    def covered(self, test_scores, test_groups):
        s, g = _np(test_scores), np.asarray(test_groups)
        qs = np.array([self.q.get(u, self.q_pooled) for u in g], dtype=np.float64)
        return s <= qs

    def coverage(self, test_scores, test_groups, per_group=False):
        cov = self.covered(test_scores, test_groups)
        if not per_group:
            return float(cov.mean())
        g = np.asarray(test_groups)
        return {str(u): float(cov[g == u].mean()) for u in np.unique(g)}


class WeightedConformal:
    """Split conformal under covariate shift (Tibshirani et al., 2019).

    Exchangeability fails the moment the input distribution moves, and the
    coverage guarantee fails quietly with it. Given the density ratio
    w(x) = p_test(x) / p_cal(x), reweighting the empirical distribution of the
    calibration scores restores it.

    **The approximation, stated rather than hidden.** Exact weighted conformal
    recomputes the quantile per test point, because that point's own weight
    enters the denominator. This computes one quantile against
    `sum(w_cal) + median(w_test)`, following the same shortcut taken in
    `wafer-tool-shift/wts/metrics.py`. It is accurate when the calibration set
    is large relative to a single weight; with n_cal = 1,024 and weights of
    order 1 the induced error in the effective alpha is O(1/n_cal) ~ 0.1pp,
    an order of magnitude inside the KPI band. It is *not* accurate when a
    handful of calibration points carry most of the weight, which is why
    `fit` records the effective sample size and `report.py` prints it: an ESS
    that has collapsed means the guarantee is being carried by a few points and
    the interval should not be believed however good the coverage looks.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None
        self.ess = None
        self.n_cal = 0

    def fit(self, cal_scores, cal_w, test_w):
        s = _np(cal_scores)
        w = np.asarray(cal_w, dtype=np.float64)
        w_extra = float(np.median(np.asarray(test_w, dtype=np.float64)))
        order = np.argsort(s)
        s_sorted, w_sorted = s[order], w[order]
        cum = np.cumsum(w_sorted) / (w_sorted.sum() + w_extra)
        j = int(np.searchsorted(cum, 1 - self.alpha))
        self.q = float(s_sorted[min(j, len(s_sorted) - 1)])
        self.ess = float(w.sum() ** 2 / np.square(w).sum())
        self.n_cal = len(s)
        return self

    def covered(self, test_scores):
        return _np(test_scores) <= self.q

    def coverage(self, test_scores):
        return float(self.covered(test_scores).mean())


class LikelihoodRatioProbe:
    """w(x) = p_test(x)/p_cal(x) from a logistic discriminator on features.

    A classifier trained to tell calibration inputs from (unlabelled) test
    inputs gives p(test|x); the odds of that are the density ratio up to the
    class-prior constant, which cancels in the conformal quantile. The guarantee
    is only ever as good as this probe, so `fit` returns its own held-out AUC
    and every consumer is expected to print it: an AUC of 0.5 means no shift was
    detected and the weights are noise; an AUC of 1.0 means the supports barely
    overlap and the ratio is unbounded, which is when weighting stops helping.

    Pure numpy (batch gradient descent on ~1k x ~20 features), so the kit adds
    no scikit-learn dependency for twelve lines of logistic regression.
    """

    def __init__(self, l2=1e-2, steps=800, lr=0.5, clip=20.0, seed=0):
        self.l2, self.steps, self.lr, self.clip, self.seed = l2, steps, lr, clip, seed
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None
        self.auc = None

    def fit(self, feat_cal, feat_test):
        from .metrics import auroc
        X = np.concatenate([_np(feat_cal), _np(feat_test)]).astype(np.float64)
        y = np.concatenate([np.zeros(len(feat_cal)), np.ones(len(feat_test))])
        self.mu, self.sd = X.mean(0), X.std(0).clip(1e-8)
        Xs = (X - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(Xs))
        cut = int(0.7 * len(Xs))
        tr, te = idx[:cut], idx[cut:]
        w = np.zeros(Xs.shape[1])
        b = 0.0
        for _ in range(self.steps):
            z = Xs[tr] @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = p - y[tr]
            w -= self.lr * (Xs[tr].T @ g / len(tr) + self.l2 * w)
            b -= self.lr * g.mean()
        self.w, self.b = w, b
        self.auc = auroc(Xs[te] @ w + b, y[te].astype(bool))
        return self

    def weights(self, feat):
        """Density ratio p_test/p_cal, clipped to [1/clip, clip].

        Unclipped importance weights on a real shift routinely span six orders
        of magnitude and hand the entire quantile to one calibration point. The
        clip is a bias-variance trade made explicit: it costs exact validity and
        buys a usable ESS, and the ESS is reported so the cost is visible.
        """
        Xs = (_np(feat) - self.mu) / self.sd
        z = np.clip(Xs @ self.w + self.b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        r = p / np.clip(1.0 - p, 1e-9, None)
        return np.clip(r, 1.0 / self.clip, self.clip)


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)
