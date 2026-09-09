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
    """Split conformal under covariate shift (Tibshirani et al., 2019), exactly.

    Exchangeability fails the moment the input distribution moves, and the
    coverage guarantee fails quietly with it. Given the density ratio
    w(x) = p_test(x) / p_cal(x), reweighting the empirical distribution of the
    calibration scores restores it.

    **The test point's own weight is in the denominator, and it is kept here.**
    The construction places mass w_j / (W + v_i) on each calibration score and
    w_i / (W + v_i) on +infinity, so the quantile is *per test point*. An
    earlier version of this class used one quantile computed against
    `W + median(v)`, following the shortcut in `wafer-tool-shift/wts/metrics.py`,
    and justified it by a large calibration set. That justification is wrong in
    the case that matters: calibration weights can be perfectly uniform while a
    few test points carry enormous weight, and those are exactly the points a
    shift produces. The exact version is one vectorized `searchsorted` over the
    test set, so there was never much to buy.

    When the calibration mass cannot reach 1 - alpha even with all of it, the
    quantile is **+infinity** -- the interval is uninformative and says so.
    Clamping to the largest calibration score instead, which the shortcut did,
    silently undercovers: with 5 uniform calibration points at alpha = 0.1 it
    returns a band whose true coverage is 5/6 = 83.3%.

    Validity still assumes p(y|x) is unchanged. Under an *operator* shift it is
    not, and no reweighting of x repairs that; `eval_conformal.py` marks those
    shards rather than reporting a number that looks like a fix.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None                 # median per-test-point quantile, for reporting
        self.q_per_test = None
        self.ess = None
        self.n_cal = 0
        self.n_inf = 0
        self.test_w_ratio = None

    def fit(self, cal_scores, cal_w, test_w):
        s = _np(cal_scores)
        w = np.asarray(cal_w, dtype=np.float64)
        v = np.asarray(test_w, dtype=np.float64)
        order = np.argsort(s)
        self._s_sorted = s[order]
        self._cum = np.cumsum(w[order])
        W = float(w.sum())
        thresh = (1 - self.alpha) * (W + v)
        j = np.searchsorted(self._cum, thresh, side="left")
        q = np.where(j < len(s), self._s_sorted[np.minimum(j, len(s) - 1)],
                     np.inf)
        self.q_per_test = q
        self.n_inf = int(np.isinf(q).sum())
        self.q = float(np.median(q[np.isfinite(q)])) if np.isfinite(q).any() \
            else float("inf")
        self.ess = float(W ** 2 / np.square(w).sum())
        self.n_cal = len(s)
        self.test_w_ratio = float(v.max() / max(np.median(v), 1e-12))
        return self

    def covered(self, test_scores):
        s = _np(test_scores)
        if len(s) != len(self.q_per_test):
            raise ValueError("weighted conformal is per test point: `covered` "
                             "must be called on the same set `fit` was given "
                             f"({len(self.q_per_test)} points, got {len(s)})")
        return s <= self.q_per_test

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
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        cut = int(0.7 * len(X))
        tr, te = idx[:cut], idx[cut:]
        # standardization is fitted on the training half only. Fitting it on
        # everything leaks the held-out half's marginals into the number that
        # is then quoted as the probe's held-out AUC -- small, but the AUC is
        # the diagnostic a reader uses to decide whether to trust the weights,
        # so it must not be the one statistic with a thumb on it.
        self.mu, self.sd = X[tr].mean(0), X[tr].std(0).clip(1e-8)
        Xs = (X - self.mu) / self.sd
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
