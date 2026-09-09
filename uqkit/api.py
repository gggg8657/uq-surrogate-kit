"""The public surface: wrap any surrogate, get an interval and a trust score.

    sim  = PDE2DSimulator("darcy")            # or your own, see `Simulator`
    uq   = UQSurrogate(DeepEnsemble(members), score="field_max", alpha=0.1)
    uq.calibrate(mean_cal, sigma_cal, truth_cal, inputs=a_cal)
    lo, hi = uq.interval(mean_test, sigma_test)
    trust  = uq.ood_score(a_test, mean_test, sigma_test)

Everything the kit needs from a simulator is in the `Simulator` protocol below;
everything it needs from a model is "given an input, return a prediction". No
step of the pipeline knows it is looking at a PDE.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .conformal import (GroupConformal, LikelihoodRatioProbe, SplitConformal,
                        WeightedConformal, get_score)
from .features import Mahalanobis, spectral_features


@runtime_checkable
class Simulator(Protocol):
    """What the kit needs from a simulator to benchmark and shift-test a surrogate.

    `residual` is optional and is the one that pays for itself: a simulator that
    can *apply* its operator cheaply, even when solving it is expensive, gives
    the surrogate a ground-truth-free error signal at negligible cost.
    """

    name: str

    def sample_inputs(self, n: int, seed: int, N: int) -> torch.Tensor: ...
    def solve(self, a: torch.Tensor) -> torch.Tensor: ...
    def residual(self, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor | None: ...


class UQSurrogate:
    """A surrogate plus a calibrated interval plus a reason to distrust it.

    `mode` selects the calibrator:
      - `"split"`   marginal split conformal.
      - `"group"`   one quantile per group (pass `groups=` to calibrate/evaluate).
      - `"weighted"` covariate-shift-weighted, needs `inputs=` at calibration
        time and the shifted inputs at evaluation time.
    """

    def __init__(self, ensemble, score="field_max", alpha=0.1, mode="split",
                 feature_fn=spectral_features, sigma_floor_frac=0.05):
        self.ens = ensemble
        self.alpha = alpha
        self.mode = mode
        self.score_name = score
        self.score_fn, self.per_sample = get_score(score)
        self.feature_fn = feature_fn
        self.frac = sigma_floor_frac
        self.cal = None
        self.detector = None
        self.cal_features = None
        self.cal_scores = None
        self._sigma_med = None

    # -- calibration -------------------------------------------------------- #
    def scores(self, mean, sigma, truth):
        if self._sigma_med is None:
            raise RuntimeError("calibrate() first: the sigma floor is frozen there")
        return self.score_fn(mean, sigma, truth, frac=self.frac, med=self._sigma_med)

    def calibrate(self, mean, sigma, truth, groups=None, inputs=None):
        self._sigma_med = float(sigma.median())
        s = self.scores(mean, sigma, truth)
        self.cal_scores = s
        # `_sigma_med` above is the calibration split's own median spread,
        # frozen before the scores are computed so that evaluation on a shifted
        # set cannot re-scale the interval using the shifted set's statistics.
        if self.mode == "group":
            if groups is None:
                raise ValueError("mode='group' needs `groups`")
            self.cal = GroupConformal(self.alpha).fit(s, groups)
        elif self.mode == "weighted":
            if inputs is None:
                raise ValueError("mode='weighted' needs `inputs`")
            self.cal_features = self.feature_fn(inputs).cpu()
            self.cal = None                   # fitted per test set in `evaluate`
        else:
            self.cal = SplitConformal(self.alpha).fit(s)
        return self

    def fit_detector(self, train_inputs):
        """Fit the input-density OOD detector on *training* inputs only."""
        self.detector = Mahalanobis().fit(self.feature_fn(train_inputs).cpu())
        return self

    # -- use ---------------------------------------------------------------- #
    def interval(self, mean, sigma):
        """Prediction band. Only meaningful for the sigma-scaled scores."""
        if self.score_name not in ("field_max", "pixel"):
            raise ValueError(f"score {self.score_name!r} certifies a scalar, "
                             "not a pointwise band; use `certify`")
        q = self._q()
        half = q * (sigma + self.frac * self._sigma_med)
        return mean - half, mean + half

    def certify(self, mean, sigma):
        """The calibrated bound on this sample's error, in the score's units."""
        q = self._q()
        if self.score_name == "rel_l2":
            return torch.full((mean.shape[0],), q, device=mean.device)
        if self.score_name == "norm_ratio":
            return q * sigma.flatten(1).norm(dim=1)
        return q * (sigma.flatten(1).max(dim=1).values + self.frac * self._sigma_med)

    def evaluate(self, mean, sigma, truth, groups=None, inputs=None):
        """Coverage of the calibrated interval on a (possibly shifted) test set."""
        s = self.scores(mean, sigma, truth)
        if self.mode == "group":
            cov = self.cal.covered(s, groups)
        elif self.mode == "weighted":
            if inputs is None:
                raise ValueError("mode='weighted' needs `inputs`")
            f_te = self.feature_fn(inputs).cpu()
            probe = LikelihoodRatioProbe().fit(self.cal_features, f_te)
            w_cal = probe.weights(self.cal_features)
            w_te = probe.weights(f_te)
            wc = WeightedConformal(self.alpha).fit(self.cal_scores, w_cal, w_te)
            self.last_probe = {"probe_auc": probe.auc, "ess": wc.ess, "q": wc.q,
                               "w_cal_max": float(w_cal.max()),
                               "w_cal_median": float(np.median(w_cal))}
            cov = wc.covered(s)
        else:
            cov = self.cal.covered(s)
        k, n = int(cov.sum()), int(len(cov))
        from .metrics import binom_ci
        lo, hi = binom_ci(k, n)
        return {"coverage": k / n, "n": n, "ci95": [lo, hi],
                "score": self.score_name, "alpha": self.alpha, "mode": self.mode}

    def ood_score(self, inputs=None, mean=None, sigma=None, kind="spread"):
        from .ood import spread_score
        if kind == "spread":
            return spread_score(sigma, mean)
        if kind == "mahalanobis":
            if self.detector is None:
                raise RuntimeError("call fit_detector first")
            return self.detector.score(self.feature_fn(inputs).cpu())
        raise KeyError(kind)

    def _q(self):
        if self.cal is None:
            raise RuntimeError("calibrate() first (weighted mode: use evaluate())")
        return self.cal.q if hasattr(self.cal, "q") and not isinstance(self.cal.q, dict) \
            else self.cal.q_pooled
