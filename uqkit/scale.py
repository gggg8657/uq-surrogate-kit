"""H15: a difficulty model for the interval width, and conformal on top of it.

H14 measured that clause 1 fails under covariate shift in the *scale* of the
conformity score rather than in the composition of the test set: under shift
sigma under-predicts the error at fixed error magnitude, so S = max|mu-u|/sigma
has a uniformly inflated upper quantile and no selection rule -- not even an
oracle on the true error -- can reach it. The alternative is to make the width
respond to difficulty:

    fit h(z) ~ the conditional (1-alpha) quantile of S,
    calibrate T = S / h(z) by split conformal,
    emit the half-width  q * h(z) * sigma(x).

If h captures how the shift inflates S then T's quantile is shift-stable and
coverage returns to the band without discarding anything -- and the intervals
on *easier* inputs get narrower, which is the direction no gate can move.

Two deliberate design choices, both to keep the result interpretable rather
than merely good:

* **h is linear in standardized log-features, fitted by pinball loss.** Convex,
  no architecture, no early stopping, and readable coefficients. A black box
  here would be indistinguishable from memorising the development shifts. An
  MLP variant is only worth measuring if the linear one already clears the
  band, so it can never be the thing that rescues the clause.
* **h is fitted on the target quantile of S directly**, not on the error. The
  quantile of a log is the log of the quantile, so fitting the 0.9-pinball loss
  on `log S` and exponentiating estimates the conditional 90th percentile of S
  exactly, with no distributional assumption.

`ScaleConformal` keeps the exact finite-sample split-conformal guarantee *in
distribution*, because T is just another score. Out of distribution it has no
guarantee -- neither does anything else in this file -- and the coverage it
achieves there is an empirical claim about h's extrapolation, which is why H15
reports a leave-one-mechanism-out reading as its headline.
"""
from __future__ import annotations

import numpy as np
import torch

from .conformal import GroupConformal, conformal_quantile


class Standardizer:
    """Zero-mean unit-variance per feature, fitted once and frozen.

    Fitted on the *fitting* rows only. Standardizing with statistics that
    include the evaluation shards would leak the shift into the feature scaling
    even if the coefficients never saw it.
    """

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, z):
        z = torch.as_tensor(z, dtype=torch.float64)
        self.mu = z.mean(0)
        self.sd = z.std(0, unbiased=False).clamp_min(1e-8)
        return self

    def __call__(self, z):
        z = torch.as_tensor(z, dtype=torch.float64)
        return (z - self.mu) / self.sd


def pinball(pred, target, q):
    """Quantile (pinball) loss at level `q`. Minimised by the qth quantile."""
    d = target - pred
    return torch.maximum(q * d, (q - 1.0) * d).mean()


class QuantileScale:
    """h(z) = exp(w . z + b), fitted to the conditional (1-alpha) quantile of S.

    Full-batch Adam on a convex objective with a fixed seed and a fixed step
    count: deterministic, and there is no validation set to leak because there
    is nothing to early-stop.

    `l2` is a ridge on the coefficients only, never on the intercept -- the
    intercept has to be free to place the overall level of h, and penalising it
    would bias every interval by a constant.
    """

    def __init__(self, alpha=0.1, l2=1e-3, steps=2000, lr=0.05, seed=0,
                 h_min=1e-3, h_max=1e4):
        self.alpha, self.l2, self.steps, self.lr, self.seed = (
            alpha, l2, steps, lr, seed)
        self.h_min, self.h_max = h_min, h_max
        self.std = None
        self.w = None
        self.b = None
        self.loss_curve = []
        self.n_fit = 0

    def fit(self, z, s):
        """`z` (n, d) features, `s` (n,) conformity scores > 0.

        Thread count is pinned for the duration of the fit. Torch defaults to
        one thread per core -- 96 on this host -- and on a problem this small
        the synchronisation dominates. `scripts/bench_scale_fit.py` measures
        the fit at each thread count and writes `runs/scale_fit_threads.json`;
        4 is chosen from that table.

        An earlier version of this docstring claimed "5.4 s at the 96-thread
        default" for 200 steps at 2e4 rows. That number was arithmetic on a
        50-step timing taken at 7,360 rows, not a measurement at the stated
        size, and it should not have been written down. The two readings that
        *were* measured at 21,760 rows -- 0.68 s at one thread, 0.20 s at four,
        200 steps -- are in the JSON with the rest.
        """
        torch.manual_seed(self.seed)
        _threads = torch.get_num_threads()
        torch.set_num_threads(min(4, _threads))
        self.std = Standardizer().fit(z)
        x = self.std(z)
        y = torch.log(torch.as_tensor(s, dtype=torch.float64).clamp_min(1e-12))
        w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.tensor(float(torch.quantile(y, 1.0 - self.alpha)),
                         dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=self.lr)
        for i in range(self.steps):
            opt.zero_grad()
            loss = pinball(x @ w + b, y, 1.0 - self.alpha) + self.l2 * (w @ w)
            loss.backward()
            opt.step()
            if i % max(self.steps // 20, 1) == 0:
                self.loss_curve.append(float(loss.detach()))
        self.w, self.b = w.detach(), b.detach()
        self.n_fit = int(len(y))
        torch.set_num_threads(_threads)
        return self

    def __call__(self, z):
        h = torch.exp(self.std(z) @ self.w + self.b)
        return h.clamp(self.h_min, self.h_max)

    def coef_table(self, names=None):
        """Coefficients, largest magnitude first -- the point of a linear h."""
        w = self.w.cpu().numpy()
        idx = np.argsort(-np.abs(w))
        return [{"feature": (names[i] if names else f"z{i}"),
                 "coef": float(w[i])} for i in idx]


class GroupScaleConformal:
    """Split conformal on T = S / h(z), with **one quantile per family**.

    This is the calibrator the in-distribution headline uses, applied to the
    rescaled score. It exists because the pooled-quantile version below is not
    comparable to the `group` baseline H15 is measured against, and the first
    H15 run proved that empirically rather than in principle: a single pooled q
    on T put in-distribution coverage at 0.588 for helmholtz and 1.000 for
    diffusion and advdiff, while `group` holds every family at 0.902. h absorbs
    difficulty *within* a family and leaves a per-family offset behind, so
    removing the per-family degree of freedom that `GroupConformal` was
    introduced to remove re-opens exactly the failure it was introduced for.

    Reporting a pooled reading against a per-family baseline is the mismatch
    this repo already caught itself making once, in the shifted-coverage table.
    Both readings are kept -- `pooled_q` is recorded beside the per-family
    ones -- but the per-family one is what the comparison uses.
    """

    def __init__(self, alpha=0.1, min_n=50):
        self.alpha = alpha
        self.inner = GroupConformal(alpha, min_n=min_n)
        self.n_cal = 0

    def fit(self, cal_scores, cal_h, cal_groups):
        t = (np.asarray(cal_scores, dtype=np.float64)
             / np.asarray(cal_h, dtype=np.float64))
        self.inner.fit(t, cal_groups)
        self.n_cal = int(len(t))
        return self

    @property
    def q(self):
        return {str(k): float(v) for k, v in self.inner.q.items()}

    @property
    def pooled_q(self):
        return float(self.inner.q_pooled)

    def _q_for(self, groups):
        return np.array([self.inner.q.get(u, self.inner.q_pooled)
                         for u in np.asarray(groups)], dtype=np.float64)

    def covered(self, scores, h, groups):
        t = (np.asarray(scores, dtype=np.float64)
             / np.asarray(h, dtype=np.float64))
        return t <= self._q_for(groups)

    def width_multiplier(self, h, groups):
        """`q_family * h` -- the per-sample multiplier on sigma."""
        return self._q_for(groups) * np.asarray(h, dtype=np.float64)


class ScaleConformal:
    """Split conformal on T = S / h(z), pooled quantile.

    Kept as the pooled reading beside `GroupScaleConformal`. Identical machinery to `SplitConformal` -- the same quantile index, the
    same exact in-distribution guarantee -- applied to a rescaled score. The
    interval is `mu +/- q * h(z) * sigma`, so `q * h(z)` is the width
    multiplier and `h` is where every claim about shift robustness lives.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None
        self.n_cal = 0

    def fit(self, cal_scores, cal_h):
        t = (np.asarray(cal_scores, dtype=np.float64)
             / np.asarray(cal_h, dtype=np.float64))
        self.q = float(conformal_quantile(t, self.alpha))
        self.n_cal = int(len(t))
        return self

    def covered(self, scores, h):
        return (np.asarray(scores, dtype=np.float64)
                <= self.q * np.asarray(h, dtype=np.float64))

    def width_multiplier(self, h):
        """`q * h` -- what the interval is multiplied by, per sample."""
        return self.q * np.asarray(h, dtype=np.float64)
