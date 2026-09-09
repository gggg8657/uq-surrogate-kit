"""Out-of-distribution scoring, and the two questions it can mean.

They are different questions with different answers, and reporting one number
for both is how a detector gets credit it has not earned:

1. **Shift detection** -- did the input distribution move? Scored as AUROC of
   in-distribution test against *each* shifted set separately, because an
   average over an easy shift and a hard one is not a measurement of either.
2. **Error detection** -- will the surrogate be wrong on *this* sample,
   whatever distribution it came from? Scored as AUROC against the binary label
   `true rel-L2 > tau`, on a pooled stream. This is the question a plant
   actually asks, and a detector can be excellent at (1) and useless at (2).

Every score here is unsupervised at deployment: none of them touches the ground
truth, and none is fitted on an OOD set.
"""
from __future__ import annotations

import numpy as np
import torch

from .metrics import auroc


# --------------------------------------------------------------------------- #
# scores  (higher = more suspicious)
# --------------------------------------------------------------------------- #
def spread_score(sigma, pred=None):
    """Ensemble disagreement, ||sigma||_2, optionally relative to ||pred||_2.

    Relative is the default because the families differ in output scale by
    orders of magnitude and an absolute spread would rank Darcy suspicious for
    being large rather than for being uncertain.
    """
    num = sigma.flatten(1).norm(dim=1)
    if pred is None:
        return num
    return num / pred.flatten(1).norm(dim=1).clamp_min(1e-12)


def residual_score(resid, rhs):
    """||L u_hat - f|| / ||f|| -- the PDE residual of the surrogate's own output.

    Available whenever the governing operator can be *applied* cheaply even
    though *solving* it is expensive, which is the common case: one Darcy
    residual is a single sparse apply against the 2,000 preconditioned-CG
    iterations the solve costs. That asymmetry is what makes this affordable
    inside the speedup claim rather than a way of giving it back.
    """
    return resid.flatten(1).norm(dim=1) / rhs.flatten(1).norm(dim=1).clamp_min(1e-12)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def shift_auroc(in_scores, ood_scores):
    """AUROC separating one OOD set from the in-distribution test set."""
    s = np.concatenate([np.asarray(in_scores), np.asarray(ood_scores)])
    y = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))]).astype(bool)
    return auroc(s, y)


def error_auroc(scores, errors, tau):
    """AUROC for 'this sample's relative error exceeds tau'."""
    return auroc(np.asarray(scores), np.asarray(errors) > tau)


def auroc_ci(scores, labels, n_boot=1000, seed=0):
    """Percentile bootstrap CI for an AUROC.

    An AUROC on 1,024 vs 1,024 samples has a standard error around 0.01, so a
    0.90 threshold cannot be adjudicated by a point estimate alone.
    """
    s, y = np.asarray(scores), np.asarray(labels).astype(bool)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        if y[i].sum() in (0, len(i)):
            continue
        out.append(auroc(s[i], y[i]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
