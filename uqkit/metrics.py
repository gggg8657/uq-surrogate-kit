"""Scoring primitives shared by the conformal, OOD and benchmark layers.

Everything here takes plain tensors, so the kit never needs to know what
simulator produced them.
"""
from __future__ import annotations

import numpy as np
import torch


def rel_l2(pred, target, reduce="none"):
    """Per-sample relative L2 error -- the standard operator-learning metric."""
    num = (pred - target).flatten(1).norm(dim=1)
    den = target.flatten(1).norm(dim=1).clamp_min(1e-12)
    err = num / den
    return err.mean() if reduce == "mean" else err


def auroc(scores, labels):
    """Rank AUROC with ties averaged. `labels` is boolean; True = positive.

    Ported from `wafer-tool-shift/wts/metrics.py`; kept here so the kit has no
    cross-repo import.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    su = s[order]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    y = torch.as_tensor(y, dtype=torch.float64).flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def binom_ci(k, n, z=1.959963985):
    """Wilson interval for an observed coverage k/n.

    Reported next to every coverage number because the KPI band is +/-2pp and a
    1,024-sample test split has a +/-1.8pp sampling interval at 90% all by
    itself -- without this, "89.4%" and "90.6%" look like different results.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))
