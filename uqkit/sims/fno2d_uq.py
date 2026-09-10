"""One network that emits a mean *and* an uncertainty, in one forward pass.

Why this file exists. Every interval in this kit used to come from the
disagreement between M ensemble members, so an interval cost M forward passes.
That is what made the coverage clause and the >=100x speedup clause mutually
exclusive: `runs/members.json` measures M=1 at 108.5x and M=2 at 58.5x, and M=1
has no spread. The fix is not a faster ensemble, it is to stop building the
interval out of an ensemble.

`FNO2dUQ` is `FNO2d` plus a second output projection on the *same* trunk:

    channel 0        mean                      (trained by rel_l2, as before)
    channel 1        log sigma                 (Gaussian NLL, heteroscedastic)
    channels 2, 3    residual quantiles lo/hi  (pinball at alpha/2, 1-alpha/2)

Two things about it are deliberate and load-bearing.

**The uncertainty head sees a detached trunk.** `x.detach()` in `forward` means
the NLL and pinball gradients never reach the shared blocks, so the mean is
trained by exactly the loss `scripts/train_member.py` trains it by. Without
that, an arm with an uncertainty head is also an arm with a different mean, and
a coverage-versus-speedup comparison against the ensemble stops being a
comparison of uncertainty mechanisms. The cost is that the head cannot ask the
trunk for features it wants; that is a real ceiling on how good sigma can get
and it is the honest trade for a clean comparison. `--no-detach-uq` runs the
other way if the ceiling turns out to bind.

**Two sigma sources, not one.** The heteroscedastic head is a parametric
Gaussian claim; the quantile pair makes no distributional assumption and is what
conformalized quantile regression calibrates. They disagree about
*asymmetric* error, and disagreeing is informative, so both are measured rather
than one being picked in advance.

Neither is calibrated by itself. Both are fed to the same split-conformal
machinery in `uqkit/conformal.py` -- the same scores, the same (n+1) quantile,
the same frozen sigma floor -- so the only thing that differs from the
ensemble's numbers is where sigma came from.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fno2d import FNO2d

# log sigma is clamped, not softplussed: a clamp keeps the gradient exactly 1
# inside the range instead of saturating, and the range is wide enough (e^-10 to
# e^5 in normalized field units) that it only ever catches a divergence.
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -10.0, 5.0


class FNO2dUQ(FNO2d):
    """FNO2d with a 3-channel uncertainty head on the shared trunk."""

    def __init__(self, *args, detach_uq=True, **kwargs):
        kwargs["out_ch"] = 1
        super().__init__(*args, **kwargs)
        self.detach_uq = detach_uq
        self.proj_uq = nn.Sequential(
            nn.Conv2d(self.width, 2 * self.width, 1), nn.GELU(),
            nn.Conv2d(2 * self.width, 3, 1))
        # Start sigma near the scale the mean's own error actually has rather
        # than at exp(0)=1, which is ~100x too wide in normalized units and
        # costs the NLL several epochs of shrinking before it learns anything
        # about *which* samples are hard.
        nn.init.zeros_(self.proj_uq[-1].weight)
        with torch.no_grad():
            self.proj_uq[-1].bias.copy_(torch.tensor([-4.0, -0.02, 0.02]))

    def forward(self, a, task):
        """(B, 4, H, W): mean, log sigma, residual q_lo, residual q_hi."""
        x = self.trunk(a, task)
        mean = self.proj(x)
        z = self.proj_uq(x.detach() if (self.detach_uq and self.training) else x)
        return torch.cat([mean, z], dim=1)


def split_heads(y):
    """(B, 4, H, W) -> mean, log_sigma, d_lo, d_hi, each (B, 1, H, W)."""
    return (y[:, 0:1], y[:, 1:2].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX),
            y[:, 2:3], y[:, 3:4])


def sigma_from(y, source):
    """The sigma the conformal layer will calibrate, in the model's own units.

    `het`   exp(log sigma).
    `cqr`   half the quantile-pair width, which is what CQR calibrates. Clamped
            at zero because nothing enforces q_lo <= q_hi during training and a
            crossed pair at one pixel would otherwise hand the score a negative
            denominator.
    `const` a single scalar, the mean of `het` over the batch, broadcast --
            deliberately useless per-sample. It exists because split conformal
            rescales whatever sigma it is given, so *any* sigma reaches ~90%
            marginal coverage; `const` is the control that shows how much of a
            coverage number is the head and how much is the rescaling.
    """
    _, log_s, d_lo, d_hi = split_heads(y)
    if source == "het":
        return log_s.exp()
    if source == "cqr":
        return ((d_hi - d_lo) / 2).clamp_min(0.0)
    if source == "const":
        s = log_s.exp()
        return torch.full_like(s, float(s.mean()))
    raise KeyError(f"unknown sigma source {source!r}")


SIGMA_SOURCES = ("het", "cqr", "const")


def nll_loss(log_sigma, resid):
    """Gaussian NLL, 0.5 r^2/sigma^2 + log sigma, on a detached residual."""
    s2 = (2.0 * log_sigma).exp()
    return (0.5 * resid.pow(2) / s2 + log_sigma).mean()


def pinball_loss(d, resid, q):
    """Quantile (check) loss for predicting the q-quantile of `resid` by `d`."""
    e = resid - d
    return torch.maximum(q * e, (q - 1.0) * e).mean()
