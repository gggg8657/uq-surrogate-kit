"""Deep ensemble wrapper -- the uncertainty head the rest of the kit consumes.

Nothing here is specific to a neural operator: `members` is any list of
callables `a -> prediction`, so a five-member ensemble of GPs, of boosted trees,
or of the same simulator at five mesh resolutions all work. The kit only needs
a mean and a spread.
"""
from __future__ import annotations

import torch


class DeepEnsemble:
    """Mean and per-pixel std over M member predictions.

    The spread is the *sample* std (ddof=1). With M=5 members the population std
    understates the spread by 11%, which propagates straight into the conformal
    quantile -- it would still calibrate, but the reported "sigma" would no
    longer be an estimate of anything.
    """

    def __init__(self, members, name="ensemble"):
        if len(members) < 2:
            raise ValueError("a spread needs at least 2 members")
        self.members = list(members)
        self.name = name

    def __len__(self):
        return len(self.members)

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        preds = torch.stack([m(*args, **kwargs) for m in self.members])
        return preds.mean(0), preds.std(0, unbiased=True), preds
