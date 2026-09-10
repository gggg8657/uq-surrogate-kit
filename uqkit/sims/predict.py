"""Load an ensemble of FNO members and run it over a cached shard.

The one subtlety worth stating: a shifted shard is evaluated under its
**parent's** task embedding and its **parent's** normalization statistics. That
is deployment. When the incoming field gets rougher, or the process changes
under you, nobody re-fits the standardizer and nobody hands the model a new task
id -- the configured surrogate keeps running on data it was not built for, and
whether the uncertainty notices is the entire question this kit exists to
answer. Re-normalizing per shard would quietly repair half the shift before the
detector ever saw it.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import load_model
from .dataset import denorm_u
from .fno2d_uq import sigma_from
from .pde2d import PARENT, TASK_ID


def load_members(paths, device="cuda"):
    models, ck = [], None
    for p in paths:
        m, ck = load_model(p, device)
        models.append(m)
    return models, ck


def shard_path(root, task, split, N=64):
    return Path(root) / f"{task}_{split}_N{N}.pt"


def load_shard(root, task, split, N=64):
    return torch.load(shard_path(root, task, split, N), map_location="cpu",
                      weights_only=False)


@torch.no_grad()
def predict_shard(models, blob, stats_by_task, device="cuda", batch=64,
                  autocast=True):
    """(mean, sigma, truth, inputs) in physical units, all on `device`.

    `sigma` is the unbiased std across members, de-normalized with the same
    statistics as the mean so it is in the units of the field.
    """
    task = blob["task"]
    parent = PARENT.get(task, task)
    st = stats_by_task[parent]
    tid = TASK_ID[parent]
    a_all, u_all = blob["a"], blob["u"]
    means, sigmas, truths, ins = [], [], [], []
    for i in range(0, len(a_all), batch):
        a_raw = a_all[i : i + batch].to(device)
        a = (a_raw - st["a_mean"].to(device)) / st["a_std"].to(device)
        t = torch.full((a.shape[0],), tid, device=device, dtype=torch.long)
        preds = []
        for m in models:
            if autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = m(a, t)
            else:
                p = m(a, t)
            preds.append(denorm_u(p.float(), st))
        preds = torch.stack(preds)
        means.append(preds.mean(0))
        sigmas.append(preds.std(0, unbiased=True))
        truths.append(u_all[i : i + batch].to(device))
        ins.append(a_raw)
    return (torch.cat(means), torch.cat(sigmas), torch.cat(truths),
            torch.cat(ins))


@torch.no_grad()
def predict_shard_single(model, blob, stats_by_task, device="cuda", batch=64,
                         autocast=True, sigma_source="het"):
    """Same contract as `predict_shard`, from ONE `FNO2dUQ` forward pass.

    Returns (mean, sigma, truth, inputs) in physical units so that every
    calibrator, score and detector downstream is byte-identical between this
    and the ensemble path -- the substitution being tested is *where sigma comes
    from* and nothing else.

    `sigma` is scaled by `u_std` alone and not shifted by `u_mean`: it is a
    spread, not a field value. Passing it through `denorm_u` would add the
    family's mean offset to every error bar, which for Darcy (offset ~0.7 in a
    field of the same order) would inflate every interval by a constant and
    manufacture coverage. `tests/test_api.py` pins the scaling.
    """
    task = blob["task"]
    parent = PARENT.get(task, task)
    st = stats_by_task[parent]
    tid = TASK_ID[parent]
    a_all, u_all = blob["a"], blob["u"]
    means, sigmas, truths, ins = [], [], [], []
    for i in range(0, len(a_all), batch):
        a_raw = a_all[i : i + batch].to(device)
        a = (a_raw - st["a_mean"].to(device)) / st["a_std"].to(device)
        t = torch.full((a.shape[0],), tid, device=device, dtype=torch.long)
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = model(a, t)
        else:
            y = model(a, t)
        y = y.float()
        means.append(denorm_u(y[:, 0:1], st))
        sigmas.append(sigma_from(y, sigma_source) * st["u_std"].to(device))
        truths.append(u_all[i : i + batch].to(device))
        ins.append(a_raw)
    return (torch.cat(means), torch.cat(sigmas), torch.cat(truths),
            torch.cat(ins))
