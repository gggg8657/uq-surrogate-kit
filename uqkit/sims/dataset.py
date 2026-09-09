"""Dataset / normalization plumbing for the multi-PDE corpus.

Each cached shard is a dict of tensors written by `scripts/gen_data.py`. Samples
carry their task id so a single DDP loader can mix PDE families in one batch,
and per-task standardization is stored in the checkpoint so evaluation at other
resolutions reuses the exact training statistics.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .pde2d import TASK_ID


def shard_path(root, task, split, N):
    return Path(root) / f"{task}_{split}_N{N}.pt"


class PDEShard(Dataset):
    """One PDE family, one split. Normalization is applied on the fly."""

    def __init__(self, path, stats=None, task=None):
        blob = torch.load(path, map_location="cpu")
        self.a, self.u = blob["a"], blob["u"]
        self.task = task or blob["task"]
        self.tid = TASK_ID[self.task]
        self.meta = {k: v for k, v in blob.items() if k not in ("a", "u")}
        self.stats = stats or compute_stats(self.a, self.u)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, i):
        a = (self.a[i] - self.stats["a_mean"]) / self.stats["a_std"]
        u = (self.u[i] - self.stats["u_mean"]) / self.stats["u_std"]
        return a, u, self.tid


def compute_stats(a, u):
    """Per-channel mean/std over the whole shard, kept as (C, 1, 1) tensors."""
    return {
        "a_mean": a.mean(dim=(0, 2, 3), keepdim=True)[0],
        "a_std": a.std(dim=(0, 2, 3), keepdim=True)[0].clamp_min(1e-8),
        "u_mean": u.mean(dim=(0, 2, 3), keepdim=True)[0],
        "u_std": u.std(dim=(0, 2, 3), keepdim=True)[0].clamp_min(1e-8),
    }


def denorm_u(u, stats):
    return u * stats["u_std"].to(u.device) + stats["u_mean"].to(u.device)


def load_corpus(root, tasks, split, N, stats=None, limit=None):
    """Returns (list[PDEShard], {task: stats}) -- stats reused across splits."""
    shards, out_stats = [], {}
    for t in tasks:
        s = PDEShard(shard_path(root, t, split, N), stats=(stats or {}).get(t), task=t)
        if limit is not None and limit < len(s):
            s.a, s.u = s.a[:limit], s.u[:limit]
        shards.append(s)
        out_stats[t] = s.stats
    return shards, out_stats


class GPUCorpus:
    """The whole (normalized) training corpus resident in HBM.

    At 64^2 the five pretraining families are ~2 GB, which fits an H100 many
    times over, so there is no reason to pay for a host-side dataloader. Keeping
    it on-device is also what makes the multi-GPU scaling numbers meaningful:
    they measure compute and gradient all-reduce, not `DataLoader` starvation.

    Sharding across ranks mirrors `DistributedSampler`: one shared permutation
    per epoch, strided by rank, tail dropped so every rank steps the same number
    of times (required -- DDP all-reduces on every step).
    """

    def __init__(self, shards, device):
        a, u, tid = [], [], []
        for s in shards:
            st = s.stats
            a.append(((s.a - st["a_mean"]) / st["a_std"]).to(device))
            u.append(((s.u - st["u_mean"]) / st["u_std"]).to(device))
            tid.append(torch.full((len(s),), s.tid, dtype=torch.long, device=device))
        self.a, self.u, self.tid = torch.cat(a), torch.cat(u), torch.cat(tid)
        self.device = device
        self.tasks = [s.task for s in shards]
        self.stats = {s.task: s.stats for s in shards}

    def __len__(self):
        return self.a.shape[0]

    def epoch_batches(self, batch, epoch, rank=0, world=1, seed=0):
        g = torch.Generator().manual_seed(seed * 10_000 + epoch)
        perm = torch.randperm(len(self), generator=g)
        per_rank = len(self) // (world * batch) * batch     # drop the tail
        idx = perm[rank * per_rank : (rank + 1) * per_rank].to(self.device)
        for i in range(0, per_rank, batch):
            sel = idx[i : i + batch]
            yield self.a[sel], self.u[sel], self.tid[sel]

    def steps_per_epoch(self, batch, world=1):
        return len(self) // (world * batch)
