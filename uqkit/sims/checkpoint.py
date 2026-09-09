"""Checkpoint load/save helpers shared by the eval and finetune scripts."""
from __future__ import annotations

from pathlib import Path

import torch

from .fno2d import FNO2d


def load_model(path, device="cuda", strict=True):
    """Rebuild an FNO2d from a checkpoint written by `scripts/train_ddp.py`."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = FNO2d(width=a["width"], modes=a["modes"], n_layers=a["layers"],
                  n_tasks=a.get("n_tasks", 8))
    model.load_state_dict(ckpt["model"], strict=strict)
    return model.to(device).eval(), ckpt


def seed_task_embedding(model, new_task_id, from_task_ids):
    """Initialize an unseen PDE's embedding row from the pretrained mean.

    A brand-new random row would put the finetune off-manifold on step 0; the
    centroid of the pretrained tasks starts it inside the learned conditioning
    space.
    """
    with torch.no_grad():
        src = model.task_emb.weight[list(from_task_ids)].mean(dim=0)
        model.task_emb.weight[new_task_id] = src
    return model
