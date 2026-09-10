# Vendored from pde-neural-operator/pdeno/fno2d.py (same author, 2026-09).
# Modified twice since: this header, and `trunk()` split out of `forward()`
# (2026-09-10) so that a second output head can be attached to the shared
# representation without duplicating the body. `forward` is unchanged in
# behaviour -- it is now `proj(trunk(...))` -- which `tests/test_sims.py`
# pins against the vendored checkpoints.
"""Multi-PDE 2D Fourier Neural Operator.

Same mechanism as the 1D model in `fno.py` -- FFT, keep the low modes, multiply
by learned complex weights, inverse FFT -- extended to 2D and conditioned on a
*task embedding* so one set of weights serves several PDE families. That
conditioning is what makes the pretrain -> finetune ("PDE foundation model")
recipe possible: pretraining fills the shared spectral layers, and a new PDE
arrives as a new embedding row plus a short finetune.

FFTs run in fp32 even under bf16 autocast (`torch.fft` has no bf16 kernels), so
the spectral block explicitly disables autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SpectralConv2d(nn.Module):
    """R^{C_in x H x W} -> R^{C_out x H x W} via a truncated Fourier multiply.

    Two weight blocks are needed because `rfft2` keeps the full range in the
    first axis: the low-positive and low-negative corners of the spectrum.

    The weights are *stored* as real tensors with a trailing size-2 axis and
    viewed as complex in `forward`. Mathematically identical (Adam already
    treats a complex parameter as two reals), but it keeps every gradient real,
    which is what lets DDP compress the all-reduce to bf16 -- the spectral
    weights dominate this model, so that halves the dominant communication
    cost.
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_lo = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w_hi = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        # Inference-only cache of the permuted weights; see `_packed`.
        self.cache_packed_weights = True
        self._packed_cache = None

    # -- inference weight cache ------------------------------------------
    # `einsum("bixy,ioxy->boxy", ...)` lowers to a batched matmul over the
    # (x, y) mode grid, so it must permute the (in, out, m1, m2) weight into
    # (x, y, in, out) order -- 13.1 MB per weight at width 64 / modes 20 --
    # on EVERY forward pass, although the weights are fixed at inference.
    # `runs/profile_surrogate.json` measures the consequence: at batch 1,
    # `aten::copy_` is 258.7 us/call (20.5% of the forward) while the eight
    # matmuls and eight FFTs together are 65.6 us. This cache does the permute
    # once and hands `bmm` the layout it wants.
    #
    # It is used only when the module is in eval mode AND grad is disabled, so
    # training, gradients and the fp64 discretization-invariance path are
    # untouched. The contraction is the same contraction in the same order:
    # `tests/test_sims.py::test_spectral_cache_is_bit_identical` pins the
    # output to `max|delta| == 0` against the einsum path, and the cache is
    # dropped whenever the weights could have changed (`train()`, a
    # `load_state_dict`).
    #
    # Cost: it duplicates the spectral weights on device (+26.2 MB for the
    # shipped model, roughly 2x its own footprint). Set
    # `cache_packed_weights = False` to opt out.

    def _packed(self, m1, m2, cdtype, device):
        key = (m1, m2, cdtype, device)
        cached = self._packed_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        w_lo = torch.view_as_complex(self.w_lo.detach().contiguous())
        w_hi = torch.view_as_complex(self.w_hi.detach().contiguous())
        pk = [w[:, :, :m1, :m2].permute(2, 3, 0, 1).contiguous()
              .reshape(m1 * m2, self.in_ch, self.out_ch).to(cdtype)
              for w in (w_lo, w_hi)]
        self._packed_cache = (key, pk[0], pk[1])
        return pk[0], pk[1]

    def clear_packed_cache(self):
        self._packed_cache = None

    def train(self, mode: bool = True):
        self.clear_packed_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *a, **kw):
        self.clear_packed_cache()
        return super()._load_from_state_dict(*a, **kw)

    def _band(self, x_band, w_packed, m1, m2, B):
        """sum_i x[b,i,x,y] * w[i,o,x,y] as one bmm over the (x, y) grid."""
        lhs = x_band.permute(2, 3, 0, 1).reshape(m1 * m2, B, self.in_ch)
        out = torch.bmm(lhs, w_packed)                    # (m1*m2, B, out)
        return out.reshape(m1, m2, B, self.out_ch).permute(2, 3, 0, 1)

    def forward(self, x):  # (B, C, H, W)
        with torch.autocast(device_type=x.device.type, enabled=False):
            # torch.fft has no half kernels; fp64 inputs (the invariance test)
            # keep their precision.
            if x.dtype in (torch.bfloat16, torch.float16):
                x = x.float()
            B, _, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            m1 = min(self.modes1, H // 2)
            m2 = min(self.modes2, W // 2 + 1)
            w_lo = torch.view_as_complex(self.w_lo.contiguous())
            w_hi = torch.view_as_complex(self.w_hi.contiguous())
            out_dtype = torch.promote_types(x_ft.dtype, w_lo.dtype)
            out_ft = torch.zeros(
                B, self.out_ch, H, W // 2 + 1, dtype=out_dtype, device=x.device
            )
            fast = (self.cache_packed_weights and not self.training
                    and not torch.is_grad_enabled())
            if fast:
                p_lo, p_hi = self._packed(m1, m2, out_dtype, x.device)
                out_ft[:, :, :m1, :m2] = self._band(
                    x_ft[:, :, :m1, :m2], p_lo, m1, m2, B)
                out_ft[:, :, -m1:, :m2] = self._band(
                    x_ft[:, :, -m1:, :m2], p_hi, m1, m2, B)
            else:
                out_ft[:, :, :m1, :m2] = torch.einsum(
                    "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w_lo[:, :, :m1, :m2]
                )
                out_ft[:, :, -m1:, :m2] = torch.einsum(
                    "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w_hi[:, :, :m1, :m2]
                )
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").to(x.dtype)


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2, norm=True):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width) if norm else nn.Identity()

    def forward(self, x):
        return x + F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class FNO2d(nn.Module):
    """Task-conditioned 2D FNO.

    forward(a, task) with a: (B, in_ch, H, W) and task: (B,) long task ids.
    Coordinate channels are built at the current resolution, so the same weights
    evaluate on grids the model never trained on (discretization invariance).
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        width: int = 96,
        modes: int = 24,
        n_layers: int = 6,
        n_tasks: int = 8,
        task_dim: int = 16,
        norm: bool = True,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.width, self.modes, self.n_layers = width, modes, n_layers
        # Recompute each block's interior in the backward pass instead of
        # storing it. At 402.81M parameters the activations, not the weights,
        # are what set the largest batch that fits (~67 MB/sample measured), so
        # this is the only lever that moves that ceiling -- FSDP shards
        # parameters and optimizer state, neither of which is the binding term.
        # `use_reentrant=False` is required for DDP with `static_graph=True`.
        self.grad_checkpoint = grad_checkpoint
        self.task_emb = nn.Embedding(n_tasks, task_dim)
        nn.init.normal_(self.task_emb.weight, std=0.02)
        self.lift = nn.Sequential(
            nn.Conv2d(in_ch + 2 + task_dim, width, 1), nn.GELU(), nn.Conv2d(width, width, 1)
        )
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes, modes, norm=norm) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, out_ch, 1)
        )

    def coords(self, B, H, W, device, dtype):
        y = torch.linspace(0, 1, H, device=device, dtype=dtype)
        x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([X, Y]).expand(B, 2, H, W)

    def trunk(self, a, task):
        """Everything before the output projection: (B, width, H, W)."""
        B, _, H, W = a.shape
        if task.dim() == 0:
            task = task.expand(B)
        emb = self.task_emb(task)[:, :, None, None].expand(-1, -1, H, W).to(a.dtype)
        x = torch.cat([a, self.coords(B, H, W, a.device, a.dtype), emb], dim=1)
        x = self.lift(x)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def forward(self, a, task):
        return self.proj(self.trunk(a, task))

    def param_count(self):
        """Number of trainable real-valued parameters."""
        return sum(p.numel() for p in self.parameters())


def rel_l2(pred, target, reduce="mean"):
    """Per-sample relative L2 error, the standard operator-learning metric."""
    num = (pred - target).flatten(1).norm(dim=1)
    den = target.flatten(1).norm(dim=1).clamp_min(1e-8)
    err = num / den
    return err.mean() if reduce == "mean" else err
