"""H17: restore the scale-equivariance the surrogate throws away.

H16 specified clause 1 under shift exactly: a width model would have to span a
**117x** dynamic range while holding **+-2%** accuracy. The 117x is not spread
across the shift suite -- it is concentrated in the four `*_amp2` shards, which
need 52-117x while everything else needs <= 19x.

That concentration is a bug in the surrogate, not a property of the physics.
Poisson, Helmholtz, diffusion and advection-diffusion are **linear** in the
field the `amp` shift scales, so the exact solution obeys

    u(c * f) = c * u(f).

The network breaks it for one reason: `predict_shard*` standardizes inputs with
**frozen calibration statistics**, so a 2x input lands 2x outside the range the
weights were fitted on and the network extrapolates instead of scaling.

The repair is a test-time wrapper, not a retrain. Divide the linear channels by
a per-sample scale, run the unchanged pipeline, multiply the mean **and sigma**
back:

    F_eq(a) = s(a) * F(a / s(a)),      s(a) = rms(a_linear) / ref

This is exactly equivariant for any F whatsoever, which is worth stating
because it makes the property testable without reference to the network:
s(c*a) = c*s(a), so (c*a)/s(c*a) = a/s(a), so F_eq(c*a) = c*F_eq(a) to floating
point. `tests/test_equivar.py` pins that identity.

**Which channels are linear is family-specific and getting it wrong would
manufacture the result.** For Poisson and Helmholtz channel 0 is the source; for
diffusion and advection-diffusion it is the initial condition; both are linear.
For **Darcy channel 0 is the log-permeability** -- `PDE2DSimulator.rhs` reads
the source out of channel 1 -- so scaling it raises permeability to a power and
is not a rescaling of anything. Darcy's linear channel is 1, which the `amp`
shift does not touch, so `darcy_amp2` must **not** improve. That makes it the
control: if it improves anyway, this wrapper is doing something other than what
is claimed here.

`ref` is the median per-sample scale on the **calibration** split, so s ~ 1 in
distribution and the wrapper is near-identity there. If in-distribution
coverage moves, the wrapper is broken.
"""
from __future__ import annotations

import torch

#: channels the solution is *linear* in, per family. Anything not listed here
#: enters the operator nonlinearly and must be left alone.
LINEAR_CHANNELS = {
    "poisson": (0,),        # source
    "helmholtz": (0,),      # source
    "diffusion": (0,),      # initial condition
    "advdiff": (0,),        # initial condition
    "darcy": (1,),          # source; channel 0 is log-permeability
}


def sample_scale(a, parent, ref=1.0, eps=1e-12):
    """Per-sample scale of the linear channels, relative to `ref`.

    **The channels are gathered by slicing, not by advanced indexing, and that
    is load bearing.** `a[:, list(ch)]` builds its index tensor on the host and
    copies it to the device on every call, which is illegal inside a CUDA graph
    capture: it raised `operation not permitted when stream is capturing` and
    took `bench_fair.py --equivariant` down with it, which is why the wrapper's
    effect on the clause-2 reading was `[not measured]`. A slice is a view and
    captures like any other op. Isolated directly -- `a[:, [0]]` fails capture
    and `a[:, 0:1]` succeeds -- and pinned by
    `tests/test_equivar.py::test_scale_is_cuda_graph_capturable`.

    `torch.cat` of single-channel slices is used rather than one slice so that
    a family with non-contiguous linear channels stays correct; with today's
    one channel per family it is a no-op copy of the same data.
    """
    ch = LINEAR_CHANNELS.get(parent)
    if ch is None:
        return torch.ones(len(a), device=a.device, dtype=a.dtype)
    sub = torch.cat([a[:, c:c + 1] for c in ch], dim=1).flatten(1)
    rms = sub.pow(2).mean(dim=1).sqrt()
    return (rms / ref).clamp_min(eps)


def reference_scale(a, parent):
    """The median per-sample scale, to be computed on calibration data only."""
    return float(sample_scale(a, parent, ref=1.0).median())


def rescale_inputs(a, parent, s):
    """`a` with its linear channels divided by `s`; other channels untouched."""
    ch = LINEAR_CHANNELS.get(parent)
    if ch is None:
        return a
    out = a.clone()
    view = s.view(-1, *([1] * (a.dim() - 1)))
    for c in ch:
        out[:, c:c + 1] = out[:, c:c + 1] / view
    return out


def predict_equivariant(predict_fn, blob, parent, ref, device=None):
    """`F_eq(a) = s * F(a / s)` around any `(mean, sigma, truth, a)` predictor.

    `truth` and the returned inputs are the originals -- only the prediction
    path is rescaled, so every score, calibrator and detector downstream sees
    exactly the data it saw before. `sigma` is multiplied by `s` along with the
    mean because it is a spread in the field's units; leaving it unscaled would
    hand the amplitude shift a narrower interval than the mean it belongs to.

    **The rescale happens on `device`, not on the host.** The first version of
    this function computed the scale and cloned the input on whatever device
    `blob["a"]` happened to live on -- the CPU, since shards are loaded with
    `map_location="cpu"` -- and then moved the raw input to the GPU a second
    time to return it. `scripts/bench_equivar_cost.py` measured that at **+206%
    of the forward at batch 1 and +857% at batch 64**, against an argument in
    the H17 write-up that it was "one extra reduction per sample, so the 100x
    row is untouched". The arithmetic was one reduction; the implementation was
    a host-side clone of the whole input plus a duplicate host-to-device copy.
    Doing it on-device is what makes the argument true, and it had to be
    measured to find that out.
    """
    a_raw = blob["a"]
    if device is not None and a_raw.device.type != torch.device(device).type:
        a_raw = a_raw.to(device, non_blocking=True)
    s_dev = sample_scale(a_raw, parent, ref)
    scaled = dict(blob)
    scaled["a"] = rescale_inputs(a_raw, parent, s_dev)
    mean, sigma, truth, _a = predict_fn(scaled)
    s = s_dev.to(mean.device).view(-1, *([1] * (mean.dim() - 1)))
    return mean * s, sigma * s, truth, a_raw.to(mean.device)
