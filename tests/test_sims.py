"""The vendored physics, and the residual hook the OOD detector leans on.

The residual detector is only meaningful if the residual of the *exact*
solution is round-off. If the operator were applied with the wrong sign, the
wrong scaling or the wrong stencil, the residual would be large for everybody
and the detector would still produce a plausible-looking AUROC -- so this is
checked as an identity on every family that exposes one, not assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit.sims import pde2d as P  # noqa: E402
from uqkit.sims.fno2d import FNO2d  # noqa: E402
from uqkit.sims.pde2d_sim import PDE2DSimulator  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# Measured noise floor of the residual check at 64^2 on the exact fp32 solution
# (`scripts/measure_residual_floor.py`). These are ceilings on what the check
# can resolve, not targets -- see `PDE2DSimulator.residual`.
RESIDUAL_FLOOR_64 = {"poisson": 1e-4, "helmholtz": 1e-4, "frac_s0p5": 1e-5,
                     "darcy": 2e-4}
UNUSABLE_IN_FP32 = {"biharmonic": 1e-2, "frac_s3": 1.0}


def _floor(task, N):
    sim = PDE2DSimulator(task, device=DEV, N=N)
    a = sim.sample_inputs(8, seed=0, N=N)
    r = sim.residual(a, sim.solve(a))
    assert r is not None, task
    return float((r.flatten(1).norm(dim=1)
                  / sim.rhs(a).flatten(1).norm(dim=1)).max())


def test_residual_floor_where_the_detector_is_used():
    """On the families the detector runs on, the exact solution's residual is small.

    If the operator were applied with the wrong sign, scaling or stencil, the
    residual would be large for everybody and the detector would still produce
    a plausible-looking AUROC -- so this is an identity check, not a smoke test.
    """
    got = {t: _floor(t, 64) for t in RESIDUAL_FLOOR_64}
    for t, lim in RESIDUAL_FLOOR_64.items():
        assert got[t] < lim, (t, got[t], lim)
    print("ok  residual floor at 64^2:", {k: f"{v:.1e}" for k, v in got.items()})


def test_residual_is_useless_above_fourth_order():
    """The limitation is pinned by a test so it cannot be quietly forgotten.

    A future reader adding `biharmonic` or `frac_s3` to the detector would
    otherwise get a number rather than an error.
    """
    got = {t: _floor(t, 64) for t in UNUSABLE_IN_FP32}
    for t, lim in UNUSABLE_IN_FP32.items():
        assert got[t] > lim, (t, got[t], lim)
    print("ok  fp32 residual is round-off-dominated for",
          {k: f"{v:.1e}" for k, v in got.items()})


def test_time_families_have_no_cheap_residual():
    """Reported as `None`, not faked -- there is no algebraic check to have."""
    for task in ("diffusion", "advdiff", "navier_stokes"):
        sim = PDE2DSimulator(task, device=DEV, N=32)
        a = sim.sample_inputs(4, seed=0, N=32)
        assert sim.residual(a, sim.solve(a)) is None, task
    print("ok  time-evolution families report residual = None")


def test_shift_configs_change_the_input_not_the_operator():
    """A `*_rough` task must share its parent's solver and differ in its inputs."""
    base = PDE2DSimulator("poisson", device=DEV, N=32)
    rough = PDE2DSimulator("poisson_rough", device=DEV, N=32)
    assert rough.base == base.base
    assert P.PARENT["poisson_rough"] == "poisson"
    a0 = base.sample_inputs(64, seed=0, N=32)
    a1 = rough.sample_inputs(64, seed=0, N=32)
    # rougher field: more energy above the half-Nyquist ring
    def high_frac(a):
        A = torch.fft.rfft2(a[:, 0], norm="ortho").abs() ** 2
        ky = torch.fft.fftfreq(32, d=1 / 32, device=a.device).abs()
        kx = torch.fft.rfftfreq(32, d=1 / 32, device=a.device)
        kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        return float((A * (kr > 8)).flatten(1).sum(-1).mean()
                     / A.flatten(1).sum(-1).mean())
    assert high_frac(a1) > 3 * high_frac(a0), (high_frac(a0), high_frac(a1))
    print(f"ok  poisson_rough moves high-k energy fraction "
          f"{high_frac(a0):.4f} -> {high_frac(a1):.4f}")


def test_operator_is_resolution_invariant():
    """The FNO evaluates on a grid it never saw -- the property the kit relies on."""
    torch.manual_seed(0)
    m = FNO2d(width=16, modes=8, n_layers=2, n_tasks=4).to(DEV).eval()
    t = torch.zeros(2, dtype=torch.long, device=DEV)
    for N in (32, 64):
        a = torch.randn(2, 2, N, N, device=DEV)
        assert m(a, t).shape == (2, 1, N, N)
    print("ok  FNO2d runs at 32^2 and 64^2 with one set of weights")


if __name__ == "__main__":
    test_residual_floor_where_the_detector_is_used()
    test_residual_is_useless_above_fourth_order()
    test_time_families_have_no_cheap_residual()
    test_shift_configs_change_the_input_not_the_operator()
    test_operator_is_resolution_invariant()
    print("all sim tests passed")
