"""`Simulator` adapter for the 2D PDE corpus, including the cheap residuals.

The residual is the interesting part. For every family whose operator is a
Fourier multiplier or a sparse stencil, *applying* the operator to the
surrogate's own output costs one FFT or one 5-point stencil, while *solving* it
costs either the same FFT (the exact families) or thousands of preconditioned-CG
iterations (Darcy). Where that asymmetry is large, the residual is a
ground-truth-free error signal that is free relative to the speedup being
claimed; where it is not, it buys nothing and is reported as such.

`residual` returns `None` for the time-evolution families (diffusion,
advection-diffusion, Navier-Stokes). Checking those means propagating a state,
which is the solve, so there is no cheap algebraic check to have -- stating that
is more useful than inventing an expensive one and calling it free.
"""
from __future__ import annotations

import torch

from . import pde2d as P


class PDE2DSimulator:
    """One PDE family, wrapped to the `uqkit.Simulator` protocol."""

    def __init__(self, task, device="cuda", N=64):
        if task not in P.CFG:
            raise KeyError(task)
        self.name = task
        self.task = task
        self.base = P.FAMILY_BASE[task]
        self.cfg = P.CFG[task]
        self.device = device
        self.N = N

    # -- protocol ----------------------------------------------------------- #
    def sample_inputs(self, n, seed=0, N=None):
        a, _, _ = P.generate(self.task, n, N=N or self.N, seed=seed,
                             device=self.device)
        return a.to(self.device)

    @torch.no_grad()
    def solve(self, a):
        """Reference solve from the packed input tensor `a` (n, 2, N, N)."""
        g = a[:, 0]
        if self.base == "poisson":
            return P.solve_poisson(g).unsqueeze(1)
        if self.base == "biharmonic":
            return P.solve_biharmonic(g).unsqueeze(1)
        if self.base == "fractional":
            return P.solve_fractional(g, self.cfg["s"]).unsqueeze(1)
        if self.base == "helmholtz":
            return P.solve_helmholtz(g, self.cfg["kappa2"]).unsqueeze(1)
        if self.base == "diffusion":
            return P.solve_diffusion(g, self.cfg["nu"], self.cfg["T"]).unsqueeze(1)
        if self.base == "advdiff":
            return P.solve_advdiff(g, self.cfg["nu"], self.cfg["cx"],
                                   self.cfg["cy"], self.cfg["T"]).unsqueeze(1)
        if self.base == "darcy":
            u, _ = P.solve_darcy(torch.exp(g), a[:, 1])
            return u.unsqueeze(1)
        if self.base == "navier_stokes":
            dt = self.cfg["dt"] * min(1.0, 64.0 / a.shape[-1])
            w, _ = P.solve_navier_stokes(g, self.cfg["nu"], self.cfg["T"], dt)
            return w.unsqueeze(1)
        raise KeyError(self.base)

    def solver_accuracy(self):
        """What the reference solve is converged to, verbatim in the report."""
        if self.base in ("poisson", "helmholtz", "biharmonic", "fractional",
                         "diffusion", "advdiff"):
            return "exact spectral propagator (fp32 round-off, ~1e-7 rel)"
        if self.base == "darcy":
            return "batched PCG, fp64, tol 1e-10 on the relative residual"
        return (f"pseudo-spectral RK4, dt={self.cfg['dt']}, 2/3 dealiasing "
                f"(T={self.cfg['T']}, {int(self.cfg['T'] / self.cfg['dt'])} steps)")

    @torch.no_grad()
    def residual(self, a, u):
        """||L u - f|| as a field, or None where there is no cheap apply.

        **This check has a noise floor, and the floor grows with the order of
        the operator.** Applying L amplifies whatever round-off is already in
        `u` by the operator's symbol, so the residual of the *exact* fp32
        solution is not zero. Measured at 64^2 by
        `scripts/measure_residual_floor.py`: 6.2e-5 for Poisson (|k|^2),
        2.5e-5 for Helmholtz, 7.8e-5 for Darcy, 5.3e-2 for the biharmonic
        (|k|^4) and 68 -- far larger than the right-hand side -- for `frac_s3`
        (|k|^6). Against surrogate errors of 1e-3 to 1e-2 the first three have
        one to two orders of headroom and the check is informative; above
        fourth order it returns nothing but round-off and must not be used.

        The apply itself is done in float64 so that it contributes nothing of
        its own to that floor; the dominant term is the precision of `u`, which
        is the surrogate's to fix, not this function's. The upcast costs one
        extra FFT pair against a solve that costs thousands of iterations.
        """
        u = (u[:, 0] if u.dim() == 4 else u).double()
        a = a.double()
        g = a[:, 0]
        N = u.shape[-1]
        k2 = P.laplacian_symbol(N, u.device, u.dtype)
        if self.base == "poisson":
            r = _mult(u, k2) - g
        elif self.base == "fractional":
            r = _mult(u, k2 ** self.cfg["s"]) - g
        elif self.base == "biharmonic":
            r = _mult(u, k2 ** 2) - g
        elif self.base == "helmholtz":
            r = _mult(u, k2 - self.cfg["kappa2"]) - g
        elif self.base == "darcy":
            r = P._darcy_apply(torch.exp(g), u) - a[:, 1]
        else:
            return None
        return r.unsqueeze(1).float()

    def rhs(self, a):
        """The right-hand side the residual is normalized by."""
        return (a[:, 1:2] if self.base == "darcy" else a[:, 0:1]).float()


def _mult(u, symbol):
    """Apply a Fourier multiplier, zeroing the constant mode (periodic solvability)."""
    s = symbol.clone()
    out = torch.fft.ifft2(torch.fft.fft2(u) * s).real
    return out
