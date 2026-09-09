"""A second, deliberately un-PDE-like simulator, to keep the kit honest.

A kit that only ever ran on one corpus is one model's eval script with a
`setup.py`. This is a lumped-parameter batch reactor -- three coupled ODEs,
stiff enough to need a small step, integrated by RK4 -- and the "field" is the
concentration trajectory. It shares nothing with the FNO path: different state
shape, different solver, no spectral structure, CPU only. `tests/test_api.py`
runs the entire conformal + OOD + bench pipeline on it, so CI proves the kit is
simulator-agnostic without needing a GPU or a trained operator.
"""
from __future__ import annotations

import torch


class BatchReactorSim:
    """A -> B -> C with temperature-dependent Arrhenius rates.

    Input: (n, 2, 1, 1) holding (log k1 prefactor offset, temperature offset).
    Output: (n, 1, 1, n_steps_out) -- the sampled trajectory of species B, whose
    peak position is the quantity a real operator would care about.
    """

    name = "batch_reactor"

    def __init__(self, n_out=64, T_end=4.0, dt=2e-4, device="cpu"):
        self.n_out, self.T_end, self.dt, self.device = n_out, T_end, dt, device

    def sample_inputs(self, n, seed=0, N=None, shift=0.0, scale=1.0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(n, 2, 1, 1, generator=g) * scale + shift
        return z.to(self.device)

    def _rates(self, a):
        k1 = torch.exp(0.4 * a[:, 0, 0, 0]) * 1.5
        k2 = torch.exp(0.25 * a[:, 1, 0, 0]) * 0.8
        return k1, k2

    @torch.no_grad()
    def solve(self, a):
        k1, k2 = self._rates(a)
        n = a.shape[0]
        y = torch.stack([torch.ones(n, device=a.device),
                         torch.zeros(n, device=a.device),
                         torch.zeros(n, device=a.device)], dim=1)
        steps = int(round(self.T_end / self.dt))
        every = max(1, steps // self.n_out)
        out = []

        def f(y):
            ra = k1 * y[:, 0]
            rb = k2 * y[:, 1] * y[:, 1]
            return torch.stack([-ra, ra - rb, rb], dim=1)

        for i in range(steps):
            h = self.dt
            k_1 = f(y)
            k_2 = f(y + 0.5 * h * k_1)
            k_3 = f(y + 0.5 * h * k_2)
            k_4 = f(y + h * k_3)
            y = y + (h / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            if (i + 1) % every == 0 and len(out) < self.n_out:
                out.append(y[:, 1].clone())
        while len(out) < self.n_out:
            out.append(y[:, 1].clone())
        return torch.stack(out, dim=-1)[:, None, None, :]

    def solver_accuracy(self):
        return f"RK4, dt={self.dt}, {int(self.T_end / self.dt)} steps"

    def residual(self, a, u):
        """Mass-balance defect: B cannot exceed the initial A, nor go negative.

        Not a full operator residual -- an inequality check -- which is the
        realistic case for a plant model whose equations are not all available.
        It is included precisely so the kit's residual hook is exercised by
        something that is not a clean PDE.
        """
        return (u.clamp(max=0.0).abs() + (u - 1.0).clamp(min=0.0))

    def rhs(self, a):
        return torch.ones_like(self.solve(a)[:, :1, :, :1])
