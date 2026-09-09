# Vendored from pde-neural-operator/pdeno/pde2d.py (same author, 2026-09).
# Kept in-repo so this kit is self-contained and CI-testable; extended below
# with the distribution-shift task configs this kit needs.
"""2D PDE operator datasets, generated entirely on-GPU (no external downloads).

Six PDE families on the periodic unit square, each posed as an *operator*
problem  a(x, y) -> u(x, y):

| task            | equation                                   | input -> output      | solver        |
|-----------------|--------------------------------------------|----------------------|---------------|
| `poisson`       | Lap u = -f                                  | f       -> u         | exact (FFT)   |
| `helmholtz`     | Lap u + k^2 u = -f                          | f       -> u         | exact (FFT)   |
| `diffusion`     | u_t = nu Lap u                              | u(0)    -> u(T)      | exact (FFT)   |
| `advdiff`       | u_t + c . grad u = nu Lap u                 | u(0)    -> u(T)      | exact (FFT)   |
| `darcy`         | -div(a grad u) = f                          | (log a, f) -> u      | batched PCG   |
| `navier_stokes` | w_t + u . grad w = nu Lap w + f, u = curl^-1 w | w(0) -> w(T)      | pseudo-spectral RK4 |

Every sample is drawn from a Gaussian random field, so the corpus is unbounded
and reproducible from a seed alone -- the point being a multi-PDE pretraining
corpus that needs no PDEBench download.

Input tensors are always `(n, 2, N, N)`: channel 0 is the primary input
function, channel 1 carries a second field (Darcy's forcing) or zeros. That
uniform layout is what lets one operator train across all six families.
"""
from __future__ import annotations

import math

import torch

# The seven original families. Their order and therefore their ids are frozen:
# every checkpoint under `runs/` stores a task-embedding row per id.
CORE_TASKS = ("poisson", "helmholtz", "diffusion", "advdiff", "darcy",
              "navier_stokes", "biharmonic")

# ---------------------------------------------------------------------------
# Interpolating families, added for the transfer-distance study.
#
# Seven families give a *binary* nonlinearity axis -- five Fourier multipliers
# and two that are not -- and a correlation against a two-valued predictor is a
# weak result however large it comes out. These sweep one property at a time:
#
#   `frac_s*`   (-Lap)^s u = f, symbol |xi|^(-2s). Exactly a linear multiplier
#               at every s, so the multiplier defect stays at round-off while
#               the differential order varies continuously. s=1 reproduces
#               poisson and s=2 reproduces biharmonic, which is a free
#               consistency check on both the solver and the distance estimator.
#   `darcy_c*`  Darcy at log-coefficient contrast c. c -> 0 *is* Poisson, so
#               this sweeps the multiplier defect upward with the equation type
#               held fixed. (The original `darcy` is c = 1.5.)
#   `ns_T*`     Navier-Stokes integrated to a shorter horizon. As T -> 0 the
#               solution map tends to the identity plus a small advective
#               correction, so its distance from a multiplier should fall
#               smoothly rather than switching off.
#
# One axis varies the symbol at fixed multiplier-ness, the other varies
# multiplier-ness at fixed equation type. That is what makes the two hypotheses
# separable.
# ---------------------------------------------------------------------------
FRAC_S = (0.25, 0.5, 0.75, 1.5, 2.5, 3.0)
DARCY_C = (0.25, 0.5, 1.0, 3.0)
NS_T = (0.25, 0.5)

def _tag(x):
    return f"{x:g}".replace(".", "p")

FRAC_TASKS = tuple(f"frac_s{_tag(s)}" for s in FRAC_S)
DARCY_TASKS = tuple(f"darcy_c{_tag(c)}" for c in DARCY_C)
NS_TASKS = tuple(f"ns_T{_tag(t)}" for t in NS_T)
SWEEP_TASKS = FRAC_TASKS + DARCY_TASKS + NS_TASKS

TASKS = CORE_TASKS + SWEEP_TASKS
TASK_ID = {t: i for i, t in enumerate(TASKS)}
# which solver branch each family dispatches to
FAMILY_BASE = ({t: t for t in CORE_TASKS}
               | {t: "fractional" for t in FRAC_TASKS}
               | {t: "darcy" for t in DARCY_TASKS}
               | {t: "navier_stokes" for t in NS_TASKS})
PRETRAIN_TASKS = ("poisson", "helmholtz", "diffusion", "advdiff", "darcy")
# Two transfer probes, deliberately at different distances from the pretraining
# set: `navier_stokes` is the only nonlinear family, `biharmonic` is a linear
# elliptic operator the model has never seen. Comparing the two separates "this
# pretraining is useless" from "this target is too far away".
HELDOUT_TASKS = ("navier_stokes", "biharmonic")
HELDOUT_TASK = "navier_stokes"

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# spectral helpers
# --------------------------------------------------------------------------- #
def wavenumbers(N, device, dtype=torch.float32):
    """Angular wavenumbers (kx, ky) on the full FFT grid, shape (N, N)."""
    k = torch.fft.fftfreq(N, d=1.0 / N, device=device, dtype=dtype) * TWO_PI
    return torch.meshgrid(k, k, indexing="ij")


def laplacian_symbol(N, device, dtype=torch.float32):
    """|xi|^2 on the FFT grid -- the symbol of -Laplacian."""
    kx, ky = wavenumbers(N, device, dtype)
    return kx**2 + ky**2


def grf(n, N, device, gen, alpha=2.5, tau=7.0, dtype=torch.float32):
    """Gaussian random field with spectrum (|xi|^2 + tau^2)^(-alpha/2).

    Zero-mean, unit-variance per sample. `alpha` sets smoothness.
    """
    k2 = laplacian_symbol(N, device, dtype)
    coef = (k2 + tau**2) ** (-alpha / 2.0)
    noise = torch.randn(n, N, N, device=device, dtype=dtype, generator=gen)
    f = torch.fft.ifft2(torch.fft.fft2(noise) * coef).real
    f = f - f.mean(dim=(-2, -1), keepdim=True)
    f = f / f.std(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return f


def _pack(primary, secondary=None):
    """(n, N, N) fields -> the uniform (n, 2, N, N) input layout."""
    if secondary is None:
        secondary = torch.zeros_like(primary)
    return torch.stack([primary, secondary], dim=1)


# --------------------------------------------------------------------------- #
# exact spectral solvers
# --------------------------------------------------------------------------- #
def solve_poisson(f):
    """Lap u = -f, periodic, zero-mean.  f: (n, N, N) -> u: (n, N, N)."""
    N = f.shape[-1]
    k2 = laplacian_symbol(N, f.device, f.dtype)
    inv = torch.where(k2 > 0, 1.0 / k2.clamp_min(1e-12), torch.zeros_like(k2))
    return torch.fft.ifft2(torch.fft.fft2(f) * inv).real


def solve_fractional(f, s):
    """(-Lap)^s u = f, periodic, zero-mean -- the symbol is |xi|^(-2s).

    s = 1 is exactly `solve_poisson` and s = 2 is exactly `solve_biharmonic`;
    both are kept as separate functions because they are what the original study
    used, and the equality is a test (`tests/test_pde2d.py`).
    """
    N = f.shape[-1]
    k2 = laplacian_symbol(N, f.device, f.dtype)
    inv = torch.where(k2 > 0, k2.clamp_min(1e-12) ** (-s), torch.zeros_like(k2))
    return torch.fft.ifft2(torch.fft.fft2(f) * inv).real


def solve_biharmonic(f):
    """Lap^2 u = f, periodic, zero-mean -- the symbol is |xi|^4."""
    N = f.shape[-1]
    k2 = laplacian_symbol(N, f.device, f.dtype)
    inv = torch.where(k2 > 0, 1.0 / (k2 ** 2).clamp_min(1e-12), torch.zeros_like(k2))
    return torch.fft.ifft2(torch.fft.fft2(f) * inv).real


def solve_helmholtz(f, kappa2):
    """Lap u + kappa^2 u = -f, periodic.

    kappa2 is deliberately off-resonance ((2 pi)^2 * 12.5 is never |xi|^2 on an
    integer grid) so the symbol |xi|^2 - kappa^2 never vanishes.
    """
    N = f.shape[-1]
    k2 = laplacian_symbol(N, f.device, f.dtype)
    return torch.fft.ifft2(torch.fft.fft2(f) / (k2 - kappa2)).real


def solve_diffusion(u0, nu, T):
    """u_t = nu Lap u -> exact propagator exp(-nu |xi|^2 T)."""
    N = u0.shape[-1]
    k2 = laplacian_symbol(N, u0.device, u0.dtype)
    return torch.fft.ifft2(torch.fft.fft2(u0) * torch.exp(-nu * k2 * T)).real


def solve_advdiff(u0, nu, cx, cy, T):
    """u_t + c . grad u = nu Lap u -> exp(-(nu |xi|^2 + i c . xi) T)."""
    N = u0.shape[-1]
    kx, ky = wavenumbers(N, u0.device, u0.dtype)
    symbol = torch.exp(-(nu * (kx**2 + ky**2)) * T) * torch.exp(
        -1j * (cx * kx + cy * ky) * T
    )
    return torch.fft.ifft2(torch.fft.fft2(u0) * symbol).real


# --------------------------------------------------------------------------- #
# Darcy: variable-coefficient elliptic solve by batched preconditioned CG
# --------------------------------------------------------------------------- #
def _darcy_apply(a, u):
    """Matrix-free  A u = -div(a grad u)  on the periodic 5-point FD stencil.

    Face coefficients are arithmetic means; A is symmetric positive
    semi-definite with the constants as its null space.
    """
    N = u.shape[-1]
    h2 = 1.0 / (N * N)
    a_xp = 0.5 * (a + torch.roll(a, -1, dims=-2))
    a_xm = 0.5 * (a + torch.roll(a, 1, dims=-2))
    a_yp = 0.5 * (a + torch.roll(a, -1, dims=-1))
    a_ym = 0.5 * (a + torch.roll(a, 1, dims=-1))
    out = (
        a_xp * (u - torch.roll(u, -1, dims=-2))
        + a_xm * (u - torch.roll(u, 1, dims=-2))
        + a_yp * (u - torch.roll(u, -1, dims=-1))
        + a_ym * (u - torch.roll(u, 1, dims=-1))
    )
    return out / h2


def _zero_mean(x):
    return x - x.mean(dim=(-2, -1), keepdim=True)


def solve_darcy(a, f, tol=1e-10, max_iter=2000, check_every=1):
    """Solve -div(a grad u) = f (periodic, zero-mean) with batched PCG.

    Preconditioner: the constant-coefficient spectral inverse scaled by the
    per-sample mean of `a`, which collapses the iteration count to O(10-100)
    even at high coefficient contrast. The iteration runs in float64 -- in
    float32 it stagnates near 1e-3 relative residual, which would put the
    "ground truth" at the same order as the model error it is meant to measure.
    Returns (u, residual_ratio) with u cast back to the input dtype.

    `check_every` amortizes the convergence test. The test calls `.max()` and
    compares it in Python, which forces a device-to-host synchronization on
    every iteration -- an adversarial review of the speedup benchmark pointed
    out, correctly, that this makes the reference solver slower than it needs to
    be and therefore subsidizes any surrogate timed against it. The default is
    1, which is what generated the corpus and what `bench_speedup.py` times, so
    the headline number is unchanged; `bench_isoaccuracy.py` also times
    `check_every=10` so the size of that subsidy is measured rather than
    argued about.
    """
    out_dtype = a.dtype
    a, f = a.double(), f.double()
    N = a.shape[-1]
    k2 = laplacian_symbol(N, a.device, a.dtype)
    inv_k2 = torch.where(k2 > 0, 1.0 / k2.clamp_min(1e-12), torch.zeros_like(k2))
    a_bar = a.mean(dim=(-2, -1), keepdim=True)

    def precond(r):
        rhat = torch.fft.fft2(_zero_mean(r))
        return torch.fft.ifft2(rhat * inv_k2).real / a_bar

    def dot(x, y):
        return (x * y).sum(dim=(-2, -1), keepdim=True)

    b = _zero_mean(f)
    b_norm = b.flatten(1).norm(dim=1).clamp_min(1e-30)
    u = torch.zeros_like(b)
    r = b.clone()
    z = precond(r)
    p = z.clone()
    rz = dot(r, z)

    for it in range(max_iter):
        Ap = _zero_mean(_darcy_apply(a, p))
        pAp = dot(p, Ap).clamp_min(1e-30)
        alpha = rz / pAp
        u = u + alpha * p
        r = r - alpha * Ap
        if (it + 1) % check_every == 0 and \
                (r.flatten(1).norm(dim=1) / b_norm).max() < tol:
            break
        z = precond(r)
        rz_new = dot(r, z)
        p = z + (rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new

    u = _zero_mean(u)
    resid = (
        (_zero_mean(_darcy_apply(a, u)) - b).flatten(1).norm(dim=1) / b_norm
    ).max().item()
    return u.to(out_dtype), resid


# --------------------------------------------------------------------------- #
# Navier-Stokes 2D, vorticity form, pseudo-spectral RK4 with 2/3 dealiasing
# --------------------------------------------------------------------------- #
def _ns_rhs(w_hat, kx, ky, inv_k2, nu, f_hat, mask):
    psi_hat = w_hat * inv_k2
    u = torch.fft.ifft2(1j * ky * psi_hat).real          #  d psi / dy
    v = torch.fft.ifft2(-1j * kx * psi_hat).real         # -d psi / dx
    wx = torch.fft.ifft2(1j * kx * w_hat).real
    wy = torch.fft.ifft2(1j * ky * w_hat).real
    nonlin = torch.fft.fft2(u * wx + v * wy) * mask      # dealiased advection
    return -nonlin - nu * (kx**2 + ky**2) * w_hat + f_hat


def solve_navier_stokes(w0, nu=1e-3, T=1.0, dt=1e-3, forcing=True, snapshots=None):
    """Advance 2D vorticity from w0 to time T.

    Kolmogorov-style forcing f = 0.1 (sin(2 pi (x+y)) + cos(2 pi (x+y))), the
    standard FNO-paper setup. `snapshots` is an optional sorted list of times at
    which to also record w, which is what the autoregressive rollout eval uses.
    Returns (w_T, {t: w_t}).
    """
    N = w0.shape[-1]
    device, dtype = w0.device, w0.dtype
    kx, ky = wavenumbers(N, device, dtype)
    k2 = kx**2 + ky**2
    inv_k2 = torch.where(k2 > 0, 1.0 / k2.clamp_min(1e-12), torch.zeros_like(k2))
    k_max = TWO_PI * (N // 3)                            # 2/3 rule
    mask = ((kx.abs() <= k_max) & (ky.abs() <= k_max)).to(dtype)

    if forcing:
        x = torch.arange(N, device=device, dtype=dtype) / N
        X, Y = torch.meshgrid(x, x, indexing="ij")
        f = 0.1 * (torch.sin(TWO_PI * (X + Y)) + torch.cos(TWO_PI * (X + Y)))
        f_hat = torch.fft.fft2(f)
    else:
        f_hat = torch.zeros(N, N, device=device, dtype=torch.complex64)

    w_hat = torch.fft.fft2(w0)
    n_steps = int(round(T / dt))
    want = sorted(snapshots or [])
    out = {}
    for step in range(n_steps):
        k1 = _ns_rhs(w_hat, kx, ky, inv_k2, nu, f_hat, mask)
        k2_ = _ns_rhs(w_hat + 0.5 * dt * k1, kx, ky, inv_k2, nu, f_hat, mask)
        k3 = _ns_rhs(w_hat + 0.5 * dt * k2_, kx, ky, inv_k2, nu, f_hat, mask)
        k4 = _ns_rhs(w_hat + dt * k3, kx, ky, inv_k2, nu, f_hat, mask)
        w_hat = w_hat + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)
        t = (step + 1) * dt
        while want and t >= want[0] - 1e-9:
            out[want.pop(0)] = torch.fft.ifft2(w_hat).real.clone()
    return torch.fft.ifft2(w_hat).real, out


# --------------------------------------------------------------------------- #
# task registry
# --------------------------------------------------------------------------- #
CFG = {
    "poisson": dict(alpha=2.5, tau=7.0),
    "helmholtz": dict(alpha=2.5, tau=7.0, kappa2=TWO_PI**2 * 12.5),
    "diffusion": dict(alpha=3.0, tau=5.0, nu=2e-3, T=1.0),
    "advdiff": dict(alpha=3.0, tau=5.0, nu=1e-3, cx=0.7, cy=-0.4, T=1.0),
    "darcy": dict(alpha=2.0, tau=5.0, contrast=1.5),
    "navier_stokes": dict(alpha=2.5, tau=7.0, nu=1e-3, T=1.0, dt=1e-3, w_scale=5.0),
    "biharmonic": dict(alpha=2.5, tau=7.0),
}
# The sweeps inherit their base family's GRF settings, so the *input*
# distribution is identical along each axis and only the operator changes.
CFG.update({t: dict(alpha=2.5, tau=7.0, s=s_) for t, s_ in zip(FRAC_TASKS, FRAC_S)})
CFG.update({t: CFG["darcy"] | dict(contrast=c) for t, c in zip(DARCY_TASKS, DARCY_C)})
CFG.update({t: CFG["navier_stokes"] | dict(T=T) for t, T in zip(NS_TASKS, NS_T)})


@torch.no_grad()
def generate(task, n, N=64, seed=0, device="cuda", chunk=512, snapshots=None):
    """Generate `n` operator samples for `task`.

    Returns (a, u) with a: (n, 2, N, N), u: (n, 1, N, N) -- plus a dict of extra
    tensors (Darcy residual, NS snapshots) as a third element.
    """
    if task not in CFG:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(CFG)}")
    cfg = CFG[task]
    base = FAMILY_BASE[task]
    gen = torch.Generator(device=device).manual_seed(seed)
    a_parts, u_parts, extra = [], [], {}
    snap_parts = {}
    worst_resid = 0.0

    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        g = grf(m, N, device, gen, alpha=cfg["alpha"], tau=cfg["tau"])
        if cfg.get("amp", 1.0) != 1.0:
            g = g * cfg["amp"]

        if base == "poisson":
            a, u = _pack(g), solve_poisson(g)
        elif base == "biharmonic":
            a, u = _pack(g), solve_biharmonic(g)
        elif base == "fractional":
            a, u = _pack(g), solve_fractional(g, cfg["s"])
        elif base == "helmholtz":
            a, u = _pack(g), solve_helmholtz(g, cfg["kappa2"])
        elif base == "diffusion":
            a, u = _pack(g), solve_diffusion(g, cfg["nu"], cfg["T"])
        elif base == "advdiff":
            a, u = _pack(g), solve_advdiff(g, cfg["nu"], cfg["cx"], cfg["cy"], cfg["T"])
        elif base == "darcy":
            coef = torch.exp(cfg["contrast"] * g)                 # log-normal a
            f = grf(m, N, device, gen, alpha=2.5, tau=7.0)
            u, resid = solve_darcy(coef, f)
            worst_resid = max(worst_resid, resid)
            a = _pack(torch.log(coef), f)
        else:  # navier_stokes
            w0 = cfg["w_scale"] * g
            # keep the advective CFL fixed as the grid refines
            dt = cfg["dt"] * min(1.0, 64.0 / N)
            # `snapshots` asks for states beyond the operator's own horizon
            # (the rollout probe), so integrate to the furthest one and take
            # the training target from the T snapshot.
            wanted = sorted({cfg["T"], *(snapshots or [])})
            T_run = max(wanted)
            _, snaps = solve_navier_stokes(
                w0, cfg["nu"], T_run, dt, snapshots=wanted
            )
            u = snaps.pop(cfg["T"])
            a = _pack(w0)
            for t, w in snaps.items():
                snap_parts.setdefault(t, []).append(w.cpu())

        a_parts.append(a.cpu())
        u_parts.append(u.unsqueeze(1).cpu())

    if task == "darcy":
        extra["cg_residual"] = worst_resid
    if snap_parts:
        extra["snapshots"] = {t: torch.cat(v) for t, v in snap_parts.items()}
    return torch.cat(a_parts), torch.cat(u_parts), extra


# --------------------------------------------------------------------------- #
# Distribution-shift task configs, added for the UQ kit.
#
# The kit's OOD clause needs shifted sets that the surrogate never trained on.
# The `SWEEP_TASKS` above are not enough on their own: by construction they
# inherit their parent's GRF settings, so the *input* distribution is identical
# and only the operator changes. That makes them a clean probe of operator
# shift -- and, as it turns out, an input-space detector's worst case.
#
# These add the other axis: the same operator, a different input distribution.
# Each carries a `parent` -- the task id whose embedding and normalization
# statistics the surrogate must be evaluated under, because at deployment the
# configured task does not change when the incoming field does.
#
#   `*_rough`   GRF exponent alpha 2.5 -> 1.5: more energy at high wavenumber.
#   `*_smooth`  alpha 2.5 -> 4.0.
#   `*_tau`     correlation length tau 7 -> 16.
#   `*_amp2`    input scaled x2 after normalization to unit variance.
#
# Out-of-range *parameters* are covered by `darcy_c3` (contrast 3.0 against a
# trained 1.5) and `frac_s3` / `frac_s0p25` from the sweep above; unseen
# resolution is a generation argument, not a task.
# --------------------------------------------------------------------------- #
SHIFT_SPECS = {
    "rough": dict(alpha=1.5),
    "smooth": dict(alpha=4.0),
    "tau": dict(tau=16.0),
    "amp2": dict(amp=2.0),
}
SHIFT_PARENTS = ("poisson", "helmholtz", "diffusion", "advdiff", "darcy")

SHIFT_TASKS = tuple(
    f"{p}_{s}" for p in SHIFT_PARENTS for s in SHIFT_SPECS
)
# parent task for every task, shifted or not: what the surrogate is configured
# as when the sample arrives.
PARENT = {t: t for t in TASKS}
PARENT.update({f"{p}_{s}": p for p in SHIFT_PARENTS for s in SHIFT_SPECS})
# the sweep families are operator shifts of their own base
PARENT.update({t: "poisson" for t in FRAC_TASKS})
PARENT.update({t: "darcy" for t in DARCY_TASKS})
# An unseen operator has to be presented under a task the surrogate actually
# has, because that is the deployment failure being modelled: the process
# changed and nobody reconfigured the model. The parent is chosen by *problem
# shape*, not by similarity of the answer --
#   biharmonic -> poisson    (both f -> u elliptic, identical input distribution)
#   frac_s*    -> poisson    (same)
#   NS         -> diffusion  (both u(0) -> u(T) initial-value problems; NS is
#                             diffusion plus a nonlinear advective term)
# Picking the *nearest* trained family is the charitable choice: it is the one a
# deployment would plausibly have configured, and it gives the detectors their
# best chance rather than a straw man.
PARENT.update({t: "diffusion" for t in NS_TASKS})
PARENT["navier_stokes"] = "diffusion"
PARENT["biharmonic"] = "poisson"

for _p in SHIFT_PARENTS:
    for _s, _ov in SHIFT_SPECS.items():
        CFG[f"{_p}_{_s}"] = CFG[_p] | _ov
        FAMILY_BASE[f"{_p}_{_s}"] = FAMILY_BASE[_p]

ALL_TASKS = TASKS + SHIFT_TASKS


# --------------------------------------------------------------------------- #
# A graded roughness ladder, added after turn 1.
#
# The first shift set jumped straight from alpha 2.5 to 1.5, and the result was
# a cliff: the shift probe separated calibration from test at AUC 1.00, and
# weighted conformal recovered nothing, because with disjoint supports there is
# no calibration point resembling a test point to up-weight. "Weighted conformal
# does not help" is a much weaker statement than "it holds to a shift of size X
# and fails beyond it", and only the second one tells a user anything.
#
# So: the same axis, in small steps, relative to each parent's own alpha. The
# quantity to plot against is the shift probe's AUC -- a monotone, protocol-free
# measure of how far the input distribution actually moved -- not the nominal
# delta, which means different things for different parents.
# --------------------------------------------------------------------------- #
GRADED_DALPHA = (-0.1, -0.2, -0.3, -0.5, -0.7, -1.0)
GRADED_PARENTS = ("poisson", "darcy")


def _dtag(d):
    return f"m{abs(d):g}".replace(".", "p")


GRADED_TASKS = tuple(f"{p}_da{_dtag(d)}"
                     for p in GRADED_PARENTS for d in GRADED_DALPHA)
for _p in GRADED_PARENTS:
    for _d in GRADED_DALPHA:
        _t = f"{_p}_da{_dtag(_d)}"
        CFG[_t] = CFG[_p] | dict(alpha=CFG[_p]["alpha"] + _d)
        FAMILY_BASE[_t] = FAMILY_BASE[_p]
        PARENT[_t] = _p

ALL_TASKS = ALL_TASKS + GRADED_TASKS
