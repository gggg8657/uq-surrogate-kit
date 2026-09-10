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
from uqkit.ood import consistency_score  # noqa: E402
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


def test_operator_key_partitions_the_ood_suite():
    """Exactly six OOD shards change L; the other 33 change only the input.

    The first run of `eval_consistency.py` compared whole `CFG` dicts and so
    labelled all 39 shards an operator shift, because `alpha` and `tau`
    parameterise the input field and live in the same dict. That made the free
    `lookup` baseline read 1.000 everywhere. This pins the partition so the
    same mistake cannot come back silently.
    """
    import json
    man = json.loads(Path(__file__).resolve().parents[1].joinpath(
        "data", "manifest.json").read_text())
    key = lambda t: PDE2DSimulator(t, device="cpu").operator_key()
    ood = [t for g in man["ood_suite"].values() for t in g]
    changed = sorted(t for t in ood if key(t) != key(P.PARENT[t]))
    assert changed == sorted(["biharmonic", "frac_s0p25", "frac_s0p5",
                              "frac_s3", "navier_stokes", "ns_T0p25"]), changed
    # and the shards that do NOT change it must be exactly the input shifts
    for t in ood:
        if t not in changed:
            assert key(t) == key(P.PARENT[t]), t
    trained = {key(t) for t in P.PRETRAIN_TASKS}
    assert all(key(t) not in trained for t in changed)
    print(f"ok  {len(changed)}/{len(ood)} OOD shards change the operator: "
          f"{', '.join(changed)}")


def test_consistency_score_is_scale_free_and_zero_on_the_truth():
    """`consistency_score` must not rank an operator by its own magnitude.

    `residual_score` divides by ||f||, so applying c*L multiplies the score by
    ~c and a sixth-order request outscores a second-order one whatever the
    prediction is. That is the artefact the dimensionless form exists to avoid,
    and it is the reason a `frac_s3` AUROC cannot be read as detection.
    """
    torch.manual_seed(0)
    sim = PDE2DSimulator("poisson", device=DEV, N=32)
    a = sim.sample_inputs(16, seed=1, N=32)
    u = sim.solve(a)
    f = sim.rhs(a)

    def cons(field, scale=1.0):
        r = sim.residual(a, field) * scale
        return consistency_score(r, r + f * scale, f * scale)

    # zero (to round-off) on the exact solution
    assert float(cons(u).max()) < 1e-3, float(cons(u).max())
    # invariant to rescaling the operator and its right-hand side together
    assert torch.allclose(cons(u, 1.0), cons(u, 1000.0), atol=1e-6)
    # and bounded in [0, 1] for a prediction that is pure noise
    bad = cons(torch.randn_like(u) * float(u.std()))
    assert float(bad.min()) > 0.3 and float(bad.max()) <= 1.0, (
        float(bad.min()), float(bad.max()))
    print(f"ok  consistency {float(cons(u).mean()):.2e} on the truth, "
          f"{float(bad.mean()):.3f} on noise, scale-invariant")


def test_operator_is_resolution_invariant():
    """The FNO evaluates on a grid it never saw -- the property the kit relies on."""
    torch.manual_seed(0)
    m = FNO2d(width=16, modes=8, n_layers=2, n_tasks=4).to(DEV).eval()
    t = torch.zeros(2, dtype=torch.long, device=DEV)
    for N in (32, 64):
        a = torch.randn(2, 2, N, N, device=DEV)
        assert m(a, t).shape == (2, 1, N, N)
    print("ok  FNO2d runs at 32^2 and 64^2 with one set of weights")




def test_spectral_cache_is_bit_identical():
    """The eval-mode packed-weight path must equal the einsum path exactly.

    H12 replaced `einsum("bixy,ioxy->boxy", ...)` with a `bmm` against weights
    permuted once at load, because einsum re-permuted 13.1 MB of fixed weights
    on every forward (`runs/profile_surrogate.json`: `copy_` 258.7 us/call vs
    65.6 us for all the matmuls and FFTs combined). It is the same contraction
    in the same order, so the bar is `max|delta| == 0`, not a tolerance -- the
    same bar the solver's `fast_apply` had to clear before it was admitted as a
    denominator. A tolerance here would let a real numerical change through.
    """
    import torch

    from uqkit.sims.fno2d import FNO2d

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = FNO2d(in_ch=2, out_ch=1, width=16, modes=6, n_layers=2,
              n_tasks=3).to(dev).eval()
    task = torch.zeros(2, dtype=torch.long, device=dev)

    for N in (16, 32):  # N=16 exercises m1 = min(modes, H//2) < modes
        a = torch.randn(2, 2, N, N, device=dev)
        with torch.no_grad():
            for blk in m.blocks:
                blk.spectral.cache_packed_weights = False
            slow = m(a, task)
            for blk in m.blocks:
                blk.spectral.cache_packed_weights = True
                blk.spectral.clear_packed_cache()
            fast = m(a, task)
            fast_again = m(a, task)  # second call hits the cache
        assert (slow - fast).abs().max().item() == 0.0, N
        assert (fast - fast_again).abs().max().item() == 0.0, N

    # The fast path must not engage where gradients are wanted, and must not
    # survive a weight change.
    a = torch.randn(2, 2, 32, 32, device=dev, requires_grad=True)
    out = m(a, task).sum()
    out.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()

    with torch.no_grad():
        m(torch.randn(2, 2, 32, 32, device=dev), task)          # fills cache
        assert m.blocks[0].spectral._packed_cache is not None
        m.blocks[0].spectral.w_lo.mul_(2.0)
        m.blocks[0].spectral.clear_packed_cache()               # explicit drop
        assert m.blocks[0].spectral._packed_cache is None
    m.train()
    assert m.blocks[0].spectral._packed_cache is None
    print("ok  packed-weight spectral path is bit-identical to einsum "
          "(max|delta| = 0 at N=16 and N=32), grads flow, cache invalidates")


if __name__ == "__main__":
    test_operator_key_partitions_the_ood_suite()
    test_consistency_score_is_scale_free_and_zero_on_the_truth()
    test_residual_floor_where_the_detector_is_used()
    test_residual_is_useless_above_fourth_order()
    test_time_families_have_no_cheap_residual()
    test_shift_configs_change_the_input_not_the_operator()
    test_operator_is_resolution_invariant()
    test_spectral_cache_is_bit_identical()
    print("all sim tests passed")
