"""The whole kit, end to end, on a simulator that is not a PDE and not a GPU.

If `uqkit` only ever worked on the FNO corpus it would be one project's eval
script. This runs calibration, shift detection, error detection and the speedup
harness against `BatchReactorSim` -- three coupled ODEs, a different state
shape, no spectral structure, CPU only -- with a *multi-fidelity* ensemble as
the surrogate: the same integrator at a 40x coarser step, its members
differing by small perturbations of the rate constants.

That is a real surrogate with a real speedup and a real error, so the test
checks the pipeline's behaviour, not just that it runs: coverage must land in
the band in distribution, and the shift AUROC must beat chance on a shift that
is genuinely detectable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uqkit import (DeepEnsemble, Simulator, UQSurrogate, rel_l2,  # noqa: E402
                   shift_auroc)
from uqkit.bench import timeit  # noqa: E402
from uqkit.ood import error_auroc  # noqa: E402
from uqkit.sims.toy import BatchReactorSim  # noqa: E402


# Members differ in *step size*, not only in a parameter jitter. That choice is
# the whole reason the spread means anything here: the coarse integrator's error
# is dominated by discretization, which is common to every member if they all
# share a dt -- an ensemble jittered only in its rate constants measures
# parameter sensitivity and ranks the true error at AUROC 0.55, measured. A
# step-size ensemble is a Richardson-style error estimator and ranks it.
STEP_SIZES = (6e-3, 8e-3, 1.0e-2, 1.2e-2)


def make_ensemble(steps=STEP_SIZES, seed=0):
    """Multi-fidelity members: the same RK4 at four different step sizes."""
    g = torch.Generator().manual_seed(seed)
    members = []
    for dt in steps:
        jitter = (1.0 + 0.01 * torch.randn(2, generator=g))

        def member(a, dt=dt, j=jitter):
            return BatchReactorSim(dt=dt).solve(
                a + (j - 1.0)[None, :, None, None])
        members.append(member)
    return DeepEnsemble(members, name="multifidelity_rk4")


def test_protocol_conformance():
    sim = BatchReactorSim()
    assert isinstance(sim, Simulator)
    print("ok  BatchReactorSim satisfies the Simulator protocol")


def test_end_to_end_coverage_and_ood():
    sim = BatchReactorSim(dt=2e-4)
    ens = make_ensemble()

    a_tr = sim.sample_inputs(256, seed=1)
    a_cal = sim.sample_inputs(512, seed=2)
    a_te = sim.sample_inputs(512, seed=3)
    # a shift the detector should see: inputs displaced by two sigma. A
    # Mahalanobis score is a squared distance, so its separation is set by the
    # noncentrality 2*shift^2 -- at 1.2 sigma the achievable AUROC is only ~0.74
    # and a threshold above that would be testing luck, not the detector.
    a_ood = sim.sample_inputs(512, seed=4, shift=2.0)

    def run(a):
        mean, sigma, _ = ens.predict(a)
        return mean, sigma, sim.solve(a)

    m_cal, s_cal, u_cal = run(a_cal)
    m_te, s_te, u_te = run(a_te)
    m_od, s_od, u_od = run(a_ood)

    # The reactor's "input" is two scalars, not a field, so the default
    # spectral feature map is degenerate on it (one band, zero variance). Swap
    # in the identity: `feature_fn` is a constructor argument precisely so a
    # simulator whose inputs are not fields does not have to pretend they are.
    uq = UQSurrogate(ens, score="field_max", alpha=0.1,
                     feature_fn=lambda a: a.flatten(1))
    uq.calibrate(m_cal, s_cal, u_cal, inputs=a_cal)
    uq.fit_detector(a_tr)

    cov = uq.evaluate(m_te, s_te, u_te)
    assert 0.85 <= cov["coverage"] <= 0.95, cov
    lo, hi = uq.interval(m_te, s_te)
    assert torch.all(hi >= lo)
    inside = ((u_te >= lo) & (u_te <= hi)).flatten(1).all(1).float().mean()
    # Tolerance of one sample, not zero: `evaluate` forms |resid| / width and
    # compares to q, while `interval` forms mean +/- q*width and compares to u.
    # Those are the same inequality in exact arithmetic and can disagree on a
    # single sample sitting exactly on the boundary in float32. More than one
    # sample apart would mean the two paths use different widths, which is the
    # failure this assertion is here for.
    n_te = u_te.shape[0]
    assert abs(float(inside) - cov["coverage"]) <= 1.5 / n_te, (float(inside), cov)
    print(f"ok  in-distribution coverage {100*cov['coverage']:.1f}% "
          f"(CI {100*cov['ci95'][0]:.1f}-{100*cov['ci95'][1]:.1f}), "
          f"and `interval` agrees with `evaluate`")

    cov_ood = uq.evaluate(m_od, s_od, u_od)
    print(f"    coverage under the input shift: {100*cov_ood['coverage']:.1f}%")

    # shift detection: an input-space detector must beat chance on this shift
    d_in = uq.ood_score(inputs=a_te, kind="mahalanobis").numpy()
    d_od = uq.ood_score(inputs=a_ood, kind="mahalanobis").numpy()
    a_shift = shift_auroc(d_in, d_od)
    assert a_shift > 0.80, a_shift
    print(f"ok  mahalanobis shift AUROC {a_shift:.3f}")

    # error detection: the ensemble spread must rank the coarse solver's failures
    err = torch.cat([rel_l2(m_te, u_te), rel_l2(m_od, u_od)]).numpy()
    spr = torch.cat([uq.ood_score(mean=m_te, sigma=s_te),
                     uq.ood_score(mean=m_od, sigma=s_od)]).numpy()
    tau = float(np.quantile(rel_l2(m_te, u_te).numpy(), 0.9))
    a_err = error_auroc(spr, err, tau)
    assert a_err > 0.85, a_err
    print(f"ok  spread error-detection AUROC {a_err:.3f} at tau={tau:.4f}")


def test_bench_harness_on_cpu():
    """The speedup harness produces a ratio with the accuracy attached."""
    sim_fine = BatchReactorSim(dt=2e-4)
    ens = make_ensemble()
    a = sim_fine.sample_inputs(16, seed=5)
    u = sim_fine.solve(a)
    mean, _, _ = ens.predict(a)
    err = float(rel_l2(mean, u).mean())
    t_sol = timeit(lambda: sim_fine.solve(a), 1, 3, device="cpu")
    t_sur = timeit(lambda: ens.predict(a), 1, 3, device="cpu")
    sp = t_sol["median_s"] / t_sur["median_s"]
    assert sp > 1.0, sp
    assert 0.0 < err < 0.5, err
    print(f"ok  coarse-RK4 ensemble: {sp:.1f}x the fine solver at rel-L2 {err:.4f} "
          f"({torch.get_num_threads()} CPU threads)")


if __name__ == "__main__":
    test_protocol_conformance()
    test_end_to_end_coverage_and_ood()
    test_bench_harness_on_cpu()
    print("all api tests passed")
