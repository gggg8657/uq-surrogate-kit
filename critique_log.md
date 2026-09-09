# Critique log — A4, UQ Surrogate Kit

Append-only. Each turn: what was measured, what the published baseline is and
where that figure comes from, why the gap exists, what the binding constraint
is, and what would distinguish the explanation from the obvious alternative.

---

## Turn 1 — 2026-09-09 — build, and three defects the tests found before any result

**State at start.** Empty. `pde-neural-operator` has class-conditional conformal
at 0.90, a 3-member ensemble with spread–error correlation +0.91, and weighted
conformal; `wafer-tool-shift/wts/metrics.py` has the weighted implementation and
its honest note. The brief is to generalize these into a kit with an API, not to
rediscover them. Nothing measured yet, so there is nothing to critique about
results — this entry is about the infrastructure and the four things already
known to be wrong or constrained.

### 1. GPU 2 is unusable, and it is not ours to fix

`nvidia-smi` reports **91 compute contexts of 534 MiB each (48 GB) on physical
GPU 2**, whose PIDs do not exist as processes (`ps` returns nothing for any of
them) — leaked contexts from something outside this workspace. Measured effect,
same probe on both leased devices:

| device | 4096² fp32 matmul | effective TFLOP/s | 64×2×256² rfft2 |
|---|---|---|---|
| GPU 2 | 47.68 ms | **2.88** | 0.899 ms |
| GPU 3 | 3.12 ms | **44.07** | 0.062 ms |

**15.3× degraded.** Confirmed independently by training throughput: the same
member ran at 78 s/epoch on GPU 2 and 4.8 s/epoch on GPU 3.

Consequence, and it is not cosmetic: **the speedup clause must be measured on
GPU 3 only.** A solver-vs-surrogate ratio taken on a device running at 6.5% of
its throughput is not obviously biased in either direction — both sides slow
down — but the FFT and the matmul degrade by different factors (14.4× vs 15.3×),
so the ratio *does* move, and the two sides of the comparison are not the same
mix of kernels. Timing on GPU 2 would be a measurement of the leak.

I did not kill those contexts: the boundary rule forbids touching another loop's
GPUs, and the PIDs are not visible from this namespace in any case. All training,
evaluation and timing runs pinned to `CUDA_VISIBLE_DEVICES=3`. Cost: no
parallelism across the lease, which is affordable — a member trains in 5 minutes.

### 2. A calibration bug my own test caught, which would have inflated coverage

`tests/test_api.py` asserts that `UQSurrogate.interval` and
`UQSurrogate.evaluate` agree sample-for-sample. They did not: 0.9277 vs 0.9258.

Cause: the σ floor. A deep ensemble reports σ≈0 wherever its members happen to
agree, so `|resid|/σ` diverges and the score's 90th percentile becomes a
numerical accident; the standard fix is to floor σ at a fraction of its median.
The first version computed that median **from whatever batch it was handed**.
Calibrate on the calibration split, evaluate on a shifted set with a larger
spread, and the shifted set silently widens its own interval — repairing part of
the very undercoverage the measurement exists to expose.

The effect in distribution was ~0.2pp. Under a shift where σ grows by 2×, the
floor term grows with it, and the direction is always *toward* apparent
coverage. This is precisely the failure mode the weekend rules name: loosening
the test to improve the number, without meaning to.

Fixed: `_floor(sigma, frac, med)` takes the median explicitly, `UQSurrogate`
freezes it at `calibrate()` before computing any score, and `scores()` raises if
called before calibration. `eval_conformal.py` freezes it the same way. The
`interval`/`evaluate` identity is now a test with a one-sample float32 boundary
tolerance, so a regression cannot pass silently.

### 3. The residual detector has a noise floor set by the order of the operator

The physics-residual score `‖L û − f‖/‖f‖` is the kit's most attractive idea: it
needs no ground truth, and for Darcy it costs one sparse apply against a solve
of thousands of PCG iterations. Before using it I measured what it reports on
the **exact** solution — the floor below which it can resolve nothing.
`scripts/measure_residual_floor.py`, `runs/residual_floor.json`:

| family | symbol order | N=32 | N=64 | N=128 |
|---|---:|---:|---:|---:|
| `frac_s0p5` | 1 | 9.5e-07 | 1.8e-06 | 4.0e-06 |
| `poisson` | 2 | 1.2e-05 | 5.1e-05 | 2.1e-04 |
| `helmholtz` | 2 | 3.8e-06 | 1.7e-05 | 7.8e-05 |
| `darcy` | 2 | 6.9e-06 | 3.8e-05 | 1.8e-04 |
| `frac_s1p5` | 3 | 2.0e-04 | 1.6e-03 | 1.1e-02 |
| `biharmonic` | 4 | 3.3e-03 | 4.2e-02 | 7.0e-01 |
| `frac_s2p5` | 5 | 4.2e-02 | 1.4e+00 | 4.6e+01 |
| `frac_s3` | 6 | 7.6e-01 | 5.3e+01 | 3.4e+03 |

Applying `L` amplifies the round-off already in `u` by the symbol, so the floor
scales roughly as `N^order`. At sixth order the residual of the exact answer is
**53× the right-hand side** at 64². Had I shipped the detector without this
table, `frac_s3` would have produced a magnificent AUROC — perfect separation
driven entirely by round-off, and completely uninformative about the surrogate.

What distinguishes this explanation from the obvious alternative ("the solver is
wrong"): solving *and* applying in float64 puts every family at 2.3e-8. The
error is in the fp32 solution, not in the operator. The apply is now done in
float64 (it removes its own contribution and costs one FFT pair), the floor is a
regenerated artifact, and `tests/test_sims.py` pins **both** directions — the
low-order families below their floor and the high-order ones above it — so a
future reader cannot quietly extend the detector upward.

Consequence for the KPI: the residual detector is quoted only for Poisson,
Helmholtz and Darcy, where the floor is 15–150× below the surrogate's error.

### 4. An ensemble jittered in the wrong variable measures the wrong thing

In `tests/test_api.py` the reactor surrogate started as four coarse-RK4 members
differing by a 3% jitter in the rate constants. Error-detection AUROC: **0.550**
— chance. The coarse integrator's error is dominated by *discretization*, which
is identical across members that share a step size, so their spread measures
parameter sensitivity and is blind to the actual failure.

Changing the members to differ in **step size** (6e-3 … 1.2e-2), a
Richardson-style estimator, moves the same AUROC to **0.999**.

This is a warning about the FNO ensemble, not just the toy. Five members that
differ only in initialization and data order share their architecture, their
resolution, their training distribution and their inductive bias. Whatever error
those share, the spread cannot see — which is the mechanism that would explain a
high in-distribution spread–error correlation collapsing under a shift that hits
all five members identically. `unseen_operator` is exactly such a shift, and it
is in the suite for that reason. Prediction, recorded before the numbers exist:
**spread will detect input shifts and fail on operator shifts.**

### Protocol fixed before any evaluation run

- α = 0.1; band [88%, 92%]; Wilson 95% interval on every coverage. On a
  512-sample shard that interval is ±2.6pp — **wider than the KPI band** — so no
  single shard can decide a clause, and the verdict counts shards in band.
- Splits `train` / `cal` / `test` disjoint by seed. Weighted conformal sees the
  shifted set's *inputs* only, never its residuals.
- Detectors are fitted on **training** inputs only. Nothing is tuned on an OOD
  shard; the reported AUROCs are of raw single-feature scores with no learned
  combination, so there is no tuning loop to leak.
- The speedup is reported per family, per batch size, per device, and for
  `single` / `ensemble` / `ensemble+residual` separately. Families whose
  reference solver is one exact FFT pair — Poisson, Helmholtz, diffusion,
  advection–diffusion — are expected to show the surrogate *losing*, and those
  rows stay in the table. Dropping them is how a speedup table becomes a
  selection effect.

### Open, to be answered by measurement, not by argument

1. Does `field_max` (simultaneous over 4,096 pixels) calibrate at all at n=1,024,
   or is its 90th percentile too far into the tail to estimate?
2. Does weighted conformal restore coverage on `input_shift`, and does it — as
   predicted — do nothing on `unseen_operator`, where the probe should report
   AUC ≈ 0.5?
3. Which families clear 100×? Darcy (PCG) and Navier–Stokes (1,000 RK4 steps)
   should; the four exact-propagator families should not. Untrained-family rows
   are timed for the solver only and marked as such.
