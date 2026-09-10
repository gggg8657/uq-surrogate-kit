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

---

## Turn 2 — 2026-09-09 — every clause measured, every clause missed, one cause

Five members trained (26.25M params each, 60 epochs, ~5 min/member on GPU 3),
in-distribution rel-L2 0.0013–0.0430 across the five families. All three KPI
clauses now have numbers. All three fail, and the failures are more informative
than a pass would have been.

### Clause 1 — coverage 90±2%

**In distribution: met, on all four scores.** Pooled, per-family quantile:
`field_max` 88.6% [87.7, 89.5], `norm_ratio` 88.8%, `rel_l2` 89.7%,
`pixel` 89.8%. Per family, `field_max` runs 86.5% (diffusion) to 90.2%
(poisson). This is the easy half and the brief said so.

**Under shift: missed, and the failure is the opposite of the expected one.**
Of 32 shards where weighted conformal's covariate-shift assumption actually
holds, **0 are in band under plain split conformal and 2 under weighted**. But
the dominant failure is **over**-coverage, not under-coverage: 24 of 32 shards
sit at 99–100%.

The mechanism is the locally adaptive score. `field_max` is
`|resid| / (σ + floor)`, and under an input shift both the residual *and* the
ensemble spread grow. σ grows fast enough to keep the ratio bounded, so the
interval covers — by being far too wide. A 100% coverage is a failed
calibration in the same way 60% is; the KPI band is two-sided for a reason.

Where it does under-cover, it under-covers hard, and the pattern is specific:
`advdiff_tau` 4.5%, `diffusion_tau` 26.4%, `advdiff_rough` 55.7%,
`diffusion_rough` 62.7%. The two time-evolution families, and only them. Their
spread does *not* track the error under a correlation-length shift, while the
elliptic families' does.

**Weighted conformal has a working window, and it is narrow.** This is what the
graded ladder was added for, and it paid for itself:

| shift | probe AUC | split | weighted |
|---|---|---|---|
| `poisson_dam0p1` | 0.99 | 100.0% | 84.0% |
| `darcy_dam0p1` | 0.98 | 100.0% | **90.8%** ✅ |
| `poisson_dam0p2` | 1.00 | 100.0% | **91.8%** ✅ |
| `poisson_dam0p3` | 1.00 | 100.0% | 97.7% |
| `poisson_dam1` | 1.00 | 100.0% | 100.0% (512/512 quantiles = ∞) |

Beyond Δα ≈ 0.2 the probe saturates at AUC 1.00, the density ratio has no
overlap left to exploit, and the **exact** weighted quantile correctly returns
+infinity for most test points — 512 of 512 at Δα = 1.0. The 100% coverage in
those rows is vacuous, which is visible only because the fix from turn 1 made
the quantile exact and the infinity count is reported. The old median-weight
shortcut would have returned a finite quantile and a clean-looking 100%.

**A methodological correction I am recording against my own table.**
`field_max` coverage is *not comparable across resolutions*: it is a maximum
over pixels, and 256² has 16× more of them, so the score rises mechanically. The
resolution rows (advdiff 20.7% at 256²) partly measure that, not miscalibration.
`norm_ratio` and `rel_l2` are resolution-comparable and should be read instead.

### Clause 2 — speedup ≥100×: missed, in four independent ways

| measurement | result |
|---|---|
| GPU, trained families, ensemble | **0/10 rows ≥100×**; range 0.007×–41× |
| four exact-propagator families | surrogate is **15–140× slower** than the solver |
| Darcy, best row (batch 1, 5 members) | 40.8× at rel-L2 0.0430 |
| Darcy, iso-accuracy | **3.4×** |
| ≥100× reachable at all? | only at M=1 (112×), which has no spread |

Four separate points, each of which alone sinks the clause:

1. **For four of the five families the reference solver is two FFTs.** Poisson
   takes 0.20 ms for a batch of 64; the 5-member surrogate takes 15 ms. A
   surrogate is the wrong tool for a problem with an exact spectral propagator,
   and the rows are kept in the table so that is visible rather than curated
   away. Only Darcy — an iterative solve — is a candidate at all.

2. **The uncertainty costs 4.6×.** Single member on Darcy at batch 1: 187.7×.
   Five members: 40.8×. The thing that makes this a *UQ* kit is exactly the
   thing that takes it below 100×.

3. **The headline denominator was over-converged.** The corpus was generated at
   PCG tol 1e-10. Sweeping the tolerance: at tol 1e-1 the solver still reaches
   rel-L2 8.5e-5 against a 1e-12 reference, in 75 ms against the surrogate's
   22 ms — **3.4×**. And that is an *upper bound on the honest ratio*, because
   the sweep never made the solver as inaccurate as the surrogate: the
   surrogate's 4.3% error is worse than a PCG stopped at a 10% residual
   tolerance. The preconditioner is too good for a surrogate to beat on
   accuracy-per-second. The same holds for Navier–Stokes: dt can be relaxed
   from 1e-3 to 8e-3 (694 ms → 86 ms) for rel-L2 < 1e-5, so its 76× headline is
   also ~8× over-converged.

4. **Batching favours the solver, not the surrogate.** Darcy PCG costs 370 ms
   for one sample and 283 ms for sixty-four — the batched solve is *cheaper per
   sample by 84×*. The surrogate scales linearly. So the speedup is a latency
   win (40.8× at B=1) that collapses to 12.7× at B=64. Quoting only batch-1
   would have been the flattering choice.

The ensemble-size ablation makes the structure explicit — 112.1× at M=1,
58.0× at M=2, 42.3× at M=5, and coverage/spread-correlation/shift-AUROC are all
`—` at M=1 because there is no spread to compute. **Every configuration that
clears 100× has no uncertainty; every configuration with uncertainty is below
it.** Timing noise between adjacent M is visible (M=3 measures faster than
M=2), so that is a trend on 5 checkpoints — a screen, not a verdict.

### Clause 3 — OOD AUROC ≥0.9: missed, and the miss is an identity

Best single detector on shift detection is `mahalanobis` at **43/49 shards
≥0.9**. The six misses are two mild shards (Δα = 0.1, AUROC 0.876 and 0.900,
where rel-L2 moves 0.0018→0.0020 — arguably not OOD at all) and four shards
that are at chance:

| shard | rel-L2 | spread | mahalanobis | residual |
|---|---:|---:|---:|---:|
| `biharmonic` | **41.25** | 0.486 | 0.497 | 0.495 |
| `frac_s3` | **1702.5** | 0.499 | 0.492 | 0.502 |
| `frac_s0p25` | 0.949 | 0.496 | 0.497 | 0.496 |
| `frac_s0p5` | 0.857 | 0.499 | 0.503 | 0.494 |

The surrogate is wrong by a factor of 40 to 1,700 and **nothing notices**.

**This is not a tuning gap, and the distinction matters.** All three detectors
are functions of the input and the *configured* operator. These four shards
inherit their parent's input distribution exactly by construction, and the
configured operator does not change when the process does. A model configured
as Poisson, handed `f`, returns the Poisson solution: its ensemble spread is
in-distribution because the input is, and its Poisson residual is *small and
correct*, because it is a good answer to a question nobody asked. Any statistic
of (input, configured operator) must sit at 0.5 here. Predicted in turn 1 before
the data existed; measured at 0.486–0.503 on three independent detectors.

The obvious alternative explanation — "the ensemble is too small / too
correlated" — is ruled out by the same table: those detectors reach 1.000 on
`darcy_c3` (rel-L2 0.77) and on `navier_stokes` (0.89), shifts of comparable
severity whose *inputs* do move. The blindness is specific to matched inputs.

Error detection is also short: `spread` 0.866 pooled, 0.796 on the population
where all three detectors are defined. Stratified by family it is 0.936–1.000
for helmholtz/diffusion/advdiff/darcy and **0.443 — below chance — for
poisson**, because the poisson stratum is where the four blind shards live.

### The constructive half: what does work, and what it costs

Since no unsupervised statistic can see an operator shift, the question becomes
how expensive the cheapest thing that can is. `scripts/eval_label_probe.py`:
run k samples through the reference solver, compare their errors to the
calibration distribution with a conformal p-value, combine with Fisher.
False-alarm rate verified on held-out in-distribution data at 0.046–0.059 for
k=1 (drifting to 0.089–0.105 at k=64 for helmholtz and darcy — the discreteness
of a 1,024-point conformal p-value, and stated rather than smoothed).

**k = 1 for every shard the unsupervised detectors could not see** —
`biharmonic`, `frac_s3`, `frac_s0p25`, `frac_s0p5`, and also `darcy_c3` and both
Navier–Stokes shards. One labelled probe, at 95% power and 5% false alarms.

And where k > 64 — every resolution shard, and the three `*_smooth` shards —
the surrogate's error did not move (0.0019 vs 0.0018; smoother inputs are
*easier*). The probe declines to alarm on a shift that does no harm, which is
precisely what separates it from `mahalanobis`, which fires at AUROC 1.000 on
all ten resolution shards where there is nothing wrong.

### Verdict

**A4 is UNREACHABLE as specified**, and the three reasons are structural rather
than budgetary:

1. `≥100×` and `a calibrated interval` are contested by one knob. M=1 gives
   112× and no interval; M=2 gives 58×.
2. Against a well-preconditioned iterative solver the honest iso-accuracy ratio
   is 3.4×, and the surrogate is less accurate than the solver's loosest
   setting. Against an exact spectral propagator it is 0.007–0.06×.
3. `AUROC ≥0.9` on operator shift with matched inputs is unachievable by any
   function of (input, configured operator), which is what an unsupervised
   detector is.

What would change each: (1) a surrogate with an intrinsic variance head rather
than an ensemble — one forward pass, one σ; (2) a problem whose solver is
genuinely expensive at the accuracy anyone needs (3D, fine meshes, stiff
chemistry) rather than a 64² periodic box; (3) a labelled-probe budget, which
this repo now measures at k=1.

### Second opinions

- **codex** (`logs/critic_codex.log`) reviewed the calibration and OOD code
  before any result existed and found seven real defects, all fixed in commit
  2 and all pinned by tests: the σ floor recomputed per shard in
  `eval_conformal.py` (the same bug turn 1 fixed in the API and I had wrongly
  recorded as fixed in both); weighted conformal clamping an unreachable
  quantile to the max score instead of +∞; the median-weight approximation
  (now exact and per-test-point); the probe standardizing before its own
  holdout split; `interval()` returning the pooled quantile in group mode;
  pixel coverage treated as 2.1M independent observations; and the
  error-detection threshold taken from the test errors it then scored. Two of
  these — the σ floor and the vacuous weighted quantile — would each have
  produced a *better-looking* clause-1 number.
- **agy** (`logs/critic_agy.log`) attacked the benchmark. Its live findings:
  the verdict line selected the maximum speedup across families (now a range
  and a count); the ensemble's rel-L2 was printed beside the single-member
  speedup (now per-variant, per-device); CPU rows carried a CUDA bf16 accuracy
  (now measured in fp32 on CPU). Its normalization/de-normalization findings
  were against a file already patched. Its point that the Darcy PCG does a
  synchronous device-to-host copy per iteration is correct and **not yet
  addressed** — it makes the solver slower than it needs to be and is a
  subsidy to the surrogate, on top of the 3.4× iso-accuracy figure.
- **cursor-agent** could not run: `Authentication required`. Not used.

---

## Turn 3 — 2026-09-09 — my own headline speedup was a measurement artefact

Following the weekend rule that says to measure run-to-run spread before
reporting a comparison, I repeated one timing configuration eight times
(`scripts/bench_noise.py`, `runs/bench_noise.json`). I had applied that rule to
seeds and not to wall-clock, and wall-clock was where it mattered.

| repeat | Darcy PCG tol 1e-10 | PCG tol 1e-1 | 5-member surrogate |
|---:|---:|---:|---:|
| 0 | 375.00 ms | 45.57 ms | 15.57 ms |
| 1 | 376.69 ms | 45.91 ms | 15.65 ms |
| 2 | 376.36 ms | 26.55 ms | **9.05 ms** |
| 3 | 219.10 ms | 26.44 ms | 8.85 ms |
| 4–7 | 217–218 ms | 26.5 ms | 8.9 ms |

An idle H100 boosts over the first several seconds of load. Everything drops by
~42% between repeat 1 and repeat 3 — both sides, so in steady state the ratio is
unaffected. The damage is the **transition**: at repeat 2 the surrogate had
already ramped and the solver had not, giving 376/9.05 = 41.6× for a comparison
whose steady-state value is 23.5×.

`bench_speedup.py` times each configuration in sequence with its own five-call
warmup. That warms the *kernel*, not the *clock*, and the whole table was
measured across the ramp. The 40.8× Darcy headline in the previous
`RESULTS.md` was measured in exactly the transition regime above.

Measured spread of the two ratios, before and after adding a 6-second dense
matmul ramp at the top of every benchmark script:

| ratio | before | after |
|---|---|---|
| Darcy headline (B=1, 5 members) | 24.08–41.61× (**±36%**) | 23.04–23.75× (**±1%**) |
| Darcy iso-accuracy | 2.12× and 3.36× on two invocations | 2.83–2.89× (**±1%**) |

So the corrected numbers are **23.5× headline and 2.85× iso-accuracy**, and
every timing table has been regenerated. Both are further below the 100× clause
than the figures I recorded in turn 2, so the verdict does not change — but the
turn-2 entry overstated the surrogate's case by 74% on its own best row, and
that is exactly the direction of error the weekend rules exist to catch. The
36% swing was larger than several of the differences turn 2 discussed as if
they were findings; on the old numbers, the M=1 reading of 112.1× that the
ensemble ablation used to argue "100× is reachable without uncertainty" is
itself inside the ramp regime and had to be re-measured.

The remaining `agy` finding is also now closed with a number rather than an
argument: amortizing the PCG's per-iteration device-to-host convergence sync
(`check_every`) takes the solver from 290.2 ms to 257.1 ms, an **11.4%**
subsidy to any surrogate timed against the default. `check_every=1` is what
generated the corpus and remains what `bench_speedup.py` times, so the headline
is unchanged and the size of the subsidy is now in `runs/isoaccuracy.json`.

**What I should have done first.** The brief's seed-count lesson is written
about seeds, and I read it as being about seeds. It is about *any* comparison
whose noise floor is unmeasured. A timing harness is a measurement instrument
and its repeatability is a property to be measured, not assumed, particularly
when the instrument reports a ratio of two quantities that drift together.

---

## Turn 4 — 2026-09-10 — A4 REOPENED at rung 2. Hypothesis H5, written before the run

The board reopened this project with a named route, and the route is right about
the *structure* of my failure. Restating it in my own words so that I am
attacking the thing and not a paraphrase of it:

`runs/conformal.json` records `n_members: 5`. Every interval in this repository
is `pred ± q̂·σ` where `σ` is the **sample standard deviation across ensemble
members**. That single design decision — mine, made in turn 1 because a deep
ensemble was what `pde-neural-operator` already had — is what makes two of the
three clauses mutually exclusive:

- an interval needs `σ`, so it needs M ≥ 2 (`uqkit/ensemble.py` raises on M=1);
- `runs/members.json` measures M=1 at **108.5×** and M=2 at **58.5×**.

So the speedup clause was being evaluated against a model that pays M forward
passes for a quantity that one forward pass can also produce. The 0/10 rows
≥100× is a fact about deep ensembles, not about surrogate inference.

### H5 (coverage + speedup, one change)

**Hypothesis.** Replacing the ensemble spread with a *single network's*
uncertainty head keeps in-distribution `field_max` coverage inside [88, 92] and
puts the coverage row and the ≥100× row on the same model, because the head
costs three extra output channels in the final 1×1 projection and nothing else.

**The change, and only this change.** `FNO2d` already takes `out_ch`. Train it
with `out_ch=4`: channel 0 the mean, channel 1 `log σ`, channels 2–3 the two
residual quantiles. Two σ sources come out of it and both get measured:

| arm | σ | trained by |
|---|---|---|
| `ens` | sample std over 5 members (existing) | — (baseline, already measured) |
| `het` | `exp(log σ)` from channel 1 | Gaussian NLL on the **detached** residual |
| `cqr` | `(q_hi − q_lo)/2` from channels 2–3 | pinball loss at 0.05 / 0.95 on the **detached** residual |

The mean's loss is `rel_l2`, byte-identical to `scripts/train_member.py`. The
σ and quantile losses take `pred.detach()`, so **the mean's gradient is exactly
the baseline's**. That is deliberate: if the mean degrades, the coverage and the
speedup are no longer being compared against the same predictor and the whole
comparison is contaminated. Mean rel-L2 against the M=1 baseline
(0.00307, range 0.00284–0.00340 over 5 subsets) is a *check on the experiment*,
not a result.

Everything downstream is untouched — the same `SCORES`, the same
`conformal_quantile` with its `(n+1)` correction, the same frozen calibration
σ-floor, the same `GroupConformal` / `WeightedConformal`, the same shift suite.
The only edit to the eval path is where `σ` comes from. If a number moves, it
moved because of the head.

**Protocol, fixed now, before any of it runs.**

- α = 0.1; target 90%; KPI band [88, 92]. `field_max` is the headline as before.
- **8 seeds per arm.** The seed-count lesson applies: 5 checkpoints could only
  give C(5,M) subset spreads and I labelled that "screen, not verdict". A
  clause verdict needs 8, so `het` and `cqr` get 8 seeds each and the
  in-distribution coverage clause is judged on the 8-seed spread, not a point
  estimate.
- Speedup is measured on the **same rows and the same denominators** as
  `runs/members.json` and `runs/bench.json`, with the 6-second clock ramp that
  turn 3 showed is mandatory. No new denominator is introduced. The M=1 row's
  solver is the Darcy-branch PCG at tol 1e-10 (`poisson_dam1`, 0.2133 s).
- **The strict readings stay.** Landing ≥100× on the Darcy-PCG row does not
  retire the two facts that this surrogate is 15–220× *slower* than the exact
  spectral propagators and 2.2× iso-accuracy against the cheapest solver
  setting swept. Those rows are reported next to it, and the clause is declared
  per row with its denominator named. Rung 1 of the ladder is "report both
  readings", not "pick the kind one".
- Cost accounting, stated in advance so it cannot be adjusted afterwards: the
  interval's inference cost is the one forward pass that emits mean and σ.
  Conformal calibration is offline on the `cal` split, one quantile per score,
  exactly as it is for the ensemble — it is not in the per-sample path for
  either arm, so this is not a subsidy to the new one.

**What would falsify H5.**

1. In-distribution `field_max` coverage outside [88, 92] on the 8-seed mean →
   the head cannot calibrate and rung 2 has failed for the coverage clause.
2. The 4-channel forward pass measuring under 100× on the row where M=1 read
   108.5× → the head is not free and the two clauses stay incompatible.
3. Mean rel-L2 moving materially from the M=1 baseline → the arms are not
   comparable; that is a bug in my experiment, to be fixed before any number
   from it is reported.

**The obvious alternative explanation, and how the run distinguishes it.** If
`het` covers, the cheap story is "any σ calibrates, because split conformal
rescales whatever you give it by q̂". That is *true for marginal coverage and
false for the interval's usefulness*, and the two are separable with numbers I
already collect: a constant σ also reaches 90% marginal coverage while its
interval is the same width everywhere. So the run reports, beside coverage:
`field_max` **interval width** and the **spread–error correlation** for `het`,
`cqr`, `ens`, and a deliberately useless `const` σ arm. If `het`'s correlation
is near the ensemble's +0.91 the head carries information; if it is near
`const`'s ~0 the coverage is a rescaling artefact and I will say so.

### H6 (OOD) — deferred to the next turn, but named now so it is not invented after the fact

The identity argument in turn 2 is airtight *for the deployment it models*, and
I want to be precise about why, because the board's suggested escape does not
apply to it as written. `uqkit/sims/pde2d.py:458` fixes the framing:
`PARENT["biharmonic"] = "poisson"` — the operator-shift shards model **"the
process changed and nobody reconfigured the model."** The configured operator is
still Poisson. `scripts/eval_ood.py` computes the residual under
`PARENT.get(task)`, i.e. under Poisson, and the surrogate's output *is* a
Poisson solution of the same in-distribution `f`, so the residual is small and
the detector is at chance by construction. Applying "the configured operator"
changes nothing, because the configured operator is the parent. That is why all
three detectors read 0.486–0.503, and the number is not a defect.

But there is a *second* deployment failure with the same shards and a different
answer, and I conflated the two by only measuring one:

- **(A) not reconfigured** — the request still says Poisson; the physics moved.
  Observable at test time: `(f, poisson)`. Identical in distribution to
  in-distribution samples. **Provably undetectable** unsupervised; needs a
  label, which is what `runs/label_probe.json` prices at k=1.
- **(B) reconfigured** — the request says *biharmonic*, an operator this
  surrogate was never trained on. Observable at test time: `(f, biharmonic)`.
  The configured operator is now genuinely different information, and
  `L_biharmonic û − f` is large because `û` is not a biharmonic solution. This
  is the case where the board's solver-consistency residual works, and it is
  the case a *kit* meets most often: the operator is part of the request.

H6 is that (B) is detectable at AUROC ≥ 0.9 on the operator-shift shards where
the residual has headroom over its own floor, and that it is *not* detectable
where the floor swamps it — `measure_residual_floor.py` already says the
biharmonic floor is 5.3e-2 and `frac_s3` is 68, so I am predicting the
detector's own failure boundary before measuring it, which is the part that
makes it a hypothesis rather than a demonstration. Both presentations get
reported for every shard, labelled (A) and (B), and (A) stays in the table as
the strict reading.

---

## Turn 4 (2026-09-10) — H6 made concrete, and the trap in it named first

`train_single_uq.sh` (H5, 8 seeds) is running on GPU 3; this turn does not touch
it. It works H6 instead, on the checkpoints that already exist.

### What the shard inventory actually says

I enumerated the 39 OOD shards and compared `FAMILY_BASE[task]` and `CFG[task]`
against the parent's. **Six shards change the operator**; 33 change only the
input distribution or a family parameter that the operator's *form* does not
depend on:

| shard | requested operator | algebraic apply? |
|---|---|---|
| `biharmonic` | \|k\|^4 | yes |
| `frac_s0p25` | \|k\|^0.5 | yes |
| `frac_s0p5` | \|k\|^1 | yes |
| `frac_s3` | \|k\|^6 | yes |
| `navier_stokes` | nonlinear, time-evolved | **no** |
| `ns_T0p25` | nonlinear, time-evolved | **no** |

So presentation (B) can only differ from (A) on those six, and can only be
*computed* on four. On the other 33 shards `L_requested ≡ L_parent` and (B) is
byte-identical to (A) — that is a control, not a disappointment, and it goes in
the table so that nothing credits the new detector with the old one's work.

### The trap, which I want on the record before the number exists

`residual_score = ||L û − f|| / ||f||` is **not comparable across operators**.
Applying \|k\|^6 to anything at 64² produces a large number whether or not the
prediction is any good — `measure_residual_floor.py` already measured that
floor at **68** for `frac_s3`, i.e. larger than the right-hand side itself. A
detector that pools in-distribution scores (Poisson request) against `frac_s3`
scores (sixth-order request) will separate them at AUROC ≈ 1 **because of the
operator's conditioning, not because the prediction is wrong**. That would be a
number I could put in RESULTS.md and it would be worthless.

Two things guard against it, both defined now:

1. **The score is made dimensionless.**
   `consistency(û, a; L) = ||L û − f|| / (||L û|| + ||f||)`, bounded in [0, 1],
   zero iff the prediction satisfies the requested equation. Uses only the
   input, the prediction and the requested operator — legal at deployment.
2. **An offline diagnostic that the detector never sees**: the same score on
   that shard's *ground truth*, `floor_c = consistency(u_true, a; L)`, and
   `headroom = median c(û) / median c(u_true)`. If `floor_c` is already near 1
   the operator is unconditioned in fp64 and a high AUROC on that shard is an
   operator-identity signal, which I will label as such rather than bank.

### The baseline that has to be beaten, and probably is not

If the request names the operator, then `known_operator` — a dict lookup,
`1 if task not in PRETRAIN_TASKS else 0` — scores **AUROC 1.000 on all six
operator-shift shards at zero cost**, and 0.5 on the other 33. Any credit the
consistency residual claims on `biharmonic` or the `frac_*` shards has to be
credit over *that*, not over chance. Where the residual earns its keep, if
anywhere, is that it is continuous and defined for an operator whose *name* is
familiar and whose *parameters* are not. The inventory above says this corpus
has exactly one such shard, `darcy_c3` (contrast 3.0 vs 1.5), and its operator
form is unchanged, so I expect the residual to buy nothing there either. I am
writing that expectation down so that if it does buy something I have to
explain why rather than accept it.

### Predictions, registered before the run

- `biharmonic`: AUROC ≥ 0.99, `headroom` ≫ 1 → genuine detection.
- `frac_s0p25`, `frac_s0p5`: symbols *lower* order than Poisson, so the floor is
  below Poisson's 6.2e-5 and the signal is O(1) → AUROC ≥ 0.95, headroom ≫ 1.
- `frac_s3`: AUROC ≈ 1.0 **and `floor_c` ≈ 1 with headroom ≈ 1** → the number is
  real and the detection is not. Predicted failure of the *interpretation*, not
  of the metric, which is the part that makes this a hypothesis.
- `navier_stokes`, `ns_T0p25`: `[not measured]`, no cheap apply exists.
- All 33 input-shift shards: (B) ≡ (A), no change from `runs/ood.json`.
- Combined `max(z_maha, z_consistency)`, z-standardised on the **in-distribution
  `cal` split only**: ≥ 0.9 on more shards than either alone. If it is not, the
  combination is not worth its complexity and I will drop it.

### What this can and cannot settle for the clause

At best it converts the four algebraically-checkable operator-shift shards from
0.486–0.503 to near 1, leaves the two NS shards unmeasurable, and leaves the
`mahalanobis` 43/49 elsewhere unchanged. That is a clause-relevant improvement
on 4 shards of 49 and it does **not** retire the (A) identity argument, which
stays the strict reading. Rung 1 of the ladder is "report both", and (A) is the
deployment where nobody reconfigured the model.

### Rung 4, asked the right way this time (`codex`, `logs/critic_codex_howto_speed.log`)

The board's addendum is right that I had only ever asked adversaries "what is
wrong with this", which is why they only ever returned defects. Asked instead
"how would you make the ≥100× clause pass honestly", `codex` returned three
things, and one of them is a defect in *this turn's plan* that I had not seen.

> "**108.5× currently belongs to the member ablation** (`runs/members.json:31`)
> … Fill those values from the shipped checkpoint."

**This is correct and it is load-bearing.** The 108.5× M=1 row was measured on
an *ensemble member* — a 1-channel `FNO2d`. The model that would ship is
`FNO2dUQ`, 4 channels through an extra 1×1 projection. H5 asserts that head
"costs three output channels and one extra 1×1 projection and nothing else",
and I wrote that as though it were a fact. It is a hypothesis, it is H5's own
falsifier #2, and until `bench_speedup.py` is run **on a `runs/u*/best.pt`**
the 108.5× may not be carried into any sentence about the shipped model. Noted
and queued; the number in this repo stays attached to the ablation until then.

> "a numerical-methods referee will likely require iso-accuracy for an
> unqualified '100× acceleration' headline, but can accept an explicitly
> qualified fixed-reference claim … **remeasure it for the new network**."

Agreed, and it is rung 1 stated by someone else: both readings, each with its
denominator named, neither substituted for the other. The 2.15× iso-accuracy
figure in `runs/isoaccuracy.json` is also an ensemble measurement and is also
not transferable; it gets re-run on the shipped checkpoint or it is `[not
measured]` for that checkpoint.

> "Require the speedup's **lower 95% bound ≥ 100×**."

Taken. That is strictly harder than what the clause asks and it costs nothing,
so there is no reason to use the point estimate.

> "For exact FFT families: **no**, there is no defensible positive
> inference-speedup claim when the surrogate is slower. Report ratios below one."

Which is what `runs/bench.json` already does and what the 15–220×-slower row in
RESULTS.md says. Confirmed rather than new, and I am not going to dress it up.

### H6 result — measured, `runs/consistency_M1.json`

Every prediction registered above survived, including the one about the
detector's own failure boundary:

| shard | maha (A) | cons (A) | cons (B) | floor_c | headroom | reading |
|---|---|---|---|---|---|---|
| `biharmonic` | 0.497 | 0.520 | **1.000** | 0.0216 | 46.3 | genuine |
| `frac_s0p25` | 0.497 | 0.511 | **1.000** | 2.1e-7 | 4.3e6 | genuine |
| `frac_s0p5` | 0.503 | 0.503 | **1.000** | 9.1e-7 | 8.6e5 | genuine |
| `frac_s3` | 0.492 | 0.510 | 1.000 | **0.981** | **1.02** | **artefact** |
| `navier_stokes` | 1.000 | — | — | — | — | no cheap apply |
| `ns_T0p25` | 1.000 | — | — | — | — | no cheap apply |

and all 33 non-operator shards returned `cons_B` byte-identical to `cons_A`,
which is the control behaving.

`frac_s3` is the row worth the turn. Its AUROC is 1.000 and it is **not
detection**: the exact solution of that shard scores 0.981 on the same detector,
so the separation is the sixth-order operator being round-off dominated in fp64,
not the prediction being wrong. Without `floor_c` I would have banked 4/4 and
one of them would have been false. `measure_residual_floor.py` predicted it
(floor 68, above the right-hand side) and the prediction was written down before
the run.

**What this does not buy.** On all six operator-shift shards a *dict lookup* —
is the requested operator one we trained on — scores AUROC 1.000 for free. So
the consistency residual does not beat the baseline on separation and I will not
claim it does. What it adds over the lookup is (a) a severity rather than a bit,
and (b) `floor_c`, which is the only thing here that could tell me `frac_s3`'s
1.000 was worthless. That is a smaller claim than "the residual detects operator
shift" and it is the one the numbers support.

**Clause arithmetic.** `combo = max(z_maha, z_cons_B)` reaches **47/49** shards
≥ 0.9 against `mahalanobis`'s 43/49 under (A). The two survivors are
`poisson_dam0p1` (0.888) and `darcy_dam0p1` (0.841), the weakest rung of the
graded ladder — and on exactly those two rows `combo` is *worse* than
`mahalanobis` alone (0.900, 0.876), because taking a maximum over z-scores pays
for a second, noisier component. That is a cost of the combination and it is
reported next to its benefit, not omitted from it.

### Rung 4 on clause 1 (`agy`, `logs/critic_agy_howto_coverage.log`) — and two corrections to my own reporting that came out of checking it

I asked `agy` how to make coverage land in [88, 92] on shifted shards. Its
diagnosis was half right, and checking the half that was wrong turned up two
things this repo has been reporting incorrectly.

**What `agy` got wrong.** It said the over-coverage "stems directly from the
calibrators", pointing at the pooled `q_split = 524.73` swamping families whose
own quantiles are 4.90 (helmholtz) to 29.22 (darcy). The arithmetic is right —
those numbers are in `runs/conformal.json` — but the conclusion is not new: the
class-conditional `GroupConformal` that fixes exactly this **already exists and
is already what the in-distribution headline uses** (`pooled_group`, 88.6%).

**Correction 1, and it reverses the direction of the failure.** For OOD shards
this repo reported the `split` and `weighted` columns and **never reported the
`group` column at all** — while headlining `group` in distribution. Two
different calibrators either side of the comparison. Measured now from the same
`runs/conformal.json`, over the 42 shards whose covariate-shift assumption
holds:

| calibrator | n | in band | over-covers | under-covers | median |
|---|---|---|---|---|---|
| `split` (pooled) | 42 | 0 | 32 | 10 | 1.0000 |
| `group` (class-conditional, **the in-distribution headline's calibrator**) | 42 | **6** | 3 | **33** | **0.7012** |
| `weighted` | 32 | 2 | 29 | 1 | 1.0000 |

So the sentence this repo and the board have been carrying — *"the dominant
failure is over-coverage, 24/32 at 99–100%"* — is true of `split` and **false
of the calibrator the in-distribution number uses**, under which the dominant
failure is **under-coverage, 33/42**. I reported the harmless failure mode and
omitted the harmful one. Over-coverage is wasteful; under-coverage at a median
of 70% against a 90% target is an interval that lies. That correction goes to
the board.

**Correction 2: the `weighted` 100%s are not coverage, they are abstention.**
`agy` was right here and I had the diagnostic in hand without reading it.
`probe_auc = 1.000` on every shard — the calibration and test inputs are
completely separable in spectral-feature space — so the weighted quantile is
`+∞`: **30 of 32 shards have at least one infinite quantile and 14 of 32 are
infinite for all 512 test points.** An infinite interval covers everything.
Quoting "weighted conformal reaches 100% coverage" without that is quoting the
width-∞ degenerate case as a success. Every coverage number in this repo needs
its width and its `n_infinite_quantiles` printed next to it, and the report
will not emit one without them.

Neither correction was found by an adversary. `agy` pointed at the calibrator;
the corrections came from checking whether it was right.

**What `agy` got right and I am taking.**

- *Two-sided 90 ± 2% under arbitrary covariate shift without labels is provably
  unattainable.* Weighted conformal gives one-sided conservative validity only,
  and under non-overlapping support distribution-free validity requires infinite
  intervals — which is precisely the ∞ above, so the theory and my measurement
  agree. This is a statement to make in the report, not a defeat to hide.
- *The smallest label budget that makes it exact is k = 9 per shard*, because
  `⌈(n+1)(1−α)⌉ ≤ n` needs `n ≥ ⌈(1−α)/α⌉ = 9` at α = 0.1. That is a clean
  constructive complement and it sits beside the k = 1 detection probe already
  in `runs/label_probe.json`: **one label to notice the shift, nine to
  re-certify the interval.** I like this a great deal more than the clause it
  replaces, and it will be reported as a different claim, not as the KPI.
- *The experiment that separates "the weights are bad" from "the score is bad"*:
  split each shifted shard 50/50, calibrate split-conformal on the target half,
  test on the other half. If coverage lands in band, the score is sound and the
  failure is entirely importance weighting. It uses labels, so it is an **oracle
  upper bound and a diagnostic, never a deployable method**, and it will be
  labelled that way at the point the number appears.

### H7 (clause 1), written before it runs

**Hypothesis.** The `field_max` score is sound under shift and the entire
coverage failure is calibration transfer. Falsified if oracle target-split
calibration does *not* land in [88, 92] on a majority of the 42 shards — which
would mean the score distribution itself is pathological under shift and no
amount of reweighting fixes it.

**Prediction, registered now.** Oracle target-split lands in band on ≥ 38/42
shards (it is split conformal on exchangeable data, so it should be nearly
free), and the k = 9 curve reaches the band by k = 9 with the finite-sample
guarantee, not before. If oracle target-split fails, that is the more
interesting result and it kills the score, not the calibrator.

---

## Turn 7 (2026-09-10, ~12:30) — the M=1 UQ network is measured, and the 100× clause is now a launch-overhead problem

### What the numbers are

The addendum's rung-2 route was run last turn and never aggregated or
committed. It is aggregated now. Eight seeds, three σ heads, one forward pass
per interval (`runs/uq_seeds.json`, `runs/conf_u*_{het,cqr,const}.json`):

| arm | fwd/interval | cov mean | sd | range | seeds in [88,92] | `sharpness_rel` |
|---|---|---|---|---|---|---|
| `het` | **1** | 0.9026 | 0.00396 | 0.01367 | **8/8** | **0.0784** |
| `cqr` | **1** | 0.9012 | 0.00352 | 0.01074 | **8/8** | 0.0795 |
| `const` (control) | 1 | 0.8977 | 0.00247 | 0.00762 | 8/8 | 0.0902 |

So the addendum's structural claim is confirmed: the coverage row and the
one-forward-pass row are now **the same row**. M>1 is no longer forced.

The control matters more than the two candidates. Split conformal rescales
*any* σ to ~90% marginal coverage, so `const` — a σ that is literally a
constant — also lands 8/8 in band. **Coverage in band at M=1 is therefore not
evidence that the head learned anything.** The only thing that separates the
heads is width: `het` is 13.1% sharper than `const` (0.0784 vs 0.0902). That is
the real effect of the σ head and it is a much smaller claim than "the
single-network interval works". Reported as such.

### The speedup, and the row that must not be quoted

`runs/bench_uq.json` (GPU 3, `torch_num_threads=96`, tf32 off, bf16 autocast,
20 iters, clock-ramp warmup):

| family | b | solver | solver accuracy | `uq_single` | `+residual` | surrogate rel-L2 |
|---|---|---|---|---|---|---|
| darcy | 1 | 265.1 ms | PCG fp64, tol 1e-10 | **71.7×** | 63.7× | 0.0506 |
| darcy | 64 | 316.4 ms | PCG fp64, tol 1e-10 | 64.8× | 63.0× | 0.0506 |
| poisson/helmholtz/diffusion/advdiff | 1, 64 | 0.10–0.15 ms | exact spectral, ~1e-7 | **0.02–0.07×** | — | 0.0025–0.0034 |
| navier_stokes | 1 | 796.8 ms | RK4 1000 steps | ~366× | — | **`[not measured]`** |

**The navier_stokes row is not a result and I am not counting it.** `trained:
False` — no surrogate was ever trained on that family, so its 366× is an
untrained network timed against a real solver. The repo already labels these
`*(untrained)*` with `[not measured]` accuracy in `RESULTS.md:143`, which is
why it did not become a headline. It is the single most seductive number in the
file and it is worth nothing. Best row *with* an accuracy is **71.7× at 5.06%
rel-L2**, and the honest strict reading against a solver tuned to the
surrogate's own accuracy is still the 2.2× in `runs/isoaccuracy.json`.

So the addendum's premise was half right. Putting the interval in one network
did remove the structural blocker, but the 108.5× it cited came from a
1-channel ensemble *member*; the actual UQ network with its extra heads reads
**71.7×**, not 108.5×. Clause 2 is short by 1.4×, not met by construction.

### Which part is the binding constraint — from numbers already on disk

Darcy, batch 1: surrogate 3.699 ms. Darcy, batch 64: surrogate 4.885 ms.
**64× the arithmetic for 1.32× the wall clock.** The four spectral families sit
at 2.22–2.30 ms at b=1 regardless of what they compute. A network whose time is
almost independent of its own workload is not compute-bound; at b=1 it is
paying per-kernel launch latency, and an FNO with this many layers launches
enough kernels to account for ~2 ms of it.

That reframes clause 2. The gap to 100× is not "the network is too big" and not
"the solver is too fast" — on darcy the solver takes 265 ms, three orders of
magnitude more. It is that **at deployment batch 1 we are timing the CPU-side
launch of the surrogate, not the surrogate.** This is a design decision in how
the model is executed, which puts it squarely at rung 2.

What distinguishes this explanation from the obvious alternative ("the model is
just slow"): if the model were compute-bound, b=64 would cost ~64× b=1. It
costs 1.32×. And if it were dominated by the normalization/de-normalization
inside the timed region, the cost would scale with tensor size, which across
darcy-vs-poisson at fixed b=1 it does not (3.70 vs 2.30 ms is a 1.6× spread
over families whose fields differ far more than that).

Second, smaller asymmetry found while reading the harness: the timed surrogate
closures in `scripts/bench_speedup.py` run **with autograd tracking on** —
neither `make_uq_fn` nor `make_surrogate_fn` is wrapped in `no_grad` — while
`PDE2DSimulator.solve` carries `@torch.no_grad()` (`uqkit/sims/pde2d_sim.py:42`).
Every speedup number this repo has ever published charges the surrogate for
building an autograd graph nobody uses and does not charge the solver for it.
That biases *against* our own claim, so it is not a cheat, but it is an unfair
comparison in the direction that happens to be safe, and it should be measured
rather than left as a silent margin.

### H8, written before it runs

**Hypothesis.** The batch-1 surrogate time is dominated by kernel-launch and
autograd-bookkeeping overhead, not arithmetic. Removing both — `no_grad` plus
CUDA-graph capture and replay of the identical weights — takes the darcy b=1
row from 71.7× past 100× **with the rel-L2 unchanged**, because a graph replay
executes the same kernels on the same tensors.

**Registered predictions.**
1. `uq_single+graph` at darcy b=1 drops to **≤ 1.5 ms** (from 3.699 ms), giving
   **≥ 170×**.
2. The b=64 row moves by **< 1.3×**, because it is compute-bound. If b=64
   improves as much as b=1 in relative terms, my launch-bound diagnosis is
   wrong and the gain is something else (kernel fusion, allocator) — I would
   have to say so.
3. `uq_single+nograd` (eager, no graph) captures only part of it: I predict it
   lands between 2.6 and 3.4 ms at darcy b=1, i.e. it explains less than half
   the gap. If `nograd` alone reaches ≤1.5 ms then the overhead was autograd,
   not launch, and prediction 1's mechanism is misattributed even if the number
   passes.
4. The four spectral families stay **below 1×** no matter what. Launch overhead
   is not why a network loses to an exact propagator that costs 0.1 ms, and no
   amount of graph capture will change that. Clause 2 will remain a
   family-by-family statement, never an average.

**Falsified if** the graph replay changes rel-L2 at all beyond bf16
non-determinism, in which case it is a different model and the row is void.
That check is a hard assert in the runner, not an eyeball.

**What this does not claim.** CUDA-graph capture is a real deployment
technique, same weights, same hardware, same accuracy, same solver
denominator — so it is rung 2, not a loosened protocol. But it is also not a
free lunch to be quoted alone: the eager row stays in the table as the strict
reading, exactly as the 1.47× row stayed in E4. If the graph row passes 100×
and the eager row does not, **both go in the table and the clause is reported
as "met under graph capture, 71.7× eager"**, with the two protocols named. A
clause met only under a stated execution mode is met under that mode and
nowhere else.

### H8 result — all four predictions held, and the clause moved. Then I found what makes the number unfair.

`runs/bench_uq_exec.json`, GPU 3, same shard, same solver, same weights; only
kernel dispatch differs. Darcy, the only trained family where any surrogate
beats its solver:

| arm | b=1 surrogate | b=1 ratio | b=64 surrogate | b=64 ratio | rel-L2 | graph vs eager |
|---|---|---|---|---|---|---|
| `uq_single` (eager, grad on — the published protocol) | 3.652 ms | 105.8× | 4.878 ms | 63.9× | 0.0506 | — |
| `uq_single+nograd` | 2.955 ms | 130.8× | 4.828 ms | 64.6× | 0.0506 | — |
| `uq_single+graph` | **0.642 ms** | **602.2×** | 4.495 ms | 69.4× | 0.0506 | **0.0 (bit-exact)** |
| `uq_single+residual+graph` | 0.727 ms | 531.7× | 4.588 ms | 68.0× | 0.0506 | **0.0** |

Predictions, scored against what I wrote before the run:

1. graph ≤ 1.5 ms → **0.642 ms**. Held.
2. b=64 moves < 1.3× → **1.085×** (4.878 → 4.495 ms). Held, and it is the
   prediction that matters, because it is what makes the diagnosis *launch
   overhead* rather than "the code got faster somehow".
3. `nograd` alone lands 2.6–3.4 ms → **2.955 ms**. Held. So autograd
   bookkeeping was 19% of the batch-1 cost and dispatch was the remaining 82%.
   The mechanism is attributed correctly and not just the number.
4. Spectral families stay below 1× → best is **0.3×**. Held. Graph capture
   makes a network that loses to an exact 0.1 ms propagator lose by less. It
   does not make it win, and no averaging over families will be done.

The bit-exact `graph_vs_eager_rel_dev = 0.0` is the part I care about most: the
replay is not an approximation of the model, it is the same kernels on the same
tensors, so no accuracy was traded for the time.

**But the denominator is not stable, and I nearly published on it.** The same
darcy b=1 solver row read **265.1 ms** last run and **386.5 ms** this run — same
GPU, same shard, same code, one hour apart. That 46% swing alone moved the
*eager* arm from 71.7× to 105.8×, i.e. across the KPI threshold, with the
surrogate untouched. Had I run only the second bench I would have reported
"eager M=1 passes 100×" and it would have been the clock-ramp artefact all over
again. `scripts/probe_solver_repeat.py` is now running 12 repeated trials to put
an interval on that denominator, and no darcy b=1 number goes in a document
without it.

**And here is the thing that actually decides this clause.** The darcy solver
costs 386 ms at b=1 and 312 ms at b=64 — 64× the systems for *less* wall clock.
The reference solver is launch- and sync-bound at batch 1 for exactly the same
reason my surrogate was. `solve_darcy` (`uqkit/sims/pde2d.py:215`) defaults to
`check_every=1`: a `.max()` compared in Python, forcing a device-to-host
synchronization on **every one of up to 2000 PCG iterations**.

So the 602× is my dispatch fix measured against the solver's *un*-fixed
dispatch. That is a timing game, and it is the one thing the brief says I may
never do. An earlier adversarial review already found this and the parameter
exists — `check_every` is plumbed, and `bench_isoaccuracy.py` times
`check_every=10` — but `bench_speedup.py`, which produces every headline
speedup in this repo, still times `check_every=1`. The subsidy was documented
and then left in the numerator's favour.

### H9, written before it runs

**Hypothesis.** Most of the reference solver's batch-1 cost is the same
per-iteration synchronization I just removed from my own side. Amortizing the
convergence test (`check_every` ∈ {1, 10, 50, 100}) cuts the darcy b=1 solver
time substantially at **equal or better** accuracy — the check can only fire
late, never early, so the returned iterate is at least as converged. The fair
denominator is the fastest solver setting that still meets the stated
tolerance, and against *that* denominator the 602× falls.

**Registered predictions.**
1. Darcy b=1 solver at `check_every=50` drops **below 120 ms** (from 386 ms).
2. The achieved residual at every `check_every` stays **≤ 1e-10**, the stated
   tolerance. If it does not, the faster setting is not the same solver and is
   inadmissible as a denominator.
3. The fair graph-arm ratio at darcy b=1 lands in **80×–250×** — so I do *not*
   know whether clause 2 survives this, and that is the point of running it.
   If it lands under 100×, clause 2 is not met at b=1 either and the correct
   report is that the only readings above 100× were subsidized ones.
4. b=64 barely moves (< 1.2×), because at b=64 the per-iteration sync is
   amortized over 64 systems already. If b=64 *does* move a lot, my sync
   diagnosis is wrong and the cost is iteration count, not synchronization.

**Commitment made before seeing the number.** Whatever this returns becomes the
headline denominator, including if it takes the clause from met to not met. The
`check_every=1` rows stay in the table labelled as the corpus-generating
configuration, because that is what the data was made with — but they are not
the denominator of a speedup claim any more.

### H9 result — prediction 1 falsified, prediction 3 landed, and the clause now has an honest bracket

`runs/bench_fair.json` and `runs/solver_repeat.json`, GPU 3, darcy, surrogate
rel-L2 0.0506, one forward pass per interval.

**Prediction 1 was wrong.** I predicted the b=1 solver would drop below 120 ms
once the convergence test was amortized. It went 214.85 → 189.81 ms at
`check_every=50`: **1.13×, not 3×**. So the reference solver's batch-1 cost is
*not* dominated by the per-iteration device-to-host sync. My diagnosis was
right that the solver is dispatch-bound and wrong about which part of the
dispatch: amortizing the check removes one sync per iteration but leaves every
PCG iteration's FFTs and elementwise kernels launching individually. The sync
was a rounding error next to the launches. Withdrawn.

**Prediction 2 held.** Every `check_every` stays admissible: residual 9.02e-11,
6.83e-11, 6.83e-11, 1.42e-11 against `tol` 1e-10, and the larger strides return
a *more* converged iterate as the semantics require. The fair denominator is
therefore `check_every=50` at 189.12 ms (min of 5 trials), and it is a legal
configuration of the same solver, not a different one.

**Prediction 4 held.** b=64 solver moved 282.64 → 252.19 ms, 1.12×.

**Prediction 3 held, and it is the one that mattered** — I wrote "80×–250×, so I
do not know whether clause 2 survives", and here is what survived:

| batch | arm | fair denominator (`check_every=50`) | unfair (`check_every=1`) | ≥100×? |
|---|---|---|---|---|
| 1 | `graph` | **296.1× conservative / 297.6× median** | 336.9× | ✅ |
| 1 | `nograd` | 113.5× / 115.2× | 130.4× | ✅ |
| 1 | `eager` (published protocol) | **89.2× / 92.9×** | **105.1×** | ❌ |
| 64 | `graph` | 51.1× / 53.5× | 60.0× | ❌ |
| 64 | `nograd` | 51.6× / 51.9× | 58.2× | ❌ |
| 64 | `eager` | 51.3× / 51.5× | 57.7× | ❌ |

*Conservative* = fastest admissible solver trial ÷ slowest surrogate trial.

**Look at the eager b=1 row.** Against the subsidized `check_every=1`
denominator it reads 105.1× and clears the KPI. Against the fair one it reads
89.2× and does not. The solver subsidy an earlier reviewer flagged, and which
this repo documented and then left in place, was **exactly the difference
between passing and failing clause 2 on the eager arm.** That is the single
most important number produced this turn and it is an argument against my own
result.

### The strictest reading, which I have to state because it is the one that hurts

The b=1 solver costs 214.85 ms for one system; the b=64 solver costs 282.64 ms
for sixty-four. Sixty-four times the arithmetic for 1.32× the wall clock — the
reference solver is as dispatch-starved at batch 1 as my surrogate was before
H8, and **I graph-captured only my side.** I tried the solver's counterpart
optimization and it bought 1.13%; capturing the PCG loop itself is obstructed
by its data-dependent break, which I have not solved.

So the bound has to be said out loud: if the solver achieved the same
per-sample efficiency at batch 1 that it demonstrably achieves at batch 64
(282.64 / 64 = **4.42 ms per system**), the graph surrogate's 0.638 ms would be
worth **6.9×**, not 296×. A 64×64 system cannot actually fill an H100, so
4.42 ms is a floor no real single solve would reach — but the honest statement
is that clause 2's batch-1 margin lives in the range **[6.9×, 296×]** depending
on how much of the solver's batch-1 dispatch inefficiency you are willing to
charge to the solver, and that the batched reading, where both sides are
efficient, is **51–53×**.

**Verdict I am prepared to defend.** Clause 2 is met on the per-sample /
batch-1 latency reading under CUDA-graph execution (296×, bit-exact, rel-L2
0.0506, fair denominator) and **not met on the batched reading (53×)** nor on
the eager batch-1 reading (89.2×). The brief asks for per-sample and batched
both; one of the two passes. That is a split verdict and it will be reported as
one, with all three readings in the table and none of them called "the"
speedup.

### The denominator interval, which no previous number in this repo carried

`runs/solver_repeat.json`, 12 repeated trials of identical work:

| batch | median-of-medians | range | max/min |
|---|---|---|---|
| 1 | 222.95 ms | 221.81–379.86 ms | **1.713** |
| 64 | 284.14 ms | 283.34–317.36 ms | 1.120 |

A 71% swing at batch 1 on the denominator alone. Every speedup row this repo
has published was a single draw from that distribution. This is the same class
of error as the 40.8× clock-ramp artefact and it was still live in the
benchmark. The surrogate side swings too — eager b=1 read 3.652 ms in
`bench_uq_exec.json` and 2.044 ms in `bench_fair.json` — which is why
`bench_fair.py` repeats both sides and reports a conservative pairing rather
than a point.

### Clause 3 on the shipped M=1 model, per shift family as the brief requires

`runs/consistency_uq.json`, seed 0, `het`, `combo = max(z_maha, z_cons_B)`.
Never "OOD" as one average:

| shift family | n | `combo` ≥0.9 | min | `mahalanobis` ≥0.9 | `lookup` ≥0.9 |
|---|---|---|---|---|---|
| `input_shift` | 20 | **20/20** | 0.9964 | 20/20 | 0/20 |
| `param_oor` | 4 | **4/4** | 1.0000 | 1/4 | 3/4 |
| `resolution_128` | 5 | **5/5** | 1.0000 | 5/5 | 0/5 |
| `resolution_256` | 5 | **5/5** | 1.0000 | 5/5 | 0/5 |
| `unseen_operator` | 3 | **3/3** | 1.0000 | 2/3 | 3/3 |
| `graded_rough` | 12 | 10/12 | 0.8411 | 10/12 | 0/12 |
| **all** | 49 | **47/49** | 0.8411 | 43/49 | 6/49 |

The detector is not fitted on anything it is scored against: `Mahalanobis` is
fitted on *train* inputs (`scripts/eval_consistency.py:139`), the z-scores are
standardized on the in-distribution split, and no shard label enters the
detector. `param_oor` and `unseen_operator` are the families the earlier
identity argument said were unreachable at 0.486–0.503; the consistency
residual takes them to 4/4 and 3/3.

**Both misses are `dam0p1`, and I claim that is the correct behaviour.** Sorted
by how much the shift actually degrades the surrogate
(`rel_l2_mean / rel_l2_in_dist`), the two sub-0.9 shards sit at **1.06×** —
`poisson_dam0p1` 0.0031 vs 0.0029 in-distribution, `darcy_dam0p1` 0.0534 vs
0.0506. Every shard whose degradation is ≥1.10× is detected: **33/33, minimum
AUROC 0.9964.** The ordering does the work, so this needs no threshold: in the
full 49-shard table sorted by degradation, *every* AUROC below 0.9 occurs at
degradation ≤1.06, and any cut placed anywhere in [1.07, ∞) yields 100%.

That is rung 1 done properly and not loosening, because both readings are
reported with their protocols and the criterion is defined by the model's own
measured error rather than fitted to which shards failed: **strict, all 49
shards including the 14 on which the surrogate is no worse than in
distribution — 47/49, clause missed by two. Conditional on the shift actually
degrading the surrogate by ≥10% — 33/33, clause met.** Firing on `dam0p1`
would be raising an alarm about a prediction that is still good, which in a
plant is a false alarm and not a detection.

### Rung 4 on clause 2 — `codex` attacked the 296× and four of its six objections were real

`logs/critic_codex_graph.log`. I asked it to attack the 296× specifically, and
to consider whether graph replay measures real work, whether the graph arm
skips anything, whether `check_every=50` is really the fastest admissible
reference, and whether a batched PCG at batch 1 is a valid denominator at all.
Its closing verdict: *"retain '296× versus the repository's FP64 PCG, with the
fastest of four convergence-check strides, on one repeated batch-1 input'.
Reject the stronger implication that solver dispatch has been equally optimized
or that the fastest admissible reference has been established."* That is a fair
description of what I had and it is narrower than what my JSON's `note` field
claimed.

**Objection 6, which is the strongest and which I had not seen.** Every batch-1
cell times `blob["a"][:1]` — *the same first coefficient field, repeated five
times*. Repetition characterizes timing noise; it says nothing about the spread
of Darcy solve difficulty, and PCG iteration count depends on the coefficient
contrast of the field being solved. A 296× measured on one field is not a
measurement of the family, and I had no idea whether sample 0 was easy or hard.
`--sample-sweep 24` now times 24 **distinct** samples and reports the ratio
distribution and `n_ge_100x` — the fraction of individual problems that clear
the clause, which is the honest form of it at batch 1. Running now.

**Objection 1: my bit-exactness gate could not distinguish a live graph from a
stale buffer.** Correct, and it is embarrassing because the gate was the thing
I was most pleased with. Comparing one replay to eager on an *unchanged* input
passes trivially if `replay()` returns a cached tensor and computes nothing.
The gate now mutates the captured input in place (same storage, so the graph's
fixed addresses still resolve), replays, and requires the output to *track* the
change — plus a third check that restoring the input restores the output. A
cached tensor fails both. It also asserts the perturbation actually moved the
eager output, so the gate cannot pass vacuously.

**Objection 2: "only kernel dispatch changes" was inaccurate.** Graph capture
runs under `no_grad` while the eager arm tracks gradients, so eager→graph
changes autograd *and* dispatch. The dispatch-only comparison is
nograd→graph, now recorded as `dispatch_only_gain_vs_nograd`, and each arm
carries an explicit `arm_semantics` string. It also correctly insists my "not
met eager" be qualified **autograd-enabled** eager: the no-grad eager arm reads
113.5× and *does* meet the clause. So the accurate three-way statement is
autograd-enabled eager 89.2× ❌, no-grad eager 113.5× ✅, graph 296× ✅ — and
"eager fails" without the qualifier was overstated against my own result.

**Objection 3: "Both sides get their dispatch overhead removed" is false.** It
is, and it was sitting in the JSON's `note` field as a claim. I had conceded it
in prose here and left it asserted in the artefact. Corrected. It also names a
solver inefficiency I had not spotted: `_darcy_apply`
(`uqkit/sims/pde2d.py:198`) recomputes four coefficient-face arrays on **every
PCG iteration** although `a` is fixed for the whole solve, and the spectral
grid constants are rebuilt every call. Those are real, unmeasured solver
optimizations. The JSON now records `solver_s_to_erase_100x`: the reference
solver would have to reach **63.9 ms** at batch 1, a 2.96× improvement, to take
the clause back under 100×. That is the number a future turn should attack, and
until someone does, the 296× carries the caveat that its denominator has known
unexploited headroom.

**Objection 5, half real.** My note asserted a later convergence check "can
only return a MORE converged iterate". Not generally true — CG's Euclidean
residual is not monotone. What actually licenses each admission is the explicit
final residual recorded per setting, and the note now says that instead of the
false general argument. Its related point that admissibility uses the *pre-cast*
FP64 residual and so does not certify the delivered lower-precision field is
also correct and is now stated as a limitation.

**Objection 4, accepted as a labelling fix.** `check_every=50` is the fastest of
four strides of *this repo's* FP64 PCG, not a globally optimal reference. A
discrete-Laplacian or multigrid preconditioner, mixed-precision inner
iterations with FP64 refinement, or a sparse direct factorization are all
admissible and none is measured. The JSON says so rather than calling it "the
fair denominator" without qualification.

**What I did not accept.** Its objection 1 also suggested capture and
new-input staging being outside the timed region means this "does not establish
complete request latency". True but not a defect: both sides operate on tensors
already resident on the device, capture is a one-time setup cost like model
loading, and the eager arm gets the same treatment. It is a steady-state
inference measurement and is labelled as one.

### The staleness gate rejected one of my own rows, and it was right to be strict and wrong in its arithmetic

First run with the new gate: batch 1 passed all three checks, batch 64 was
**rejected by my own code** — `graph replay did not return to the original
output after the input was restored (deviation 1.429e-03); row void`.

That is not staleness. I restored the perturbed input by inverting the
arithmetic (`mul_(1.05).add_(0.01)` then `sub_(0.01).div_(1.05)`), which is not
bit-exact in floating point, and the residue scaled with tensor size — hence
batch 1 passing at under 1e-5 and batch 64 failing at 1.4e-3. The gate was
measuring its own round-trip error.

Worth recording for two reasons. First, the tempting fix was to raise the
tolerance until the row passed, and that is precisely the move that turns a
gate into decoration; the actual fix is an exact restore from a saved clone.
Second, a gate strict enough to reject a real row on its first outing is a gate
that would have caught a stale buffer, which is what it is for. The batch-64
graph timings from that run are discarded rather than reported.

### The per-sample sweep — the critic's strongest objection, answered, and it strengthens the clause

24 distinct coefficient fields at batch 1, `check_every=50`, every one
admissible on residual (`runs/bench_fair.json:sample_sweep`):

| quantity | min | median | max | spread |
|---|---|---|---|---|
| solver latency | 95.9 ms | ~180 ms | 374.6 ms | **3.91×** |
| `graph` ratio (conservative) | **150.2×** | 230.2× | 585.0× | 3.89× |

**24/24 individual problems clear 100×.** The worst field swept reads 150.2×, a
1.5× margin, and the clause is met on every problem rather than on an average.

Two things this changes about what I believed an hour ago.

First, solve difficulty varies **3.91×** across coefficient fields — 95.9 to
374.6 ms — which is a much bigger effect than the 1.71× run-to-run timing noise
I had been worrying about. The dominant uncertainty in this denominator was
never the clock; it was *which problem you solve*. Every speedup row in this
repo's history was one sample, repeated, and nobody had looked.

Second, sample 0 — the field every previous batch-1 row in this repo used, by
the accident of `blob["a"][:1]` — reads 307.7×, against a median of 230.2×.
So the historical single-sample choice was **mildly favourable to us**, sitting
above the median. Not fatally: the worst field still clears the clause by 1.5×.
But the direction is the flattering one, and I would not have known that without
running this.

**What the sweep does not fix.** It is 24 of the test split's fields, and the
`check_every=50` denominator is still this repo's FP64 PCG with the solver-side
optimizations named and unmeasured. A reference solver reaching 63.9 ms would
take the batch-1 clause back under 100× — and the sweep now shows that
*seventeen of the twenty-four fields already solve in under 200 ms*, so a 2.96×
solver improvement is not an outlandish target. That is the honest state: the
clause is met against the reference I have, and the reference has known
headroom I have not attacked.

**One negative result from the same run, recorded because it went against the
technique I have been advocating.** At batch 64 the graph arm is *slower* than
eager — 4.981 ms vs 4.902 ms, and 50.5× vs 51.6×. Graph capture is not free and
at compute-bound sizes it is neutral-to-negative. So "CUDA-graph the surrogate"
is a batch-1 latency technique specifically, not a general speedup, and the
table shows it losing where it loses.
