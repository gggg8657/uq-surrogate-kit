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
