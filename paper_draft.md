# When a surrogate cannot know it is wrong: three walls in surrogate uncertainty

*Working draft. Every number is produced by a run in this repository and
regenerated into `RESULTS.md` by `scripts/report.py`; nothing here is typed by
hand or taken from a paper. Published baselines, where cited, are marked as such
and kept in their own column.*

## Abstract

We build a simulator-agnostic toolkit for surrogate uncertainty — split, group
and covariate-shift-weighted conformal calibration, deep-ensemble and
physics-residual scores, and a timing harness — and evaluate it against a
three-clause target: 90±2% conformal coverage, ≥100× inference speedup, and
≥0.9 OOD detection AUROC. In-distribution coverage is met (88.6%, [87.7, 89.5]).
The other two clauses fail, and the failures are structural rather than
budgetary. **(i)** Speedup and calibrated uncertainty are contested by a single
knob: the M-member ensemble that produces the interval costs M forward passes,
and the only configuration clearing 100× (M=1, 108.5×) has no spread and
therefore no interval and no OOD score. **(ii)** Against a well-preconditioned
iterative solver at *matched accuracy*, the surrogate's advantage is 2.2×, not
the 26.8× obtained against the over-converged tolerance the corpus was generated
at; against exact spectral propagators the surrogate is 15–220× slower.
**(iii)** When the governing operator changes but the input distribution does
not, every unsupervised detector is at chance — measured at 0.486–0.503 across
ensemble spread, input-space Mahalanobis distance and PDE residual, on cases
where the surrogate's relative error is 41 and 1,702 — because each is a
function of the input and the *configured* operator, and such a shift changes
neither. We show the hole is closed by one labelled probe: a conformal p-value
on k solver evaluations reaches 95% power at a measured 5% false-alarm rate with
**k = 1** on exactly those cases, while correctly declining to alarm on shifts
that do not degrade the surrogate.

## 1. Setup

Five PDE families on a 64² periodic torus (Poisson, Helmholtz, diffusion,
advection–diffusion, variable-coefficient Darcy), reference solutions from exact
spectral propagators, a batched preconditioned CG in fp64, and pseudo-spectral
RK4. A task-conditioned Fourier neural operator (26.25M parameters, 60 epochs)
is trained five times from different seeds to form a deep ensemble; in-
distribution relative L2 is 0.0013–0.0430. Calibration, test and training splits
are disjoint by seed; 49 shifted shards — input-distribution shifts, out-of-range
parameters, unseen operators and unseen resolutions — are generated once and
seen by no training, calibration or detector fit.

## 2. The interval: four questions, not one

"90% coverage" of a *field* is ambiguous, and the ambiguity is worth several
points. We calibrate four nonconformity scores side by side (§`uqkit/conformal`):
simultaneous over pixels (`field_max`), marginal over pixels (`pixel`),
aggregate per sample (`norm_ratio`), and σ-free (`rel_l2`). All four land in
band in distribution (88.6–89.8% pooled). Pixel coverage is reported with a
cluster bootstrap over fields rather than a binomial interval, because 4,096
pixels of one field are not 4,096 independent observations — the naive interval
is roughly 64× too narrow.

**Under shift the dominant failure is over-coverage.**
24 of 32 covariate-shift shards sit at 99–100%: the σ-scaled score's denominator
grows with the shift, so the band widens faster than the error. A two-sided band is the right target for
exactly this reason.

**Weighted conformal has a support-limited window.** Adding a graded roughness
ladder (Δα ∈ {0.1 … 1.0}) turns a cliff into a threshold: coverage returns to
band at Δα ≤ 0.2 and beyond that the density-ratio probe saturates at AUC 1.00,
the supports cease to overlap, and the exact weighted quantile returns +∞ for
512/512 test points. This is visible only because the quantile is computed
exactly per test point; the common median-weight approximation returns a finite
quantile and a clean-looking but vacuous 100%.

## 3. The speedup: three denominators

A surrogate speedup is a ratio, and the denominator is a choice.

| denominator | Darcy, batch 1, 5 members |
|---|---|
| the tolerance the corpus was generated at (1e-10) | 26.8× |
| the same, repeated 8× under steady clocks | 23.5× [23.0, 23.7] |
| the cheapest solver setting at matched accuracy | **2.2×** |

The third is the honest one and it is an *upper bound*: the sweep never made the
PCG as inaccurate as the surrogate. At a 10⁻¹ residual tolerance the solver still
reaches rel-L2 8.5e-5 against a 10⁻¹² reference — 500× better than the
surrogate's 4.3% — in 48 ms against the surrogate's 22 ms. On the four families
with an exact propagator the surrogate loses outright by 15–220×.

Two measurement lessons are recorded rather than hidden. Batching favours the
*solver*: the PCG costs 255 ms for one sample and 316 ms for sixty-four, so the
speedup is a latency win that falls from 26.8× to 14.2× per sample. And an idle
H100's clock ramp produced a 74% inflation of our own first headline; repeating
one configuration eight times gave a ±36% spread, and a 6-second warmup before
any measurement brought it to ±1%.

## 4. The blind spot, and its price

Let `D` be any deployment-time detector: a function of the input `x` and the
configured operator `L̂`. Consider a shift that replaces the true operator `L`
with `L′` while leaving the input distribution and `L̂` unchanged. Then `D` is
distributed identically before and after, so its AUROC is 0.5 by construction.
This is not a bound to be tightened; it is an identity.

Measured, on shards built to satisfy exactly that condition (the interpolating
families inherit their parent's input distribution by design):

| shard | rel-L2 | spread | Mahalanobis | residual |
|---|---:|---:|---:|---:|
| `biharmonic` | 41.25 | 0.486 | 0.497 | 0.495 |
| `frac_s3` | 1702.5 | 0.499 | 0.492 | 0.502 |
| `frac_s0p25` | 0.949 | 0.496 | 0.497 | 0.496 |
| `frac_s0p5` | 0.857 | 0.499 | 0.503 | 0.494 |

The alternative explanation — an under-powered ensemble — is ruled out by the
same detectors reaching 1.000 on `darcy_c3` (rel-L2 0.77) and `navier_stokes`
(0.89), shifts of comparable severity whose inputs *do* move.

**The price of closing it.** k labelled probes, scored by a conformal p-value
against the calibration error distribution and combined with Fisher's method.
False-alarm rate verified on held-out in-distribution data (0.046–0.059 at k=1).
**k = 1** suffices for 95% power on every shard the unsupervised detectors miss.
And where the surrogate is *not* degraded — all ten resolution shards, where
`mahalanobis` fires at AUROC 1.000 — the probe requires k > 64, i.e. it declines
to alarm. Distribution-shift detection and error detection are different
questions; a detector can be perfect at one and worse than useless at the other.

## 5. A note on the residual as a trust signal

Applying `L` to a surrogate's output amplifies the round-off already in it by
the operator's symbol, so the residual of the *exact* solution is not zero and
its floor scales as `N^order`: 5.1e-5 (Poisson, order 2) to 53 (order 6) at 64².
Above fourth order the check returns nothing but round-off, and a detector built
on it would report a magnificent AUROC driven entirely by numerical noise. The
floor is measured (`runs/residual_floor.json`) and pinned in both directions by
tests.

## 6. Limitations

Five checkpoints, so the ensemble-size trend is a screen, not a verdict; the
C(5,M) subset spread is reported per row. One resolution (64²), one geometry
(periodic square), one architecture. `field_max` coverage is not comparable
across resolutions — it is a maximum over a pixel count that changes — and the
resolution rows should be read on `norm_ratio` instead. The covariate-shift
guarantee assumes `p(y|x)` is unchanged, which the operator-shift shards violate
outright; those rows are marked and excluded from the clause. Timing ratios
should not be read to better than ±15% between scripts.

## 7. What this says about the pitch

"A surrogate that also says when not to trust it" survives contact with the
first two of three questions and fails the third in a way that is worth knowing:
it can tell you when its *input* is unfamiliar, it cannot tell you when its
*question* has changed, and the second failure is the one that produced a 1,702×
error here. The deployable form of the pitch is therefore a surrogate plus a
small standing budget of solver calls — one, on this corpus — and a kit that
knows which of the two questions it is answering.
