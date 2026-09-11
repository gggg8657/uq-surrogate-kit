# When a surrogate cannot know it is wrong: four walls, and which three we
built ourselves

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
We first report all three clauses as failing, then show that **two of the three
failures were consequences of our own design decisions and dissolve when those
decisions are reversed** — which we regard as the paper's main result, because
the two surviving walls are then the ones that are actually about the problem.
**(i)** Speedup and calibrated uncertainty appear to be contested by a single
knob: the M-member ensemble that produces the interval costs M forward passes,
and the only configuration clearing 100× (M=1, 108.5×) has no spread and
therefore no interval and no OOD score. This is an artefact of building the
interval out of disagreement. A heteroscedastic σ head calibrated by the same
split conformal emits mean and interval in **one** forward pass and reaches
**0.9026** mean coverage with 8 of 8 seeds in band — so the coverage row and the
speedup row become the same row. We stress the control: a *constant* σ also
lands 8/8 in band, because split conformal rescales any σ to ~90% marginal
coverage, so coverage at M=1 is not evidence the head learned anything; the
head's measurable effect is a 13.1% sharper interval. On the timing side, the
clause turned out to be decided by a subsidy in our own benchmark — the
reference solver was timed with a device-to-host synchronization on every
iteration of up to 2000, and the surrogate was timed with autograd tracking
left on. Removing that subsidy, and then two further ones we found in our own reference
(face coefficients rebuilt every iteration on a fixed field; a preconditioner
whose symbol did not match the stencil being applied), the batch-1 ratio under
CUDA-graph replay falls from 602× to 216.4× on the field this repository
timed by default and to 21 of 24 distinct fields clearing 100×, worst 94.0×,
when the reference is given its best admissible configuration on each field.
Every fix is verified to leave the solver's answer bit-identical with its
residual inside tolerance. Only then did we audit the *surrogate*, and found
the same class of defect on our own side: its spectral convolutions permuted a
13.1 MB tensor of fixed weights on every forward pass, worth 1.246× once
cached and bit-identical when removed. That restores the clause at batch 1 —
**24 of 24 fields, worst 117.0×, median 187.4×, best 315.4×** — with the
single-field reading at **267.5×**. We report the whole sequence rather than
the endpoint, because the sequence is the finding: an apparent 6× margin over
the target was three layers of our own sloppiness in the baseline, and the
recovery was a fourth layer on the other side that three rounds of one-sided
scrutiny never looked for. The win is also a batch-1 latency win only: at batch
64, where both sides are dispatch-efficient, the ratio is 41.1× and graph
capture is slightly *slower*
than eager. **(ii)** Against a well-preconditioned
iterative solver at *matched accuracy*, the surrogate's advantage is 2.2×, not
the 26.8× obtained against the over-converged tolerance the corpus was generated
at; against exact spectral propagators the surrogate is 15–220× slower.
**(iii)** When the governing operator changes but the input distribution does
not, every unsupervised detector *of a certain family* is at chance — measured at 0.486–0.503 across
ensemble spread, input-space Mahalanobis distance and PDE residual, on cases
where the surrogate's relative error is 41 and 1,702 — because each is a
function of the input and the *configured* operator, and such a shift changes
neither. We show the hole is closed by one labelled probe: a conformal p-value
on k solver evaluations reaches 95% power at a measured 5% false-alarm rate with
**k = 1** on exactly those cases, while correctly declining to alarm on shifts
that do not degrade the surrogate. **(iv)** Coverage under *covariate*
shift is the clause that survives longest, and it too is partly self-inflicted.
Deliberate abstention — certify where competent, refuse elsewhere — gives **0 of
32** shards in band on 8 of 8 seeds, at every abstention rate we swept up to
0.95, and **also 0 of 32 when the gate is an oracle on the true error**: the
conformity score is a ratio, a gate selects on its own score, and the two
correlate at only +0.256 (+0.697 for the oracle). Selection acts on the
population; the failure is in the scale. Restating the clause without any
uncertainty method in it — the widths giving coverage in [0.88, 0.92] are
exactly [Q₀.₈₈(S), Q₀.₉₂(S)] — turns it into a requirement to predict interval
width to **4.47%** at the median across a **107.6×** range, and that reframing
exposed the bug: four of five families are linear in the shifted channel, so
`u(c·f) = c·u(f)`, but inputs are standardized with frozen calibration
statistics and a 2× input extrapolates instead of scaling. A test-time
equivariance wrapper, exact for any predictor and requiring no retraining,
takes relative error on those shards from 0.3111–0.4711 to
**0.981–0.997× the in-distribution error of the same checkpoints**, with a
registered control (Darcy, whose shifted channel enters the operator
nonlinearly) unchanged at ratio 1.0000. A *learned* width model reaches 4 of 32
— and its dominant coefficient is the log input amplitude, 5.78× the next, with
0 of 32 on 8 of 8 seeds once that feature pair is deleted, so the learned and
the closed-form corrections are the same correction and composing them is
significantly worse than either. After the repair the residual requirement is
±4% width accuracy over 68.4×, and nothing we tried reaches it, including a
model fitted on the evaluation shards. We therefore report **three** clauses
whose apparent walls were our own construction and one that is not, and we
regard the recurrence — the same class of defect found four separate times,
three of them only after the number had been published — as the transferable
result.

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

A fourth denominator turned out to matter more than any of these, and it is one
we had chosen without noticing. The reference solver's convergence test called
`.max()` and compared it in Python on **every** PCG iteration, forcing a
device-to-host synchronization up to 2000 times per solve, while the surrogate
was timed with autograd tracking enabled. Neither is what a deployment runs.
Amortizing the solver's test over 50 iterations (which returns an iterate whose
*measured* final residual is 6.8e-11 against a 1e-10 tolerance, so it is the
same solver) and removing the surrogate's autograd and per-kernel dispatch:

| arm at batch 1, sample 0 | vs the repo's PCG | vs the *optimized* PCG | + packed weights | subsidized |
|---|---|---|---|---|
| eager, autograd on (the protocol above) | 90.0× | 64.5× | **67.6×** | 79.5× |
| eager, no autograd | 113.6× ✅ | ~90× ❌ | **101.4×** | 120.0× |
| CUDA-graph replay | 328.5× | 216.4× | **267.5×** | 312.1× |

The middle column is the one to read, and it exists because we went looking for
work our own reference was doing redundantly. Three defects, each verified to
leave the solver's answer **bit-identical** (solution deviation 0.000e+00) with
its residual inside tolerance, so none of them changes the problem being solved:
(i) the convergence test forced a device-to-host synchronization on every one of
up to 2000 iterations; (ii) `_darcy_apply` rebuilt four face-coefficient arrays
every iteration although the coefficient field is fixed for the whole solve;
(iii) the FFT preconditioner used the *continuous* Laplacian symbol while the
operator applies a 5-point stencil — a ~2.5× mismatch at Nyquist — costing
iterations (839 → 775 with the matching discrete symbol).

**Each round moved the number against us, and the third took the clause under
the target.** We report the sequence rather than the endpoint, because the
sequence is the finding: an apparent 6× margin over the KPI was three layers of
our own sloppiness in the reference.

The eager row crosses the 100× threshold *in the subsidy alone*. We report this
as the paper's most uncomfortable measurement: for the protocol under which
every earlier number here was produced, the KPI verdict was determined by a
defect in the reference, not by the surrogate.

Two caveats keep the 328.5× honest. First, the batch-1 comparison is not
symmetric: the reference solver is as dispatch-starved at batch 1 as the
surrogate was — 215 ms for one system against 283 ms for sixty-four — and only
the surrogate was graph-captured. `_darcy_apply` recomputes four coefficient-face
arrays on every iteration although the coefficient field is fixed for the whole
solve; that optimization is named and unmeasured. A reference reaching 63.9 ms
would take the clause back under 100×. Second, every batch-1 row in the
literature-style table above times *one* coefficient field, and that turns out to
decide the clause. Solve difficulty varies 3.26× across fields (62.4–203.6 ms).
Sweeping 24 distinct fields, with the reference given its best admissible
convergence stride on **each field independently** — imposing one field's stride
on all of them inflated our own worst case by 20%, because the stride rounds the
stopping iteration up to a multiple of itself and so penalises the easy fields —
21 of 24 clear 100×; the worst reads 94.0× and the median 144.4×, and the three
failures are the *easiest* fields, where the solver finishes in 62–64 ms against
the surrogate's 0.638 ms. We had published the break-even before the run that
reached it: "a reference reaching 63.73 ms on the hardest field ends the
clause". The field that ended it solves in 63.77 ms.

### Auditing the numerator, three rounds late

At that point the missing factor had to come from the surrogate, and we had
never looked. A kernel census of the batch-1 forward says the arithmetic is 5%
of it: eight matmuls and eight FFTs total 65.6 µs against 258.7 µs of
`aten::copy_`. The cause is a lowering detail. `einsum("bixy,ioxy->boxy", ·)`
becomes a batched matmul over the `(x, y)` mode grid, so it must permute the
`(in, out, m1, m2)` weight — 13.1 MB per spectral layer — into `(x, y, in, out)`
order on **every forward pass**, although those weights are frozen at inference
and the permutation is identical every time. Packing them once at load and
contracting with `bmm` costs +105 MB of device memory (a second copy of the
spectral weights) and returns **1.246×** at batch 1.

This is an execution change, not a model change, and we hold it to the bar we
held the solver's fixes to: `max|Δ| = 0`, exact equality rather than a
tolerance, on the predicted mean *and* both conformal interval bounds, across
**64 of 64** data shards at resolutions 64, 128 and 256 — the bounds because a
change confined to σ would leave the mean identical while silently moving every
coverage number, and the resolutions because the weight cache is keyed on the
retained mode counts, which saturate above N = 40. Relative error is unchanged
to every stored digit.

**Clause 2 is therefore met at batch 1: 24 of 24 fields, worst 117.0×, median
187.4×.** The paired comparison is what makes it credible rather than a lucky
draw. The three fields that had failed carry denominators that moved by less
than 1% between the two runs while their ratios rose 23–25%, matching the
surrogate's own 1.246×; two unrelated fields show 1.7× denominator swings, which
is inside the 1.71 max/min this reference exhibits on repeated identical work,
and those rows are excluded rather than banked.

Three conditions travel with the result. At batch 64 the ratio is 41.1× — the
reference batches better than the surrogate does (1.41× wall clock for 64× the
arithmetic, against 8.5×). Without graph capture the same 24 fields give 2 of
24. And the reference remains a Python PCG loop launching individual kernels;
fused stencil kernels, cached grid-dependent preconditioner data and
graph-captured fixed-length PCG chunks are all admissible and all unmeasured, so
the defensible claim is *24/24 measured fields against this specified PCG
implementation*, not an advantage over an equivalently optimized reference. At a
0.512 ms surrogate, a reference reaching 51.2 ms on a field ends the clause on
it; the fastest field currently solves in 60.3 ms.

The methodological point is the one we would keep if we kept only one. Three
consecutive rounds of scrutiny went into the denominator, each correctly, each
moving the number against us — and none into the numerator, where a repeated
memcpy of constants was worth 1.246× and took four minutes to find. One-sided
scrutiny is not conservatism. It finds every reason a number is too high and
none of the reasons it is too low, and it feels like rigour the entire time.

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

## 5. Coverage under shift: what three attacks and one bug taught us

In distribution the interval clause is met (0.9026, 8/8 seeds, one forward
pass). Under covariate shift, on 32 shards that leave the governing operator
alone, the shipped single-network model puts **0 of 32** shards inside the
90±2% band under the same per-family calibrator the in-distribution headline
uses. Covariate-shift-weighted conformal, on an honest held-out choice of its
density-ratio clip, reaches **1.94 of 32**. We attacked the gap three ways.

**(a) Deliberate abstention does not work, and an oracle does not rescue it.**
The natural product answer is to certify where the model is competent and
refuse elsewhere, reporting the abstention rate rather than hiding it. We gated
the certificate on the model's own relative predicted spread at a
pre-registered 5% in-distribution false-alarm rate, calibrating the conformal
quantile on the accepted calibration points so the certified and calibrated
populations coincide. Result: **0 of 32 shards in band on 8 of 8 seeds**, at a
median abstention of 0.390. Sweeping the abstention rate to **0.95** does not
help — only 2 of 32 shards enter the band at *any* rate, and they are the
mildest rung of the graded ladder and an over-covering shard that reaches 0.90
from above by discarding most of itself. Replacing the gate with an **oracle on
the true relative error** — not shippable, reported as a ceiling — also gives
**0 of 32**, at abstention 0.858.

The mechanism is measurable and it generalizes past our gate. Coverage fails on
the conformity score max|μ−u|/σ̃, a *ratio*; a gate selects on its own score.
The median within-shard rank correlation between the two is **+0.256**, and
**+0.697** even for the oracle, which knows only the numerator. Under shift σ̃
under-predicts the error at fixed error magnitude, so the ratio is inflated
across the whole shard rather than in a selectable tail. **Selection acts on
the population; the failure is in the scale.** No gate is the right shape of
tool, and the gate we built is not the reason.

The gate is nonetheless an excellent *shift detector* — Spearman correlation
between its abstention rate and shard error is 1.000 and 0.971 on the two
graded ladders — and it costs only its calibrated 5% in distribution. "The
surrogate says when not to trust it" is true; "and then its interval is 90%
correct where it does trust itself" is false, and these are different claims.

**(b) The clause, restated without any uncertainty method in it.** For a
per-sample score S the width achieving coverage exactly *p* **is** the *p*-th
quantile of S, so the widths keeping coverage inside [0.88, 0.92] span exactly
[Q₀.₈₈(S), Q₀.₉₂(S)]. The relative tolerance is therefore
(Q₀.₉₂ − Q₀.₈₈)/Q₀.₉₀ — three order statistics, no calibrator, no detector.
Measured over the 32 shards it is **4.47%** at the median (0.40–19.75%), and
changing the score does not escape it: `norm_ratio`, an aggregate rather than a
maximum over 4,096 pixels, is *tighter* at 4.22%. So the clause is arithmetically
a requirement to **predict the interval width to about ±4%** across a required
dynamic range that we measure at **107.6×**. Stating it this way is what made
the next result findable.

**(c) Most of that dynamic range was a bug of ours, not a property of the
physics.** The range is concentrated in the four amplitude-shifted shards.
Four of the five families are *linear* in the shifted channel, so
`u(c·f) = c·u(f)` exactly — but the surrogate standardizes its inputs with
frozen calibration statistics, so a 2× input extrapolates instead of scaling. A
test-time wrapper `F_eq(a) = s(a)·F(a/s(a))` with `s(c·a) = c·s(a)` restores the
equivariance for *any* F, with no retraining. On the four linear amplitude
shards relative error falls from **0.3111–0.4711** to **0.00257–0.00338**, i.e.
to **0.981–0.997×** the in-distribution error of the same checkpoints: the
shift is not improved, it is *neutralised*. The registered control is Darcy,
whose shifted channel is log-permeability and enters the operator nonlinearly,
so the algebra must not apply — its ratio is **1.0000**, unchanged. Every
non-amplitude shard moves by at most 2.4e-05.

**(d) And the learned alternative turns out to be the same correction.** Fitting
a difficulty model h(z) to the conditional 90th percentile of S and
conformalizing S/h(z) — a linear pinball-quantile regression on
deployment-observable features, fitted on a disjoint development shift suite
and evaluated leave-one-mechanism-out — moves the clause from 0 to a median of
**4 of 32** (per seed 2–8, exact sign-flip p = 0.0078) with in-distribution
coverage unchanged. But its largest coefficient is the log input amplitude at
**+2.8946**, 5.78× the next term on 8 of 8 seeds, and **deleting that one
feature pair returns it to 0 of 32 on 8 of 8 seeds**. Composing the learned
model with the closed-form repair is significantly *worse* than either alone
(1 of 32, paired p = 0.0234), which is what double-counting one correction
looks like. Fitting h per family instead of pooled is a null (4 vs 4,
p = 0.8438) whose in-sample ceiling nonetheless rises from 4 to 6 — capacity
that does not transfer. The learned width model's held-out result already
equals its own in-sample ceiling (p = 0.5156).

**What survives.** Every intervention that has ever moved this clause moved it
by correcting the input amplitude, and that correction is available exactly and
for free. Once it is applied the residual requirement is to predict width to
±4% over a **68.4×** range, and nothing here reaches it — not selection at any
abstention rate, not an oracle on the error, not a learned width model even
when fitted on the evaluation shards themselves. We report this as the wall,
stated in units that do not mention uncertainty quantification, rather than as
a property of conformal prediction.

## 6. A note on the residual as a trust signal

Applying `L` to a surrogate's output amplifies the round-off already in it by
the operator's symbol, so the residual of the *exact* solution is not zero and
its floor scales as `N^order`: 5.1e-5 (Poisson, order 2) to 53 (order 6) at 64².
Above fourth order the check returns nothing but round-off, and a detector built
on it would report a magnificent AUROC driven entirely by numerical noise. The
floor is measured (`runs/residual_floor.json`) and pinned in both directions by
tests.

## 7. Limitations

Five checkpoints, so the ensemble-size trend is a screen, not a verdict; the
C(5,M) subset spread is reported per row. One resolution (64²), one geometry
(periodic square), one architecture. `field_max` coverage is not comparable
across resolutions — it is a maximum over a pixel count that changes — and the
resolution rows should be read on `norm_ratio` instead. The covariate-shift
guarantee assumes `p(y|x)` is unchanged, which the operator-shift shards violate
outright; those rows are marked and excluded from the clause. Timing ratios
should not be read to better than ±15% between scripts.

## 8. What this says about the pitch

"A surrogate that also says when not to trust it" survives contact with the
first two of three questions and fails the third in a way that is worth knowing:
it can tell you when its *input* is unfamiliar, it cannot tell you when its
*question* has changed, and the second failure is the one that produced a 1,702×
error here. The deployable form of the pitch is therefore a surrogate plus a
small standing budget of solver calls — one, on this corpus — and a kit that
knows which of the two questions it is answering.
