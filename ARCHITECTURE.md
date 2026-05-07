# MAK2+ Engine Architecture

> **Audience.** A senior software engineer with no molecular-biology
> background. The biochemistry is explained where it matters; the engineering
> is the focus.
>
> **File location.** This file lives at the repo root during Phase 0
> (the Streamlit-app source repo). At the start of Phase 1, the engine
> files will be copied into `backend/engine/` of the new
> Next.js + FastAPI repo, and this document moves with them to
> `backend/engine/ARCHITECTURE.md`.

---

## 1. What MAK2+ does

MAK2+ takes raw fluorescence vs. cycle-number data from a quantitative-PCR
experiment and returns the absolute number of starting template DNA
molecules per well, without requiring the user to also run a standard
curve from known dilutions. The fluorescence is what a qPCR instrument
records — one number per well per cycle, typically across 40-45 cycles
— and the engine fits a mechanistic model of the chemical reaction to
each well's curve. The fitted starting-template parameter (`D0`),
multiplied by a per-instrument calibration constant, gives the absolute
copy count.

---

## 2. The MAK2 model

The model is from **Boggy & Woolf (2010), PLoS ONE 5(8):e12355**, with an
extension that explicitly tracks primer depletion across cycles.

### 2.1 Per-cycle update equations

Given DNA fluorescence `D[n-1]` and primer pool `P[n-1]` at the previous
cycle, three equations advance to the next cycle:

```
k_eff      = k * P[n-1]
D[n]       = D[n-1] + k_eff * ln(1 + D[n-1] / k_eff)
P[n]       = max(0, P[n-1] - (D[n] - D[n-1]))
F[n]       = D[n] + F_bg_intercept + F_bg_slope * n
```

Plain-English meaning of each term:

- **`D[n]`** — DNA fluorescence at cycle `n`. The kinetic state variable.
  Note this is in **fluorescence units** (RFU), not molecule counts —
  see §2.3.
- **`k`** — A single rate constant. Has units of inverse primer
  concentration. Encodes how aggressively each new DNA molecule
  consumes primers. Small `k` → primers abundant → near-perfect
  doubling. Large `k` → primers scarce → curve plateaus quickly.
  Typical fitted values: 0.05 – 1.5.
- **`P[n]`** — Primer pool at cycle `n`. Starts at `P0`, depletes by
  exactly the per-cycle DNA increment (one primer pair consumed per
  new double-stranded product). When `P[n]` hits zero, growth stops.
- **`k_eff`** — Effective per-cycle rate. Shrinks as `P[n-1]` shrinks;
  this is what bends the exponential into a plateau.
- **`F[n]`** — Total observed fluorescence: amplification signal `D[n]`
  plus a linear instrument background. Two background parameters
  (`F_bg_intercept`, `F_bg_slope`) capture the optical baseline plus
  any per-cycle drift (typically photobleaching or dye degradation).

### 2.2 Why this produces a sigmoid

The closed-form `D[n] = D[n-1] + k_eff * ln(1 + D[n-1] / k_eff)` is the
analytical solution to the Michaelis–Menten-style differential equation
`d[D]/dt = k * P * D / (D + k*P)`. Two limits:

- **Early cycles** (`D << k*P`): `ln(1 + x) ≈ x`, so
  `D[n] ≈ 2 * D[n-1]` → near-perfect exponential doubling.
- **Late cycles** (`D >> k*P` after primer depletion): `ln(1 + x) ≈ ln(x)`,
  so `D[n] ≈ D[n-1] + k_eff * ln(D[n-1]/k_eff)` → growth rate saturates
  at a small fraction of `k_eff`, producing the plateau.

The sigmoid shape qPCR practitioners draw on a whiteboard is the visible
signature of this two-regime behaviour.

### 2.3 The `D0` units choice (and why it bites you)

The original Boggy & Woolf formulation factored the unobservable
fluorophore-per-template scale out as a separate parameter `F_scale`.
Here we fold it into `D0` directly:

```
D0_observable  =  D0_molecules  ×  F_scale
```

This makes `D0` directly observable (the optimiser is fitting an RFU-scale
parameter against RFU-scale data), at the cost that converting `D0` to
molecule counts requires a per-instrument calibration constant
(`D0_single`, the fluorescence per single template molecule). The
calibration step is `Copies_D0 = D0 / D0_single`, implemented in
`calibration.apply_calibration`.

If you find yourself confused why `D0` is ~1e-4 instead of "around 100
molecules" — this is why.

---

## 3. Why mechanistic modelling matters

In engineering terms: **Ct-based qPCR is a calibration trick; MAK2 is a
mechanistic model**. The contrast looks like this:

| Concern | Ct method | MAK2 |
|---|---|---|
| What the per-well calculation finds | The cycle index where fluorescence crosses a fixed threshold | Best-fit values for `(D0, k, P0, F_bg_intercept, F_bg_slope)` |
| What you need to convert it to copies | A standard curve from known dilutions on the *same plate* | A single per-instrument calibration constant `D0_single`, captured once |
| Implicit assumption about per-well chemistry | Same amplification efficiency across wells | None — efficiency is fitted per well |
| Failure mode when assumption breaks | Quietly wrong copy counts; no per-well diagnostic | Per-well R² and quality gates flag the failure |

The standard-curve requirement is the practical pain. Every plate needs a
serial dilution of known concentrations to anchor the threshold, which
typically eats 10-20% of well capacity and adds reagent cost. With MAK2,
the calibration is run once during instrument commissioning, and every
subsequent plate gets absolute copies without consuming wells for
standards.

The "efficiency assumption" failure mode is more subtle but
scientifically more important: per-well efficiency varies in real plates
because of pipetting variability, primer-template mismatch, inhibitors
co-purified with the template, etc. Ct-based methods read all wells
through the same standard curve, so per-well efficiency variation
propagates directly into systematic copy-number error. MAK2 fits each
well independently, so per-well efficiency just becomes part of `k`.

---

## 4. File responsibilities

The engine is seven Python files, all at the repo root in this Streamlit
phase. Lines in parentheses are post-Phase-0-cleanup file sizes.

### Core engine (will be `backend/engine/` in Phase 1)

- **`mak2_model.py`** (1473 lines) — The forward simulator and the
  analytical bounds-derivation code. Class `MAK2Model` simulates `F[n]`
  forward from `(D0, k, P0)`. Free functions `estimate_D0_bounds`,
  `estimate_k_from_exponential`, and
  `estimate_MAK2_params_from_exponential` derive data-driven box bounds
  for the optimiser by fitting two exponentials to the early cycles of
  each well. Also contains `pre_estimate_background` (the linear
  baseline regression) and `find_slope_threshold_cycle` (locates the
  inflection cycle for fit-window truncation).

- **`optimizer.py`** (2856 lines) — The tiered fitter. Class
  `MAK2Optimizer.fit()` orchestrates seven optimisation tiers
  (T1-Full, T1-Fixed, T1.5, T2-LHS, T2.5, T3-DE, T4) and the post-fit
  quality gates. Also exposes `predict`, `calculate_fit_metrics`,
  `calculate_ct`, and `check_plateau_overshoot` for downstream callers.

- **`config.py`** (1-line constant) — Exposes `RANDOM_SEED` from the
  `MAK2_RANDOM_SEED` environment variable. Drives test reproducibility.

- **`data_processing.py`** (256 lines) — Two small utilities:
  `estimate_baseline_end` (algorithmic baseline-end discovery, used as
  a fallback / cross-check against the instrument's metadata) and
  `detect_no_signal_samples` (plate-wide triage to flag wells with no
  amplification before the expensive fit runs).

- **`qpcr_data_converter.py`** (1389 lines) — Format adapters. Class
  `QPCRDataConverter` reads any of six vendor formats (simple CSV, wide
  CSV, Bio-Rad CFX, single- and multi-sheet QuantStudio, ABI sectioned
  CSV) into one canonical `(cycles, samples, metadata)` shape.
  `load_abi_results_csv` parses the per-well metadata CSV that ABI
  instruments export alongside fluorescence.

- **`calibration.py`** (1357 lines) — Standard-curve and
  limited-dilution calibration. `build_standard_curve` fits a power-law
  `log10(copies) = slope * log10(D0) + intercept` from STANDARD wells.
  `build_limited_dilution_calibration` derives the conversion factor
  from a Poisson-distributed dilution panel without standards. The
  `apply_*` functions add `Copies_D0` / `Copies_Ct` columns to the
  results DataFrame.

- **`replicate_analysis.py`** (524 lines) — Replicate aggregation and
  the D0-vs-Ct precision comparison. `calculate_replicate_stats`
  computes per-group mean/SD/CV%; `compare_precision` derives the
  `|ln(E)| × SD(Ct)` conversion that lets D0-CV% and Ct-CV% be
  compared on the same concentration scale. `analyze_dilution_series`
  fits standard curves on both axes and reports linearity.

- **`bootstrap.py`** — Parametric bootstrap for kinetic-parameter
  uncertainty. Resamples residuals with replacement, refits the MAK2
  model on each resample, returns the per-parameter CI. Optional
  diagnostic; not used by the production batch path.

### Application code (out of scope for the Phase-1 port)

- **`app.py`** (~6000 lines) — Streamlit UI. Will be replaced by a
  Next.js front-end in Phase 1. The per-well preprocessing logic
  duplicated here is one of the major sources of tech debt to be
  resolved during the unification work; see CLAUDE.md.
- **`run_batch.py`** (1777 lines) — Headless batch driver. Same per-well
  preprocessing as `app.py`'s batch mode, plus Excel writing and
  multi-plate orchestration. Will be replaced by the FastAPI batch
  endpoint in Phase 1.
- **`benchmark_tiers.py`**, **`example_data_loader.py`**,
  **`sample_selector_ui.py`** — Tooling and UI helpers. Out of scope.

### Call graph (engine only)

```
                                  ┌───────────────────────┐
                                  │ qpcr_data_converter   │
                                  │  (format → canonical) │
                                  └────────────┬──────────┘
                                               │
                                  ┌────────────▼──────────┐
                                  │ data_processing       │
                                  │ - estimate_baseline   │
                                  │ - detect_no_signal    │
                                  └────────────┬──────────┘
                                               │ per-well
                                  ┌────────────▼──────────┐
                                  │ mak2_model            │
                                  │ - pre_estimate_bg     │
                                  │ - estimate_D0_bounds  │
                                  │ - estimate_MAK2_params│
                                  └────────────┬──────────┘
                                               │ bounds + initial guess
                                  ┌────────────▼──────────┐
                                  │ optimizer             │
                                  │  MAK2Optimizer.fit()  │
                                  │  - T1 → T1.5 → T2 →   │
                                  │    T2.5 → T3 → T4     │
                                  │  - quality gates      │
                                  │  - calculate_ct       │
                                  └────────────┬──────────┘
                                               │ per-well results
                                  ┌────────────▼──────────┐
                                  │ replicate_analysis    │
                                  │  + calibration        │
                                  └───────────────────────┘
```

---

## 5. The fitting pipeline

End-to-end, from a raw plate file to copies-per-well, for a single well:

1. **Parse** (`qpcr_data_converter.QPCRDataConverter.load_from_file`).
   Format-detect the input file, return a canonical
   `(cycles, samples, metadata)` triple.

2. **Triage** (`data_processing.detect_no_signal_samples`). Plate-wide
   pass that flags NTC controls, failed reactions, and obvious
   non-amplifiers before spending optimiser time on them. Three-stage
   classification: range-only rejection → exponential R² screen →
   linear-vs-exponential discriminator.

3. **Per-well preprocessing** (currently duplicated between
   `run_batch.run_pass1` and `app.py`'s batch mode — Phase-1 unification
   target):

   a. **No-amplification recheck** — local 5σ-vs-tail comparison;
      catches any borderline wells that slipped past the plate-wide
      triage.
   b. **Background pre-estimate** (`mak2_model.pre_estimate_background`)
      — linear regression on the cycles in the metadata's
      `Baseline Start`/`Baseline End` window, when available.
   c. **Smart-start inflection search** — locate the qPCR sigmoid's
      inflection cycle using a 5-point-stencil derivative, scanning
      right-to-left to avoid early-cycle noise spikes that look like
      gradient peaks.
   d. **Adaptive window extension** — walk the fit window's start cycle
      backward in 2-cycle steps until at least 3 in-window cycles fall
      below the projected background line. The optimiser needs those
      baseline cycles to anchor the background away from `D0`.
   e. **Safety-net check** — if the truncated window's signal range is
      <70% of the full-trace range, smart-start probably missed the
      sigmoid (very late amplifier) — fall back to using the whole
      trace from the floor cycle.

4. **Bounds derivation**
   (`mak2_model.estimate_MAK2_params_from_exponential`). Fit two
   exponentials (perfect doubling and fitted-efficiency) to the
   automatically detected exponential region; use them to bracket
   `D0`. Combine with empirical `k_upper = 0.2 - 0.03 * log10(D0)`
   and `P0 ≈ F_max` to produce a complete 5-parameter box.

5. **Optimisation** (`optimizer.MAK2Optimizer.fit`). Walk the tier
   escalation described in §6. Stop at the first tier whose result
   clears the R² threshold and the §7 quality gates.

6. **Ct calculation** (`optimizer.MAK2Optimizer.calculate_ct`).
   Threshold-crossing on the fitted curve, with baseline regression
   matching what the instrument firmware does. Optionally
   ROX-normalised (run_batch handles the per-well swap).

7. **Pass 2 channel-aware retry** (`run_batch.run_pass2`). After all
   wells finish Pass 1, the channel-typical `k` and `P0` distributions
   are known. Wells with poor R² or otherwise-suspect fits are re-fit
   with the channel medians as priors, anchored bounds. Whichever fit
   has lower SSR (Pass 1 vs Pass 2) wins.

8. **Quality gates** (`run_batch.run_quality_gates`). Stamp each well
   with `PASS` / `FAIL` / `INDETERMINATE`. See §7.

9. **Calibration** (`calibration.build_standard_curve` →
   `calibration.apply_calibration`). Fit the `log10(copies) =
   slope * log10(D0) + intercept` curve from the STANDARD wells; apply
   to every well to populate the `Copies_D0` column.

---

## 6. The tiered optimiser

Why a tier system rather than one optimiser: the MAK2 loss surface has
shallow local minima around the true optimum and a few sharp wrong
basins (notably background-absorbs-D0 solutions that look great on R²
but quantify nonsense). No single optimiser handles every well well, and
the slow-but-thorough optimisers cost too much to run on every well.
Tiered escalation tries cheap-and-fast first, escalates only when needed.

The tier tag is recorded in `optimal_params['tier']` and surfaced in the
batch results table so the user can see at a glance which tier produced
which fit.

| Tier | Method | Fires when | Cost (typical) |
|---|---|---|---|
| **T1-Full** | TRF, all 5 params free | Default first attempt | ~50 ms |
| **T1-Fixed** | TRF with background pinned | `fix_background=True` (production default) | ~50 ms |
| **T1.5** | TRF retry with adjusted bounds | Tier 1 R² < threshold AND residual pattern matches a known failure shape (k-stuck-at-bound, plateau overshoot, baseline elbow) | ~50-200 ms |
| **T2-LHS** | TRF with 3D LHS-seeded multi-start | SSR > 1% of squared signal range (suggests local minimum) | ~300-800 ms |
| **T2.5** | TRF with 5D LHS, background unfixed | Tier 2 still failed | ~500-1200 ms |
| **T3-DE** | scipy `differential_evolution`, global stochastic | R² ∈ [0.95, 0.999) after Tier 2.5 | ~1-3 s |
| **T4** | Full pipeline refit with `P0` capped | Fit's plateau exceeds observed plateau | ~50 ms - 3 s |

Notes on the tier transitions:

- **T1.5's pattern-matching block** contains `if`-conditions tagged with
  comments referencing specific Rutledge-dataset wells (X6.R4.2,
  X6.R2.1, X6.R5.4, X5.R1.4). The conditions test residual *shapes*,
  not literal sample names — but the comments honestly record that the
  heuristics were tuned against those wells. **Phase 1 should review
  whether these patterns generalise to other instruments and chemistries
  or are overfit to the qpcR fixtures.** Flagged in CLAUDE.md.
- **T3 (DE) skips below R² = 0.95.** Wells with R² that low are almost
  certainly non-amplifying or pathological; spending 2+ seconds per
  well on DE doesn't recover meaningful additional fits and severely
  slows batch mode.
- **T4 is a post-fit correction**, not a parallel tier. After whichever
  earlier tier wins, `check_plateau_overshoot` runs; if the fit's
  plateau exceeds the data's, the full pipeline reruns with `P0`'s
  upper bound capped at the current fitted `P0`. The `_skip_overshoot_refit`
  flag prevents infinite recursion.
- **Per-fit deadline**: a 20-second wall-clock cap on all tiers
  combined. Ensures no single difficult well can stall the batch.

---

## 7. Quality gates

Five gates evaluated post-fit. Each writes a verdict that combines into
the well's final `Status`: `PASS` (cleared everything), `INDETERMINATE`
(failed a gate but the result is still reported with reduced
confidence), or `FAIL` (hit a hard rejection).

### Gate 0 — R² floor

- **Threshold:** R² ≥ 0.999, relaxed to ≥ 0.997 for late amplifiers
  (when the fit window includes the last data cycle and there's less
  plateau information).
- **Why this threshold:** qPCR fluorescence has noise floors of ~0.5-1%
  of plateau height. R² ≥ 0.999 means the fit's residuals are at the
  noise floor — beyond that, additional optimisation is fitting noise.
  The 0.997 relaxation for late amplifiers reflects that they have
  fewer plateau cycles to constrain `P0` against.

### Gate 2 — Fit-window width

- **Threshold:** at least 12 cycles in the fitted window.
- **Why this threshold:** MAK2 has 5 parameters. With <12 fit points,
  the parameter-to-data ratio approaches 1:2 and the fit is over-
  determined in the wrong direction — many parameter combinations can
  fit equally well. 12 was chosen empirically as the point where
  parameter uncertainty starts to dominate the answer.

### Gate 2b — Sigmoid vs linear

- **Threshold:** MAK2 R² must beat a linear fit on the same window by
  at least 0.04.
- **Why this threshold:** any monotone curve fits both a sigmoid and
  a line reasonably well over a short window; a high R² alone doesn't
  prove the curve is actually a sigmoid. The 0.04 R² gap is the
  empirical discriminator that catches "linear drift dressed up as
  amplification" while not rejecting genuine but noisy late
  amplifiers.

### Gate 3 — Sigmoid shape

- **Threshold:** the fitted curve's second derivative must change sign
  inside the window (i.e. there must be an inflection point present).
- **Why this gate:** a high R² combined with a passing 2b discriminator
  can still hide a degenerate fit where the optimiser landed in a basin
  that's nominally MAK2-shaped but is, in practice, modelling something
  monotone. Requiring an in-window sign change in `d²F/dn²` enforces the
  S-shape geometry.

### Gate 4 — Plateau overshoot (Tier 4)

- **Threshold:** fitted-plateau / observed-plateau ratio ≤ 1.05 (5%
  tolerance), measured on background-subtracted signal.
- **Why this gate:** as discussed in §6 (T4), high R² ≠ correct `P0`.
  The `k * P0` coupling can let the optimiser inflate `P0` substantially
  while the early-cycle exponential fit stays excellent. Catching this
  requires comparing the fitted plateau to the data plateau — what the
  R²-based gates can't see.

---

## 8. Background separation

This is the most important architectural decision in the engine. It
exists because the MAK2 kinetic parameters and the linear background are
**mutually substitutable** in the pre-amplification region of the curve.

### What goes wrong without separation

Joint optimisation of `(D0, k, P0, F_bg_intercept, F_bg_slope)` in one
TRF call routinely converges to a fit with R² > 0.999 where:

- `F_bg_intercept` has absorbed what should be `D0`, sitting tens or
  hundreds of percent away from where the actual baseline regression
  would put it.
- `D0` has compensated by being correspondingly wrong (typically much
  too small).
- The plateau region is fit fine because the optimiser found
  `(k, P0)` combinations that compensate for the wrong `D0`.

The fit *looks* great on every R²-based metric. The quantification
answer is meaningless. This failure mode is silent — neither the
optimiser nor any single quality gate flags it.

### How separation prevents it

Two-stage fitting:

1. **Stage 1**: `pre_estimate_background` runs a plain linear regression
   on the pre-amplification cycles only. There is no kinetic signal in
   this window, so the regression cannot be fooled by `D0` — it
   genuinely sees just the background.

2. **Stage 2**: when `MAK2Optimizer.fit` runs with
   `fix_background=True`, the kinetic optimisation pins
   `(F_bg_intercept, F_bg_slope)` to the Stage 1 values. The
   3-parameter optimisation over `(D0, k, P0)` then has no way to hide
   a wrong `D0` inside the background.

The user can override this with `fix_background=False` for diagnostic
A/B comparison, but production code (`run_batch`, `app.py` batch mode)
always uses the separated path.

### Why this isn't just a regularisation knob

Some regression libraries offer a "soft prior" alternative: penalise
deviations of background from the pre-estimate during joint
optimisation. Why we don't do this:

- The optimal penalty weight depends on per-well noise levels and
  baseline-window length, neither of which is known a priori.
- Soft priors degrade gracefully when the prior is wrong but degrade
  gradefully when it's right too — the joint fit just slightly relaxes
  the prior. We want the prior to be hard, because we know it's right
  (we measured it directly).
- Hard pinning lets the kinetic optimisation use a smaller, tighter
  parameter space; the optimiser is faster and more reliable.

---

## 9. Log-space `D0`

The optimiser fits `log10(D0)`, not `D0`. Three reasons.

### 9.1 Dynamic range

In a typical dilution series, `D0` spans 6+ orders of magnitude (an
undiluted standard might give `D0 ≈ 1e-2` while the 10⁻⁶ dilution gives
`D0 ≈ 1e-8`). Gradient-based optimisers compute steps in linear units;
a step size that's reasonable at `D0 = 1e-2` is hopelessly large at
`D0 = 1e-8`, and vice versa.

### 9.2 Loss-surface geometry

The MAK2 loss surface's curvature with respect to `D0` is approximately
proportional to `1/D0²` (because halving `D0` shifts the entire fit
curve right by one cycle, and the fit error cost depends on cycle-shift
which is linear in `log(D0)`). Optimising in log space gives the
optimiser a roughly-Hessian-equilibrated landscape, where the
recommended step size is similar across the whole range.

### 9.3 Initial-guess robustness

Multi-start strategies (LHS, T1.5 retries) sample initial guesses from
the parameter box. In linear space, uniform sampling of `D0` puts
almost all samples near the upper bound (a uniform sample from
`[1e-8, 1e-2]` has probability < 0.001% of landing below `1e-5`). In
log space, uniform sampling distributes the initial guesses across
orders of magnitude — exactly what's needed to escape local minima
across the full dynamic range.

### What's exposed externally

The public API (`fit()` return values, the `params['D0']` field, the
`Copies_D0` calibration column) all use `D0` in linear space. The
log10 transform is internal to the optimiser. Don't apply additional
transforms in calling code; the engine has already back-transformed.

---

## 10. Stochastic components

### What's stochastic

- **Tier 2 LHS** (`scipy.stats.qmc.LatinHypercube`, 3D): generates
  initial guesses for the multi-start retry. The LHS sequence depends
  on its `seed`.
- **Tier 2.5 LHS** (5D): same as above but includes background.
- **Tier 3 DE** (`scipy.optimize.differential_evolution`): population-
  based global optimiser. Mutation + crossover are stochastic.
- **`_fit_attempt` random initialisation**: when no LHS sample is
  provided and the analytical-seeded path doesn't apply, all 5
  parameters are drawn from `np.random.uniform` within bounds.
- **Bootstrap.py**: residual resampling for kinetic-parameter CI
  (when used; not in the production batch path).
- **calibration.py limited-dilution**: bootstrap CI on the
  conversion factor.

Every other component of the engine is deterministic — the forward
simulator, the analytical bounds derivation, the linear regressions,
the Ct calculation, the quality gates. The stochasticity is confined
to optimisation strategies that genuinely benefit from random
exploration.

### What `MAK2_RANDOM_SEED` controls

`config.RANDOM_SEED` is read once at module import from the
`MAK2_RANDOM_SEED` environment variable. It threads through:

- `optimizer.py` — every stochastic call uses
  `_derive_seed(SITE_OFFSET, attempt=attempt)`, which returns
  `RANDOM_SEED + offset + attempt` (or `None` when `RANDOM_SEED is None`).
  Per-site offsets keep the LHS samplers and per-attempt retry loops
  from accidentally exploring the same parameter-space corners.
- `bootstrap.py` — `BootstrapAnalyzer` falls back to `RANDOM_SEED`
  when no explicit `random_seed` is passed; both the global numpy
  state and the per-iteration `RandomState` are seeded.
- `calibration.py` — limited-dilution bootstrap uses `RANDOM_SEED`
  when set, falls back to a hardcoded 42 when not. (Confidence-
  interval estimators are deterministic-by-default to avoid
  confusing users with bootstrap-noise variation in CI bounds.)

### Why the seeding setup matters

**Production** (`MAK2_RANDOM_SEED` unset): the optimiser is genuinely
stochastic. Replicate fits of the same well show small natural
variation, exposing the optimiser's spread to users. This is honest:
when 5 replicate fits give nearly-identical `D0`, the user knows the
fit is robust; when they spread by 5%, the user knows there's real
ambiguity.

**Testing** (`MAK2_RANDOM_SEED=42`): the engine is byte-reproducible
across runs and machines. The Phase 0.5 regression test relies on
this for its exact-equality assertions on `D0`, `R²`, `Ct`, and `Status`
across the reference plate.

**The seeding pattern is not free determinism for production**: it's
determinism on demand, configurable per-environment. The default
behaviour reflects genuine epistemic uncertainty in the optimiser's
output.

---

## 11. Known limitations

Honest list of things the engine does not handle well today. These are
issues to be aware of, not necessarily things to fix in Phase 1.

### 11.1 k-bounds construction can produce inverted bounds

`mak2_model.estimate_MAK2_params_from_exponential` computes `k_lower`
and `k_upper` independently from different observables; for some wells
in dilution series, `k_lower > k_upper` and the optimiser raises
`ValueError: Invalid bounds for k`. Empirically this trips on 9 of 12
wells in `example_data/Boggy.csv` (F2.1 through F6.1). **A separate
task has been spawned to investigate and fix.** Until the fix lands,
the regression-test fixture in Phase 0.5 will only cover the wells
that fit; capture the fixture *after* the fix so the test suite has
meaningful coverage.

### 11.2 Tier 1.5 pattern blocks are tuned to specific fixtures

The pattern-matching code in T1.5 contains adjustment branches with
comments naming Rutledge-dataset wells (X6.R4.2, X6.R2.1, X6.R5.4,
X5.R1.4). The actual `if`-conditions test residual *shape* (slope
trend, sign of plateau residuals, etc.), but they were calibrated
against those specific wells. Cross-instrument generality has not been
systematically tested. Phase 1's per-well unification work should
review whether to keep, generalise, or replace these blocks.

### 11.3 Single-target fitting only

The engine fits one fluorescence channel per well. Multiplexed plates
are handled by the upstream parser
(`qpcr_data_converter._try_quantstudio_multisheet`) — it pivots each
target into its own per-well dict before fitting. The model itself
does not co-fit two targets simultaneously, even when they share a
well.

### 11.4 Plateau cycles are mandatory for `P0`

`P0` is constrained primarily by the height of the plateau region.
Wells where amplification only just started before the run ended (very
late amplifiers) have no plateau information; the optimiser falls back
to the bounds prior, which gives a much wider `P0` posterior. The fits
still report a numerical `P0`, but its uncertainty is not surfaced in
the result table.

### 11.5 The bootstrap CI doesn't account for fit-window-choice variance

`bootstrap.py` resamples residuals from the chosen fit window to derive
parameter CIs. The CI doesn't account for the fact that the fit window
itself depends on the data (smart-start, adaptive extension); a
different window choice would give a different fit and a different
"correct" CI. For wells where the smart-start choice is borderline,
the reported CI is narrower than the genuine epistemic uncertainty.

### 11.6 Cross-file duplication of per-well preprocessing

The per-well preprocessing pipeline (no-amp pre-check → smart-start →
adaptive window → background pre-estimate → bounds → optimiser call →
Ct → tier classification) is implemented three times: in
`run_batch.run_pass1`, in `app.py`'s batch mode, and in `app.py`'s
single-sample mode. The single-sample path lacks several features
that the batch paths have (no-amp pre-check, metadata baseline-end
handling, ROX normalisation, tier classification, Ct computation,
Pass 2 retry), so single-mode and batch-mode fits of the same curve
are not strictly comparable today. **Phase 1's `fit_well()` extraction
will resolve this**; see the dedicated section in CLAUDE.md.

### 11.7 No first-class uncertainty propagation

The engine reports point estimates for `D0` / `k` / `P0` with R² as
the only universal quality metric. There is no per-well covariance
matrix, no `D0 ± δD0`, no Wald CI from the Hessian. Bootstrap.py can
fill this gap on demand, but it adds ~10× to the per-well cost and is
not in the production batch path.

### 11.8 Performance

Per-well median fit time is ~50-300 ms in production (Tier 1 / 1.5),
extending to ~3 seconds for wells that escalate to Tier 3 DE. A 96-well
plate typically completes in 30-60 seconds; 384-well plates in 2-5
minutes. There is no parallelism — the optimiser is single-threaded by
choice (DE's `workers=1` setting; see `_fit_with_differential_evolution`'s
docstring for why).

---

## Appendix: vocabulary glossary

| Term | Meaning |
|---|---|
| **qPCR** | Quantitative polymerase chain reaction. The wet-lab assay this engine analyses. |
| **Cycle / cycle number** | One round of thermal cycling. A typical run is 40-45 cycles. |
| **RFU** | Relative fluorescence units. The arbitrary-scale output of the instrument's detector. |
| **Rn** | Background-corrected fluorescence: `Rn = (sample fluorescence) / (passive reference fluorescence)`. ABI convention. |
| **ΔRn** | Rn minus the baseline regression: the amplification-only signal. |
| **ROX** | A passive-reference dye that doesn't participate in amplification, used by ABI instruments to normalise per-well loading variation. |
| **Ct (cycle threshold)** | The cycle index at which fluorescence first exceeds a fixed threshold. The output of "Ct-based" qPCR analysis. |
| **Standard curve** | A regression of `log10(copies)` against `Ct` (or `log10(D0)`) from wells with known template concentrations. Used for absolute quantification. |
| **NTC** | No-template control. A well that should not amplify; used to detect contamination. |
| **STANDARD / UNKNOWN / NTC** | The `Task` field values in instrument metadata. STANDARD wells have known quantities and feed the calibration; UNKNOWN wells are samples to quantify. |
| **D0, k, P0** | The MAK2 model's three kinetic parameters. See §2. |
| **F_bg_intercept, F_bg_slope** | The MAK2 model's two background parameters (linear baseline). |
| **Tier (T1, T1.5, …)** | The optimisation strategy that produced a given fit. See §6. |
| **Inflection cycle** | The cycle of maximum slope (steepest growth) on a qPCR sigmoid. Marks the boundary between exponential and plateau regions. |
| **Smart-start** | The algorithm that locates the inflection cycle and sets the fit-window start a few cycles upstream of it. |

---

## References

- Boggy GJ, Woolf PJ. **A mechanistic model of PCR for accurate quantification of quantitative PCR data.** *PLoS ONE* 5(8): e12355 (2010). DOI: [10.1371/journal.pone.0012355](https://doi.org/10.1371/journal.pone.0012355)
- The qpcR R package (source of the Boggy and Rutledge example datasets used in tests): https://cran.r-project.org/package=qpcR
- The live MAK2+ Streamlit app (functional reference for the Phase-1 port): https://mak2-plus.streamlit.app/
