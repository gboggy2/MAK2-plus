# MAK2+ Fitting Workflow

## Table of Contents
1. [Overview](#overview)
2. [The MAK2 Model](#the-mak2-model)
3. [Single-Sample Fitting](#single-sample-fitting)
4. [Batch Fitting](#batch-fitting)
5. [Optimizer Internals](#optimizer-internals)
6. [Single vs Batch: Key Differences](#single-vs-batch-key-differences)

---

## Overview

MAK2+ fits qPCR amplification curves using a mechanistic model with primer depletion (the MAK2 model). The system has two user-facing modes:

- **Single-sample fitting**: Interactive exploration of one well at a time
- **Batch fitting**: Automated fitting of all wells on a plate, with multi-pass retry logic and quality gates

Both modes ultimately call the same `MAK2Optimizer.fit()` method, but they prepare data, set bounds, and handle results differently.

### The Five Model Parameters

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| `D0` | Initial DNA fluorescence (fluorescence units) | 1e-15 to ~500K |
| `k` | PCR characteristic constant (relates to efficiency) | 0.05 to 1.2 |
| `P0` | Initial primer concentration | Scales with fluorescence range |
| `F_bg_intercept` | Background fluorescence (constant component) | ~baseline level |
| `F_bg_slope` | Background fluorescence drift (per-cycle component) | Near zero |

---

## The MAK2 Model

### Core Equation

Each PCR cycle updates DNA concentration:

```
D[n] = D[n-1] + k_eff * ln(1 + D[n-1] / k_eff)
```

where `k_eff = k * P[n-1]` (efficiency scales with remaining primers).

Primers are consumed each cycle: `P[n] = P[n-1] - (D[n] - D[n-1])`

Total fluorescence: `F[n] = D[n] + F_bg_intercept + F_bg_slope * n`

### Two Simulation Modes

- **`simulate_cycles(D0, k, P0, n_cycles, ...)`**: Simulates `n_cycles` from cycle 0. Returns 0-indexed arrays.
- **`simulate_to_cycle(D0, k, P0, cycles, ...)`**: Simulates enough cycles to cover the requested cycle numbers, then extracts values at those specific cycles. Handles non-zero-start data correctly.

---

## Single-Sample Fitting

**Triggered by:** clicking "Fit Model" in the single-sample view.

### Flowchart

```
User clicks "Fit Model"
        |
        v
[1] FLOOR INDEX
    Convert first_fit_cycle (sidebar slider, default 3) to array index
        |
        v
[2] SMART START (inflection search)
    Compute smoothed gradient of fluorescence from floor onward
    Scan right-to-left: find peak where gradient drops to <50% of max
      |-- Peak found? --> inflection = peak index
      |-- No peak?    --> inflection = argmax of smoothed gradient
    fit_start = max(floor_idx, inflection - cycles_before_max)
        |
        v
[3] BACKGROUND PRE-ESTIMATION
    Linear regression on cycles [fit_start - 8, fit_start)
      |-- >= 2 points? --> bg_slope, bg_intercept from regression
      |-- < 2 points?  --> slope=0, intercept=fluorescence[fit_start]
        |
        v
[4] ADAPTIVE WINDOW EXTENSION
    Count baseline cycles in first 6 cycles of fit window
    (baseline = fluorescence <= bg_level + 3*noise)
      |-- >= 3 baseline cycles? --> proceed
      |-- < 3 baseline cycles?  --> extend fit_start backwards
          (increment by 2 cycles, recompute background, repeat)
        |
        v
[5] OPTIMIZER CALL
    optimizer.fit(
        cycles[fit_start:],
        fluorescence[fit_start:],
        bounds = custom_bounds or None,
        fixed_background_values = {slope, intercept},
        verbose = True
    )
        |
        v
[6] POST-FIT WARNINGS (informational only, results NOT nulled)
    - R2 < 0.99? --> show warning
    - Fit window < 8 cycles? --> show warning
    - No sigmoid inflection? --> show warning (skip if late-amp or R2>=0.999)
```

**Important:** Single-mode does NOT reject results. All warnings are informational. The user sees the fit regardless of quality.

---

## Batch Fitting

**Triggered by:** clicking "Batch Fit All Samples."

Batch fitting has three passes plus pre-processing. It can resume from checkpoints if interrupted.

### Phase 0: Signal Detection + Thresholds

```
[0a] DETECT NO-SIGNAL SAMPLES
     Compute max fluorescence range across all samples on plate
     For each sample:
       |-- range < 2% of plate max? --> flag as no-signal
       |-- range 2-5% of plate max? --> borderline: check exponential fit
           |-- exp R2 < 0.80? --> flag as no-signal
           |-- exp R2 >= 0.80? --> keep
       |-- range >= 5%? --> keep
     (Run per-channel if multiple channels selected)
         |
         v
[0b] METADATA RESCUE
     For each flagged sample:
       |-- Instrument reports a Ct (not Undetermined/NOAMP)? --> rescue back to valid
       |-- No instrument Ct? --> stays flagged as no-signal
         |
         v
[0c] PER-CHANNEL Ct THRESHOLDS
     For each channel:
       Compute baseline SDs and means from first 15% of cycles
       threshold = max(10 * median_SD, 0.05 * dynamic_range)
       |-- Instrument thresholds available + ROX active? --> override computed
```

### Pass 1: Initial Fitting

For each sample:

```
[1a] NO-AMPLIFICATION PRE-CHECK
     baseline_SD = std(first min(12, len/4) cycles)
     total_range = max(fluor) - min(fluor)
       |-- range >= 5 * baseline_SD? --> proceed to fitting
       |-- range < 5 * baseline_SD?
           |-- mean(last 5 cycles) > baseline_mean + 2*SD? --> late-amp rescue, proceed
           |-- otherwise? --> "No amplification detected", SKIP
               |
               v
[1b] METADATA BACKGROUND (informational)
     If instrument metadata available:
       Linear regression on Baseline Start -> Baseline End cycles
       Store bg_slope_est, bg_intercept_est
       (Used for Ct and diagnostics, NOT for fitting bounds)
               |
               v
[1c] SMART START (same algorithm as single mode)
     Floor index --> Baseline end anchor --> Right-to-left inflection search
     Baseline end anchor decision:
       |-- algorithmic estimate < 85% of total cycles?
       |       --> use max(metadata_end, algorithmic_end)
       |-- algorithmic estimate >= 85%? (unreliable)
               --> fall back to metadata only
     fit_start = max(floor, inflection - cycles_before_max)
               |
               v
[1d] WINDOW-BASED BACKGROUND
     Linear regression on [fit_start - 8, fit_start)
     This is the background used for FITTING (not metadata)
               |
               v
[1e] ADAPTIVE WINDOW EXTENSION
     Same logic as single mode (extend backward if <3 baseline cycles)
               |
               v
[1f] BOUNDS CONSTRUCTION
     D0:             (1e-15, max(F_range, 1.0))
     F_bg_slope:     bg_slope +/- max(|slope|*0.40, F_range*0.002)
     F_bg_intercept: bg_int +/- max(|int|*0.005, F_range*0.03)
     + any user custom bounds (non-bg bounds only)
               |
               v
[1g] OPTIMIZER CALL
     optimizer.fit(cycles_fit, fluor_fit,
         bounds = merged_bounds,       # includes D0 explicitly
         fixed_background_values = {slope, intercept},
         verbose = False)
               |
               v
[1h] Ct COMPUTATION
     Priority: instrument Ct > ROX-normalized MAK2 Ct > raw MAK2 Ct
     Undetermined/NOAMP from instrument --> force Ct = NaN
               |
               v
[1i] TIER CLASSIFICATION
     T3-DE (differential evolution) | T2-LHS (adaptive fallback) |
     T1-Fixed (fixed background) | T1-Full (5-param fit)
```

### Pass 2: Channel-Aware Retry

```
[2a] LEARN PER-CHANNEL PRIORS
     From Pass 1 "reliable" fits (R2 > 0.95, k < 0.5):
       Compute median k, P0, F_bg_intercept, F_bg_slope per channel
       |-- >= 2 reliable fits? --> use channel medians
       |-- < 2 reliable?      --> use plate-wide fallback medians
               |
               v
[2b] IDENTIFY RETRY CANDIDATES (any of these trigger retry):
     (a) High SSR relative to F_range AND R2 < 0.999
     (b) k is None (optimization failed entirely)
     (c) Degenerate k > 0.5 AND R2 < 0.999
     (d) R2 < 0.999 (always retry imperfect fits)
     (e) Tail overshoot: mean(last 3 residuals) < -3% of F_range, R2 < 0.999

     HOPELESS FILTER: R2 < 0.85 excluded from retry
       |-- UNLESS late amplifier (fit_end near last cycle)
               |
               v
[2c] PER-WELL RETRY (for each candidate):
     Timeout: 10s (late amplifiers) or 30s (normal)

     Build channel-informed bounds:
       bg: prefer per-well estimate (+/-40%), fallback channel median (+/-300%)
       k:  (max(0.05, prior_k*0.20), min(1.0, max(0.5, prior_k*5.0)))
       P0: (max(prior*0.05, F_range*0.01), max(prior*7.0, F_range*2.0))
       D0: (1e-15, F_range*10) initially, then (1e-8, F_range) after bg estimation

     Smart start (same algorithm) with extended truncation:
       cycles_after_max + 3 (exposes more post-inflection data)

     Late-amp enhancement:
       |-- Pass 1 R2 < 0.90 AND late amplifier?
           --> run analytical exponential estimation for tight D0/k priors

     INITIAL RETRY FIT
       |-- R2 >= 0.999? --> accept and stop
       |-- R2 < 0.999?  --> try window variations
               |
               v
     WINDOW VARIATIONS (up to 8 combinations):
       (cbm, cam), (cbm, cam-1), (cbm+4, cam), (cbm+8, cam),
       (cbm, cam+3), (cbm-2, cam+3), (cbm+4, cam+3), (cbm-4, cam)
       Each: re-estimate background, re-fit with new window
       |-- Any hit R2 >= 0.999? --> accept best and stop
       |-- All below?           --> try k relaxation
               |
               v
     K RELAXATION:
       |-- k near lower bound AND still below target?
           --> retry with k bounds (0.001, upper), D0 widened to (1e-15, F_range*10)
               |
               v
     ACCEPTANCE:
       |-- retry R2 > original R2? --> accept retry result
       |-- retry R2 <= original?   --> keep original Pass 1 result
       Labels: "window-retry", "late-amp", "timeout@stage", "R2 below target"
```

### Pass 3: Post-Fit Quality Gates

Applied to ALL results. **These are hard rejections** -- failing wells have D0/k/P0/Ct set to None.

```
For each result:
  |-- Already has error? --> skip
  |
  v
GATE 0: R2 THRESHOLD
  Detect if late amplifier (fit_end >= last_cycle - min(max(1, cycles_after_max), 5))
    |-- Late amplifier?  --> threshold = 0.85
    |-- Normal?          --> threshold = 0.99
  |-- R2 < threshold? --> REJECT: "R2 = X < threshold"
  |
  v
GATE 2: FIT WINDOW WIDTH
  |-- fit_end - fit_start < 10 cycles? --> REJECT: "Fit window N cycles < 10"
  |
  v
GATE 2b: LINEAR vs MAK2 (skip for late amplifiers)
  Compute predictions in pre-inflection region (fit_start to max-slope cycle)
  Fit a straight line to same region
  |-- R2_MAK2 - R2_linear < 0.05? --> REJECT: "MAK2 not better than linear"
  |-- Improvement >= 0.05?         --> pass
  (Requires >= 4 pre-inflection points)
  |
  v
GATE 3: SIGMOID SHAPE (skip for late-amp or R2>=0.999 with window>=10)
  Compute 2nd derivative of model prediction within fit window
  Curvature threshold = 1% of predicted signal range
  |-- Sign change in d2 above threshold? --> pass (has inflection)
  |-- No sign change?                    --> REJECT: "No inflection (monotone curve)"
  |
  v
  PASSED ALL GATES --> result stands
```

---

## Optimizer Internals

### Auto-Truncation

```
fit() called with cycles, fluorescence, options
  |
  v
TRUNCATION DECISION:
  |-- truncate_cycle provided?   --> manual truncation at that cycle
  |-- auto_truncate=True?        --> find_slope_threshold_cycle()
  |      Uses 5-point stencil derivative to find max slope
  |      Truncates at max_slope + cycles_after_max
  |-- Neither?                   --> no truncation (use full data)
  |
  v
cycles_fit, fluorescence_fit = truncated arrays
```

### Bounds & Initial Estimation

```
BOUNDS PATH DECISION (key branch):
  |
  |-- D0 NOT in user-provided bounds? (use_analytical_init = True)
  |     |
  |     v
  |   Try estimate_MAK2_params_from_exponential():
  |     Calls estimate_D0_bounds() internally:
  |       1. Sliding-window baseline detection
  |       2. Fits "perfect doubling" model: F = bg + D0 * 2^n
  |       3. Fits "efficiency" model: F = bg + D0 * E^n
  |       4. D0 bounds from both fits with 10x margin
  |     Then estimates k analytically from MAK2-vs-exponential match
  |     |-- Success? --> tight analytical bounds + initial estimates
  |     |-- Fail?    --> try estimate_D0_bounds() alone
  |         |-- Success? --> D0 bounds only, default k/P0/bg bounds
  |         |-- Fail?    --> fallback defaults from data range
  |
  |-- D0 IN user-provided bounds? (batch mode always does this)
        |
        v
      Try estimate_MAK2_params_from_exponential() for INIT GUESSES only
        |-- Success? --> analytical_estimates available for seed=1
        |-- Fail?    --> try heuristic fallback
            D0 ~= F_range / 2^(midpoint_cycle - first_cycle)
            k = 0.3, P0 = 1.5 * F_range
              |-- Success? --> heuristic estimates for seed=1
              |-- Fail?    --> analytical_estimates = None
                              seed=1 uses random init
```

### Fixed Background

```
fix_background = True? (default)
  |
  v
Source priority:
  1. fixed_background_values from caller (app.py's window regression)
  2. analytical_estimates (if available)
  3. Midpoint of F_bg bounds
  |
  v
Lock F_bg_intercept and F_bg_slope --> fit only D0, k, P0 (3 parameters)
```

### LHS Sampling & Evaluation

```
Generate 20 * (max_attempts - 1) LHS samples in 3D
  D0: log-uniform across full D0 range
  k:  uniform across k range
  P0: uniform across P0 range

Override sample[0]: high-P0, low-k, upper-D0 corner
Override sample[1]: analytical/heuristic estimate (if available)
  |
  v
EVALUATE all samples: forward-simulate with simulate_to_cycle(),
compute SSR against data
  |
  v
RANK by SSR (lowest = best)
Keep top (max_attempts - 1) as starting points for Tier 1
```

### Tier Structure

```
TIER 1: MULTI-START TRF
  20-second global timeout for all tiers
  Attempt 1: analytical estimates (seed=1)
  Attempts 2+: ranked LHS starting points
  Each attempt:
    |-- Stuck? (k,P0 changed < 5%) --> skip, next attempt
    |-- Param at bound + R2 < threshold? --> adjust bounds, retry
    |-- R2 >= threshold + no bound issues? --> ACCEPT, STOP
  |
  v (if R2 < threshold)
TIER 1.5: RESIDUAL PATTERN ANALYSIS
  Divide data into baseline (30%) / elbow (middle) / plateau (20%)
  Detect patterns:
    1. Baseline too low       --> increase F_bg_intercept
    2. All positive           --> shift background
    3. Late transition        --> shift D0/k
    4. Early transition       --> shift D0/k other direction
    5. Plateau saturation     --> increase P0
    6. Plateau overshoot      --> decrease P0
    7. Increasing residuals   --> adjust F_bg_slope
    8. k stuck at lower bound --> relax k lower bound
  Each pattern: 3 targeted retry attempts
  |
  v (if R2 < threshold)
TIER 2: SSR-BASED RETRY
  Fires when SSR > 0.01 * F_range^2
  Low-template path: narrow k, sample D0 from lower range
  Normal path: increase k by 5x, decrease P0 by 0.5-0.7x
  Up to max_attempts retries
  |
  v (if fixed_bg was used AND R2 < 0.999)
TIER 2.5: ADAPTIVE BACKGROUND FALLBACK
  Clear fixed background --> switch to full 5-parameter fit
  Widen bounds (D0*10, k/10, P0*2)
  Generate 40 LHS samples in 5D space
  Evaluate, rank, optimize from top candidates
  |-- Better than fixed-bg result? --> accept
  |-- Worse?                       --> keep fixed-bg result
  |
  v (if 0.95 <= R2 < 0.999)
TIER 3: DIFFERENTIAL EVOLUTION
  Global optimizer (no local minima trapping)
  Wider bounds (D0*20, k/20, P0*3)
  popsize=30, maxiter=200, polish=True (Nelder-Mead refinement)
  |
  v (if model overshoots observed plateau)
TIER 4: PLATEAU OVERSHOOT REFIT
  Cap P0 at current value, re-run full pipeline
  Accept if: overshoot reduced AND R2 >= 0.995
```

### _fit_attempt() Detail

```
Called for each starting point (seed):
  |
  |-- seed=1 AND analytical_estimates available?
  |     D0: analytical +/- 20% noise
  |     k:  analytical +/- 50% noise (or re-ranged if near bounds)
  |     P0: analytical +/- 20% noise
  |
  |-- seed>1 (LHS-guided)?
        D0: from LHS ranking (or random log-uniform)
        k:  from LHS ranking (or random uniform)
        P0: from LHS ranking (or random uniform)
  |
  v
REPARAMETERIZE: D0 --> log10(D0) for numerical stability
  |
  v
WEIGHTING:
  Detect signal start (first point > F_min + 10% of F_range)
  Linear ramp from weight=1 at signal start to plateau_weight_multiplier at end
  sigma = 1/sqrt(weight)  (lower sigma = higher weight)
  |
  v
CURVE_FIT (scipy, TRF method, maxfev=10000):
  |-- Fixed background? --> 3 params: log10(D0), k, P0
  |-- Full?             --> 5 params: log10(D0), k, P0, bg_int, bg_slope
  |
  v
Convert log10(D0) back to D0
Compute R2 on fitted data
Return (params, R2)
```

---

## Single vs Batch: Key Differences

### Summary Table

| Feature | Single | Batch |
|---------|--------|-------|
| **Signal detection** | None | Per-channel detect_no_signal + metadata rescue |
| **No-amp pre-check** | None | Range < 5*SD gate with late-amp rescue |
| **Metadata bg** | Not used | Used for Ct and diagnostics (not fitting) |
| **D0 in bounds** | Only if user specifies | Always (1e-15 to F_range) |
| **Analytical estimation** | Runs for bounds AND init | Runs for init only (bounds bypassed) |
| **Verbose** | True | False |
| **Multi-pass retry** | None | Pass 2: channel priors, window variations, k relaxation |
| **Quality gates** | Warnings only | Hard rejection (results nulled) |
| **Ct computation** | Not in fit path | Per-channel thresholds, ROX normalization |
| **Checkpointing** | None | After every well |
| **Timeout** | Optimizer's 20s default | Optimizer's 20s + retry timeouts (10s/30s) |

### Why These Differences Exist

**Signal detection (batch only):** In single mode, the user chose the well deliberately -- they want to see results regardless. In batch mode, hundreds of wells are processed automatically and many may be empty/negative controls. Automated filtering prevents wasted compute and false positives.

**D0 always in bounds (batch):** Batch mode constructs explicit D0 bounds from the data range `(1e-15, F_range)`. This *bypasses* the optimizer's analytical bounds estimation (which was designed for interactive use where the user might not provide any bounds). The analytical estimation can produce overly narrow D0/k bounds that trap the optimizer in local minima, so batch mode avoids this by providing wide D0 bounds and using the analytical estimation only for initial guesses.

**No retry in single mode:** The user can interactively adjust parameters, change the fit window, or modify bounds. Automated retry would be confusing in an interactive context. Batch mode needs retry because there's no human in the loop.

**Warnings vs hard rejection:** In single mode, showing a bad fit is useful -- the user can diagnose why. In batch mode, a bad fit mixed into hundreds of results is misleading. Hard rejection with error messages ("No amplification detected (R2 = 0.87 < 0.99)") flags problems for downstream analysis.

**Channel-aware priors (batch only):** With hundreds of wells, batch mode can learn what "good" parameters look like per channel (e.g., FAM typically has k~0.25, JOE has k~0.30). These priors help rescue borderline wells. Single mode has no population to learn from.

---

## Appendix: Key Thresholds Reference

| Threshold | Value | Where Used |
|-----------|-------|------------|
| No-signal range | < 2% of plate max | Phase 0 signal detection |
| No-signal borderline | 2-5% of plate max | Phase 0, needs exp R2 < 0.80 |
| Pre-check range/SD | < 5x baseline SD | Pass 1 no-amp pre-check |
| Late-amp rescue (pre-check) | tail > baseline + 2*SD | Pass 1 pre-check |
| Gate 0 R2 (normal) | 0.99 | Pass 3 |
| Gate 0 R2 (late-amp) | 0.85 | Pass 3 |
| Gate 2 window width | 10 cycles | Pass 3 |
| Gate 2b linear improvement | 0.05 | Pass 3 |
| Gate 3 curvature | 1% of signal range | Pass 3 |
| Retry R2 target | 0.999 | Pass 2 |
| Hopeless retry cutoff | R2 < 0.85 | Pass 2 candidate selection |
| Tier 2.5 trigger | R2 < 0.999 | Optimizer (fixed bg only) |
| Tier 3 trigger | 0.95 <= R2 < 0.999 | Optimizer |
| Tier 4 overshoot R2 | >= 0.995 | Optimizer |
| Gradient peak threshold | 5% of gradient range | Smart start |
| Gradient drop-off | 50% of peak | Smart start |
| Optimizer global timeout | 20 seconds | All tiers |
| Retry timeout (late-amp) | 10 seconds | Pass 2 |
| Retry timeout (normal) | 30 seconds | Pass 2 |
