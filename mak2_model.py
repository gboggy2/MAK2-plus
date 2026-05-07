"""MAK2 mechanistic PCR model with primer depletion.

This module implements the forward simulator and the analytical
parameter-estimation pipeline for the MAK2 model of qPCR amplification
described in Boggy & Woolf (2010), PLoS ONE 5(8):e12355, with an
extension that explicitly tracks primer depletion across cycles.

The MAK2 model — at a glance
----------------------------
For cycle ``n``, given DNA fluorescence ``D[n-1]`` and primer pool
``P[n-1]``:

    k_eff      = k * P[n-1]                       # effective per-cycle rate
    D[n]       = D[n-1] + k_eff * ln(1 + D[n-1] / k_eff)
    P[n]       = max(0, P[n-1] - (D[n] - D[n-1])) # primers consumed = DNA produced
    F[n]       = D[n] + F_bg_intercept + F_bg_slope * n

The closed-form per-cycle update is the Michaelis–Menten-like solution
to the differential equation d[D]/dt = k * P * D / (D + k*P), which
captures *why* qPCR curves are sigmoid: when D is small relative to
k*P (early cycles) the rate is approximately k*P*D/(k*P) = D, giving
exponential growth; when D becomes large compared to k*P (after
primer depletion) the rate saturates at k*P, producing the plateau.
The single parameter ``k`` (units of inverse primer concentration)
encodes how aggressively primers are consumed; small k means primers
are abundant relative to template and the curve looks nearly
exponential, large k means depletion dominates and the plateau
arrives quickly.

Why this matters vs. Ct-based methods
-------------------------------------
Ct (cycle-threshold) quantification is essentially a calibration
trick: it locates a fixed-fluorescence threshold and reads off the
cycle number, then converts to copies via a standard curve from
known dilutions. It works, but it requires that standard curve and
it implicitly assumes every well has the same amplification
efficiency. The MAK2 model is mechanistic — fitting it directly
yields D0 (initial template fluorescence), which is proportional to
the absolute number of starting copies. With one well-characterised
"D0_single" calibration constant (template molecules per fluorescence
unit), the same fit gives absolute quantification on any plate
without per-experiment standard curves.

Parameterisation choices that bite you if you don't know about them
-------------------------------------------------------------------
- ``D0`` is in **fluorescence units**, not molecules. The original
  Boggy & Woolf formulation factored the unobservable
  fluorophore-per-template scale out as ``F_scale``; here we fold it
  into D0 directly so D0 is observable. Convert to copies post-hoc
  by dividing by ``D0_single``.
- ``D0`` is fit in **log10 space** by the optimizer (see
  ``optimizer.py``). Linear-space optimization fails because D0
  spans 6+ orders of magnitude across a dilution series; gradient-
  based optimizers can't navigate that landscape without log
  transformation.
- The **background** (``F_bg_intercept``, ``F_bg_slope``) is fit
  separately from the kinetic parameters via two-stage estimation:
  first the pre-amplification cycles are linearly regressed (see
  ``pre_estimate_background``), then those values are passed in as
  near-fixed constants. Letting the optimizer fit background jointly
  with D0/k/P0 leads to the background "absorbing" a wrong D0 — the
  fit looks fine on R² but the kinetic parameters are wildly off.

Public API
----------
``MAK2Model.simulate_cycles``        — forward simulation, fixed-grid output
``MAK2Model.simulate_to_cycle``      — forward simulation, irregular cycles
``calculate_amplification_efficiency``
``find_slope_threshold_cycle``       — locate inflection (max-slope) cycle
``pre_estimate_background``          — linear regression on a baseline window
``estimate_D0_bounds``               — exponential-fit-based D0 bounds for the optimizer
``estimate_k_from_exponential``      — analytic k from observed efficiency
``estimate_MAK2_params_from_exponential`` — full data-driven prior for the optimizer
"""

import numpy as np
from typing import Tuple, Optional


class MAK2Model:
    """Forward simulator for the MAK2 mechanistic PCR model.

    Holds no state; the class exists as a namespace so the simulator can
    be passed around as a single object (see ``optimizer.MAK2Optimizer``,
    which composes one). Each method takes the kinetic parameters
    explicitly and returns a new array — calling the same instance
    repeatedly with different parameters is the normal use pattern.

    The two simulation methods differ only in how the output cycle grid
    is specified:

    - ``simulate_cycles`` returns the full ``[0, n_cycles)`` grid.
    - ``simulate_to_cycle`` returns predictions at an arbitrary array
      of (possibly non-integer, possibly truncated) cycle numbers and
      supports a ``cycle_offset`` for late-amplifying wells.

    See the module docstring for a description of the underlying
    biochemical model.
    """

    def __init__(self):
        # No instance state — kinetic parameters are method arguments.
        # Kept as a class so callers can hold a single ``model`` reference
        # (see ``MAK2Optimizer.__init__``).
        pass
    
    def simulate_cycles(
        self,
        D0: float,
        k: float,
        P0: float,
        n_cycles: int,
        F_bg_intercept: float = 0.0,
        F_bg_slope: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the MAK2 forward simulator on the cycle grid ``[0, n_cycles)``.

        Iterates the per-cycle update rule (see module docstring) and
        adds a linear background, returning DNA-only fluorescence and
        total fluorescence separately so the caller can plot or
        diagnose them independently.

        Args:
            D0: Initial DNA fluorescence at cycle 0, in instrument
                fluorescence units (RFU after baseline subtraction, or
                Rn after ROX normalization). NOT a molecule count — the
                fluorophore-per-template scale is folded in. Convert to
                copies by dividing by the calibration constant
                ``D0_single``.
            k: MAK2 rate constant. Has units of inverse primer
                concentration (matches ``P0``'s units). Encodes the
                trade-off between exponential growth and primer
                depletion. Typical fitted values: 0.05 – 1.5.
            P0: Initial primer pool, in the same fluorescence units as
                ``D0``. The optimizer fits this jointly with k; the
                product ``k * P0`` controls the plateau height.
            n_cycles: Number of cycles to simulate.
            F_bg_intercept: Background fluorescence at cycle 0
                (instrument optical baseline + dye fluorescence).
            F_bg_slope: Per-cycle drift, typically from photobleaching
                or dye degradation. Usually small.

        Returns:
            A 3-tuple ``(cycles, D, F)``:

            - ``cycles``: integer cycle numbers ``0, 1, ..., n_cycles-1``.
            - ``D``: DNA-only fluorescence at each cycle (no background).
            - ``F``: total observed fluorescence (``D`` + linear background).
        """
        # Pre-allocate the cycle, DNA, and primer arrays. We iterate one
        # cycle at a time because the MAK2 update is a recurrence — there
        # is no closed-form for D[n] given D[0], k, P0 (unlike pure
        # exponential growth).
        cycles = np.arange(n_cycles)
        D = np.zeros(n_cycles)
        P = np.zeros(n_cycles)

        D[0] = D0
        P[0] = P0

        for n in range(1, n_cycles):
            # ``k_eff`` is the effective per-cycle rate constant. As the
            # primer pool ``P[n-1]`` depletes, k_eff shrinks, which is
            # what bends the exponential into a plateau.
            k_eff = k * P[n-1]

            # Defensive guards: a depleted primer pool, exhausted DNA,
            # or a numerically pathological combination would put us
            # outside the domain of the analytical MAK2 update. In
            # those cases freeze the state — the cycle made no usable
            # progress.
            if k_eff <= 0 or D[n-1] <= 0:
                D[n] = D[n-1]
                P[n] = P[n-1]
                continue

            # The MAK2 per-cycle update rule:
            #     D[n] = D[n-1] + k_eff * ln(1 + D[n-1] / k_eff)
            # When D[n-1] << k_eff (early cycles, plenty of primer):
            #     ln(1 + x) ≈ x, so D[n] ≈ 2 * D[n-1] (perfect doubling).
            # When D[n-1] >> k_eff (late cycles, primer depleted):
            #     ln(1 + x) ≈ ln(x), so D[n] ≈ D[n-1] + k_eff*ln(D[n-1]/k_eff)
            #     — growth proportional to k_eff, hence the plateau.
            log_arg = 1 + D[n-1] / k_eff
            if log_arg <= 0:
                # Cannot happen for D[n-1] > 0 and k_eff > 0, but we
                # guard against numerical underflow that could push the
                # argument fractionally below zero.
                D[n] = D[n-1]
                P[n] = P[n-1]
                continue

            D_increment = k_eff * np.log(log_arg)
            D[n] = D[n-1] + D_increment

            # Stoichiometry of PCR: every new double-stranded product
            # consumes one primer pair, so the primer pool depletes by
            # exactly the same fluorescence amount that was just added
            # to D. (Both quantities are in the same fluorescence units
            # because we folded ``F_scale`` into D0 — see the parameter
            # discussion in the module docstring.)
            primers_consumed = D_increment
            P[n] = max(0, P[n-1] - primers_consumed)

        # Total observed fluorescence = amplification-derived signal
        # plus the linear instrument background. ``D`` is already in
        # fluorescence units, so this is a straight elementwise add.
        F = D + F_bg_intercept + F_bg_slope * cycles

        return cycles, D, F
    
    def simulate_to_cycle(
        self,
        D0: float,
        k: float,
        P0: float,
        cycles: np.ndarray,
        F_bg_intercept: float = 0.0,
        F_bg_slope: float = 0.0,
        cycle_offset: float = 0.0
    ) -> np.ndarray:
        """Predict fluorescence at an arbitrary array of cycle numbers.

        This is the per-call workhorse used by the optimizer — it takes
        the cycle numbers actually present in the fitted window
        (which may be e.g. 13–26, not 0–N) and returns predicted
        fluorescence aligned with the data.

        Args:
            D0: Initial DNA fluorescence at cycle 0 (fluorescence units).
                See ``simulate_cycles`` for unit discussion.
            k: MAK2 rate constant.
            P0: Initial primer pool, same units as ``D0``.
            cycles: Cycle numbers at which to evaluate the model. Need
                not start at 0 and need not be contiguous; the method
                runs a single forward simulation up to ``max(cycles)``
                and indexes into it.
            F_bg_intercept: Background fluorescence at cycle 0.
            F_bg_slope: Per-cycle background drift.
            cycle_offset: Lag phase. If the well doesn't start
                amplifying until cycle C, set ``cycle_offset=C`` so
                that the simulator treats cycle C as its "n=0" — i.e.
                ``D0`` is the DNA quantity at the start of the
                exponential phase, not at the literal first measurement
                cycle. Cycles before the offset return pure background.

        Returns:
            Predicted fluorescence array, same shape as ``cycles``.
        """
        # Translate input cycle numbers into the simulator's internal
        # frame. Cycles before ``cycle_offset`` clip to 0 — they're in
        # the lag phase and return pure background below.
        effective_cycles = np.maximum(0, cycles - cycle_offset)
        max_cycle = int(np.max(effective_cycles)) + 1

        # Run one forward simulation deep enough to cover every
        # requested cycle, then sample.
        _, _, F_all = self.simulate_cycles(
            D0, k, P0, max_cycle, F_bg_intercept, F_bg_slope
        )

        F_result = np.zeros_like(cycles, dtype=float)
        for i, eff_cyc in enumerate(effective_cycles):
            if eff_cyc < 0:
                # Pre-amplification: instrument sees only background.
                # Use the original (un-offset) cycle so the linear
                # baseline keeps drifting through the lag phase.
                F_result[i] = F_bg_intercept + F_bg_slope * cycles[i]
            else:
                idx = int(eff_cyc)
                if idx < len(F_all):
                    F_result[i] = F_all[idx]
                else:
                    # Defensive: simulator was sized to ``max_cycle``
                    # so this branch should be unreachable. Clamp to
                    # the last simulated value if floating-point
                    # arithmetic ever slips past the end.
                    F_result[i] = F_all[-1]

        return F_result


def calculate_amplification_efficiency(D: np.ndarray) -> np.ndarray:
    """Compute per-cycle PCR efficiency from a DNA-fluorescence array.

    Efficiency at cycle n is defined as the fractional gain over the
    previous cycle:

        E[n] = (D[n] - D[n-1]) / D[n-1]

    With this definition E=1.0 means perfect doubling and E=0 means
    no amplification. (Other parts of the codebase use the alternative
    convention E = D[n]/D[n-1], where E=2.0 is perfect doubling — be
    careful when comparing.) This is a diagnostic function used in
    the UI to plot how efficiency drops across the run as primers
    deplete; it is not used in the fit itself.

    Args:
        D: DNA-only fluorescence per cycle, e.g. the ``D`` returned by
            ``MAK2Model.simulate_cycles``. Must have at least 2 entries.

    Returns:
        Efficiency array of length ``len(D) - 1``. Entry ``i``
        corresponds to the transition from cycle ``i`` to cycle ``i+1``.
        Cycles where ``D[n-1] == 0`` get efficiency 0 to avoid division
        by zero.
    """
    efficiency = np.zeros(len(D) - 1)
    for n in range(1, len(D)):
        if D[n-1] > 0:
            efficiency[n-1] = (D[n] - D[n-1]) / D[n-1]
    return efficiency


def find_slope_threshold_cycle(
    fluorescence: np.ndarray,
    cycles_after_max: int = 3
) -> int:
    """Locate the inflection (max-slope) cycle plus a small offset.

    Used by the optimizer to truncate the fit window: the inflection
    point of the qPCR sigmoid marks the boundary between exponential
    growth and primer-depletion plateau. Fitting only up to a few
    cycles past the inflection avoids the noisy tail (where the
    plateau is sensitive to dye degradation and the model has very
    little residual structure to learn from), while still giving the
    optimizer enough plateau information to constrain ``P0``.

    Uses a 5-point stencil for the first derivative:

        f'[i] = (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2]) / 12

    rather than ``np.gradient``'s 2-point central difference, because
    qPCR data is noisy and the higher-order stencil reduces the
    chance that a single noisy cycle gets picked as the inflection.

    Args:
        fluorescence: Per-cycle fluorescence array (smoothed or raw).
        cycles_after_max: How many cycles past the inflection to
            include before truncating. Default of 3 mirrors the
            optimizer's ``CYCLES_AFTER_MAX`` setting.

    Returns:
        Index into ``fluorescence`` corresponding to the truncation
        cycle. If the array is too short for the 5-point stencil, or
        if no positive slope is detected (flat baseline / failed
        well), returns the last index — i.e. "use everything".
    """
    if len(fluorescence) < 5:
        # 5-point stencil needs ≥5 points; fall back to "use all data".
        return len(fluorescence) - 1

    # Compute the smoothed first derivative. We leave the first two
    # and last two entries as zero (the stencil is undefined at the
    # boundaries); they're excluded from the argmax search below.
    f = fluorescence
    n = len(f)
    f1 = np.zeros(n)
    for i in range(2, n - 2):
        f1[i] = (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2]) / 12.0

    slope = f1

    # Search only the valid interior of the stencil for the peak slope.
    interior_slope = slope[2:-2]
    if len(interior_slope) == 0:
        return len(fluorescence) - 1

    max_slope_idx = np.argmax(interior_slope) + 2  # un-offset
    max_slope = slope[max_slope_idx]

    # A non-positive maximum slope means the curve never goes up — the
    # well failed to amplify. Tell the caller to use the whole array.
    if max_slope <= 0:
        return len(fluorescence) - 1

    cutoff_idx = max_slope_idx + cycles_after_max
    return min(cutoff_idx, len(fluorescence) - 1)


def pre_estimate_background(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    bl_start_idx: int,
    bl_end_idx: int,
) -> tuple:
    """Linear-regress the pre-amplification baseline window.

    Why this function exists separately from the optimizer's joint fit:
    the kinetic parameters (D0, k, P0) and the background
    (F_bg_intercept, F_bg_slope) are mutually substitutable in the
    pre-amplification region — a wrong D0 can be hidden by a
    correspondingly wrong background, leaving R² high while the
    quantification answer is meaningless. The mitigation is to pin
    background down *first* using only baseline cycles (where there is
    no amplification signal to confound it), then pass those values
    in as near-fixed constants to the kinetic optimization. See the
    "Background separation" entry in the module docstring.

    The regression itself is exactly what most qPCR instruments do
    internally for their own baseline correction, so the slope and
    intercept this returns should match (up to baseline-window
    differences) the values reported by the instrument's own
    Ct calculator.

    Args:
        cycles: Full cycle array (cycle numbers — *not* array indices).
        fluorescence: Full per-cycle fluorescence array.
        bl_start_idx: Inclusive start index of the baseline window.
        bl_end_idx: Exclusive end index. The window is
            ``cycles[bl_start_idx:bl_end_idx]``.

    Returns:
        ``(slope, intercept)`` floats. If fewer than 2 points are in
        the window, returns ``(0.0, mean(fluorescence))`` as a
        defensive fallback rather than raising.
    """
    bl_cycles = cycles[bl_start_idx:bl_end_idx].astype(float)
    bl_fluor  = fluorescence[bl_start_idx:bl_end_idx].astype(float)
    if len(bl_cycles) < 2:
        # Degenerate window — return zero drift and the plate-wide
        # average as the intercept so downstream code doesn't crash.
        return 0.0, float(np.mean(fluorescence))
    coeffs = np.polyfit(bl_cycles, bl_fluor, 1)
    return float(coeffs[0]), float(coeffs[1])


def estimate_D0_bounds(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    bg_slope: float = None,
    bg_intercept: float = None,
) -> tuple:
    """Bracket D0 by fitting two exponential models in the early cycles.

    Why this is necessary: the MAK2 optimizer needs box bounds on every
    parameter, but D0 spans 6+ orders of magnitude across a typical
    dilution series — there is no universal default ``[D0_lo, D0_hi]``
    that works on both an undiluted standard and a 1:1e6 dilution.
    Computing per-well bounds from the data itself is what makes the
    full pipeline run unattended on plates with mixed templates.

    Strategy: in the early cycles (before the inflection / before
    primer depletion bites), the MAK2 model is well approximated by
    pure exponential growth. Fit two exponentials over an
    automatically detected exponential region:

    1.  ``F = F_bg + D0 * 2^n``        — perfect doubling.
        This OVER-estimates the rate of growth (real efficiencies are
        always a bit below 2.0), so the fitted D0 *under-estimates*
        the true D0. → **lower bound.**
    2.  ``F = F_bg + D0 * E^n``        — fitted efficiency E ∈ (1, 2).
        E settles to roughly 1.8 in practice. This *over-estimates*
        D0 because the exponential fit absorbs into D0 some of the
        early curvature that the full MAK2 model would attribute to
        primer depletion. → **upper bound.**

    The two fits also yield a fitted background (``F_bg``) which is
    averaged and returned so callers (the optimizer and the UI) have
    a starting point even when the metadata baseline window is
    unavailable.

    Pipeline overview (each step is heavily commented inline below):

    - Detect the baseline-end cycle by scanning for a sustained
      slope+fluorescence increase (5σ above noise).
    - Fit a fresh background regression on all baseline points and
      derive uncertainty bounds (±3σ intercept, ±5σ slope, with a
      50% safety margin) — these become the ``F_bg`` box bounds.
    - Find the inflection cycle by scanning the smoothed gradient
      from the right (avoids early-cycle noise spikes that look like
      growth peaks).
    - Define two cycle ranges within the exponential region:
      ``cycles_lower`` = first few cycles (before primer depletion
      reduces efficiency below 2) for the doubling fit;
      ``cycles_upper`` = the full exponential region up to the
      inflection for the efficiency fit.
    - Fit each exponential with a multi-start adaptive strategy
      (10 initial guesses for doubling, 5 for efficiency) and accept
      the best R².
    - Add 10× margin to each bound so the downstream optimizer has
      slack, and cap the upper bound by the fluorescence range so
      raw ABI multicomponent data (~1e6 RFU) doesn't get clipped to
      the same ceiling as normalized Rn data (~1–3 RFU).

    Args:
        cycles: Per-cycle cycle-number array.
        fluorescence: Per-cycle fluorescence array (raw RFU or
            normalized Rn — the function is scale-aware).
        bg_slope: Optional externally-supplied background slope. When
            provided, it's used for the detrending step that finds
            the amplification onset; the bounds returned for the
            *background itself* are still derived from the local
            baseline regression. Pass this when you have a metadata
            baseline window (more accurate than the function's own
            10-cycle fallback).
        bg_intercept: Companion to ``bg_slope``.

    Returns:
        4-tuple ``(D0_lower, D0_upper, F_bg_estimate, fit_info)``:

        - ``D0_lower``: lower bound on D0 (from doubling fit, with
          10× safety margin applied).
        - ``D0_upper``: upper bound on D0 (from efficiency fit, with
          10× safety margin and a fluorescence-range cap).
        - ``F_bg_estimate``: averaged background intercept from the
          two exponential fits (NOT the same as the regression-only
          intercept — this version "saw" the early exponential).
        - ``fit_info``: dict with everything the UI and the
          downstream ``estimate_MAK2_params_from_exponential`` call
          consume — fitted curves, R², bounds, the efficiency E, the
          detected inflection cycle, and the analytical k estimate.
          Empty dict if the exponential fits failed entirely.
    """
    from scipy.optimize import curve_fit
    import numpy as np
    
    # Find baseline (median of first ~20 cycles or 25% of data)
    # For background-subtracted data, only consider cycles above a minimum threshold
    baseline_n = min(20, max(3, len(fluorescence) // 4))
    
    # IMPROVED: Filter out near-zero values when computing baseline
    # This helps with background-subtracted data
    min_baseline_threshold = 0.001
    baseline_values = fluorescence[:baseline_n]
    baseline_values_filtered = baseline_values[baseline_values > min_baseline_threshold]
    
    if len(baseline_values_filtered) > 0:
        baseline = np.median(baseline_values_filtered)
        print(f"  Baseline computed from {len(baseline_values_filtered)} cycles > {min_baseline_threshold}")
    else:
        # Fallback to all values if nothing above threshold
        baseline = np.median(baseline_values)
        print(f"  Warning: No cycles above {min_baseline_threshold}, using all baseline values")
    
    # SLIDING WINDOW SLOPE DETECTION
    # Fit linear model to windows of 5 points, advance 1 point at a time
    # Detect when slope changes significantly → baseline ended
    
    window_size = 5
    if len(fluorescence) < window_size + 5:
        print("Warning: Not enough data points for sliding window")
        return (1e-4, 1.0, baseline, {})
    
    # Calculate slopes for sliding windows
    # IMPROVED: Only calculate slopes for windows where fluorescence is above threshold
    slopes = []
    window_centers = []
    valid_windows = []
    
    for i in range(len(fluorescence) - window_size + 1):
        window_cycles = cycles[i:i+window_size]
        window_fluor = fluorescence[i:i+window_size]
        
        # Skip window if average fluorescence too low (noisy baseline)
        if np.mean(window_fluor) < min_baseline_threshold:
            continue
        
        # Fit linear model to this window
        coeffs = np.polyfit(window_cycles, window_fluor, 1)
        slope = coeffs[0]
        
        slopes.append(slope)
        window_centers.append(np.mean(window_cycles))
        valid_windows.append(i)
    
    if len(slopes) < 5:
        print(f"  Warning: Only {len(slopes)} valid windows above threshold")
        # Fallback: use conservative default
        baseline_end_idx = min(baseline_n, len(cycles) // 3)
        slopes = np.array([])
    else:
        slopes = np.array(slopes)
        window_centers = np.array(window_centers)
        valid_windows = np.array(valid_windows)
    
    # Detect significant slope change
    # Compare each slope to the median of first few slopes (baseline)
    if len(slopes) > 0:
        baseline_window_count = min(5, len(slopes) // 3)
        baseline_slope_median = np.median(slopes[:baseline_window_count])
        baseline_slope_std = np.std(slopes[:baseline_window_count])
        
        # Use 5-sigma threshold (more conservative than 3-sigma)
        slope_threshold = baseline_slope_median + 5 * baseline_slope_std
        
        # Detrend the data to find amplification onset.
        # Use pre-estimated background when available (from metadata baseline region)
        # — more accurate than fitting only the first 10 cycles internally.
        baseline_fluor_cycles = min(10, len(fluorescence) // 4)

        if bg_slope is not None and bg_intercept is not None:
            # Use externally supplied linear background estimate
            trend_slope     = bg_slope
            trend_intercept = bg_intercept
            fluorescence_detrended = fluorescence - (trend_intercept + trend_slope * cycles)
            baseline_fluor_values  = fluorescence_detrended[:baseline_fluor_cycles]
            baseline_fluor_median  = np.median(baseline_fluor_values)
            baseline_fluor_std     = np.std(baseline_fluor_values)
            fluor_threshold        = baseline_fluor_median + 5 * baseline_fluor_std
        else:
            # Fall back: fit linear trend to first ~10 cycles
            early_cycles = cycles[:baseline_fluor_cycles]
            early_fluor  = fluorescence[:baseline_fluor_cycles]
            if len(early_fluor) >= 2:
                trend_coeffs    = np.polyfit(early_cycles, early_fluor, 1)
                trend_slope     = trend_coeffs[0]
                trend_intercept = trend_coeffs[1]
                fluorescence_detrended = fluorescence - (trend_intercept + trend_slope * cycles)
                baseline_fluor_values  = fluorescence_detrended[:baseline_fluor_cycles]
                baseline_fluor_median  = np.median(baseline_fluor_values)
                baseline_fluor_std     = np.std(baseline_fluor_values)
                fluor_threshold        = baseline_fluor_median + 5 * baseline_fluor_std
            else:
                # Fallback: no detrending
                baseline_fluor_values  = fluorescence[:baseline_fluor_cycles]
                baseline_fluor_median  = np.median(baseline_fluor_values)
                baseline_fluor_std     = np.std(baseline_fluor_values)
                fluor_threshold        = baseline_fluor_median + 3 * baseline_fluor_std
                fluorescence_detrended = fluorescence
        
        # Require 3 consecutive windows above threshold (sustained increase)
        baseline_end_idx = None
        baseline_detected = False
        consecutive_above = 0
        first_above_idx = None
        
        for i in range(baseline_window_count, len(slopes)):
            # Check fluorescence level at this window (use DETRENDED fluorescence)
            window_idx = valid_windows[i]
            window_fluor_mean = np.mean(fluorescence_detrended[window_idx:window_idx+window_size])
            
            # Require BOTH conditions: slope increase AND fluorescence increase
            slope_above = slopes[i] > slope_threshold
            fluor_above = window_fluor_mean > fluor_threshold
            
            if slope_above and fluor_above:
                if consecutive_above == 0:
                    first_above_idx = i  # Remember where this started
                consecutive_above += 1
                
                if consecutive_above >= 3:
                    # Sustained slope increase detected
                    # Use the START of the first window that exceeded threshold
                    # Map back to original window index
                    baseline_end_idx = valid_windows[first_above_idx]
                    baseline_detected = True
                    print(f"  Sliding window slope detection:")
                    print(f"    Baseline slope: {baseline_slope_median:.6f} ± {baseline_slope_std:.6f}")
                    print(f"    Slope threshold (5σ): {slope_threshold:.6f}")
                    print(f"    Baseline fluor (detrended): {baseline_fluor_median:.4f} ± {baseline_fluor_std:.4f}")
                    print(f"    Fluor threshold (5σ): {fluor_threshold:.4f}")
                    print(f"    Sustained increase: windows {first_above_idx}-{i} (3 consecutive)")
                    print(f"    Baseline ends at cycle {baseline_end_idx}")
                    break
            else:
                # Reset if either condition fails
                consecutive_above = 0
                first_above_idx = None
        
        if not baseline_detected:
            # Fallback: use conservative default
            baseline_end_idx = min(baseline_n, len(cycles) // 3)
            print(f"  Warning: No sustained slope change detected, using default cycle {baseline_end_idx}")
    else:
        # No valid slopes computed - use default
        baseline_end_idx = min(baseline_n, len(cycles) // 3)
        print(f"  Warning: No valid slope windows, using default cycle {baseline_end_idx}")
    
    # For background fitting, use MORE cycles than baseline_end_idx
    # Especially for background-subtracted data, we want to fit all the near-zero noise
    # Use cycles up to where real signal starts (first_signal_cycle if available)
    # This gives better background estimate even with negative intercepts
    
    # Calculate fluorescence range first (needed for threshold calculation)
    F_min = fluorescence.min()
    F_max = fluorescence.max()
    F_range = F_max - F_min

    # D0 upper bound must scale with the fluorescence range so that both
    # normalized Rn data (~1–3 RFU) and raw ABI multicomponent data
    # (~10 k – 1 M RFU) can be fitted without the initial guess immediately
    # violating the bound and crashing scipy's curve_fit.
    D0_bound_upper = max(10.0, F_range * 100)
    
    # Find where real signal starts (will be computed later, so estimate it here)
    min_signal_for_bg = max(0.002, F_min + 0.02 * F_range)
    signal_start_idx = np.where(fluorescence > min_signal_for_bg)[0]
    if len(signal_start_idx) > 0:
        bg_fit_end_idx = min(signal_start_idx[0], len(cycles) - 1)
        # But don't use less than baseline_end_idx
        bg_fit_end_idx = max(bg_fit_end_idx, baseline_end_idx)
    else:
        bg_fit_end_idx = baseline_end_idx
    
    # Now fit background using ALL baseline points (cycles 0 to bg_fit_end_idx)
    baseline_cycles_final = cycles[:bg_fit_end_idx]
    baseline_fluor_final = fluorescence[:bg_fit_end_idx]
    
    print(f"  Background fitting: using cycles 0-{bg_fit_end_idx} ({len(baseline_cycles_final)} points)")
    
    # Refit background with all baseline data
    bg_coeffs = np.polyfit(baseline_cycles_final, baseline_fluor_final, 1)
    bg_slope_est = bg_coeffs[0]
    bg_intercept_est = bg_coeffs[1]
    
    # Calculate proper uncertainties using covariance matrix
    bg_pred = np.polyval(bg_coeffs, baseline_cycles_final)
    bg_residuals = baseline_fluor_final - bg_pred
    residual_std = np.std(bg_residuals)
    
    # Calculate uncertainty in slope and intercept
    n = len(baseline_cycles_final)
    x_mean = np.mean(baseline_cycles_final)
    x_var = np.var(baseline_cycles_final)
    
    # Standard error of slope: σ_slope = σ_residual / sqrt(n * var(x))
    slope_uncertainty = residual_std / np.sqrt(n * x_var) if x_var > 0 else residual_std
    
    # Standard error of intercept: σ_intercept ≈ σ_residual * sqrt(1/n + x_mean²/(n*var(x)))
    intercept_uncertainty = residual_std * np.sqrt(1/n + x_mean**2/(n * x_var)) if x_var > 0 else residual_std
    
    # Set bounds with margin to avoid exact boundary hits
    # ±3σ for intercept, ±5σ for slope, plus 50% margin for safety
    # For background-subtracted data, allow negative intercepts
    # Wider margin (50%) prevents "WARNING: at bound!" for noisy baseline data
    margin_factor = 1.5  # 50% wider than pure statistical bounds
    
    intercept_range = 3 * intercept_uncertainty * margin_factor
    bg_intercept_min = bg_intercept_est - intercept_range
    bg_intercept_max = bg_intercept_est + intercept_range
    
    slope_range = 5 * slope_uncertainty * margin_factor
    bg_slope_min = bg_slope_est - slope_range
    bg_slope_max = bg_slope_est + slope_range
    
    # Safety check: ensure bounds are valid (lower < upper)
    # Add minimum separation if bounds are too close
    min_separation = 1e-6
    if bg_intercept_max - bg_intercept_min < min_separation:
        bg_intercept_min = bg_intercept_est - min_separation / 2
        bg_intercept_max = bg_intercept_est + min_separation / 2
        print(f"  Warning: Intercept bounds too tight, widening to [{bg_intercept_min:.6f}, {bg_intercept_max:.6f}]")
    
    if bg_slope_max - bg_slope_min < min_separation:
        bg_slope_min = bg_slope_est - min_separation / 2
        bg_slope_max = bg_slope_est + min_separation / 2
        print(f"  Warning: Slope bounds too tight, widening to [{bg_slope_min:.6f}, {bg_slope_max:.6f}]")
    
    print(f"  Background from baseline fit (cycles {cycles[0]:.0f}-{baseline_cycles_final[-1]:.0f}):")
    print(f"    Intercept: {bg_intercept_est:.4f} ± {intercept_uncertainty:.4f} → bounds [{bg_intercept_min:.4f}, {bg_intercept_max:.4f}]")
    print(f"    Slope: {bg_slope_est:.6f} ± {slope_uncertainty:.6f} → bounds [{bg_slope_min:.6f}, {bg_slope_max:.6f}]")

    # Store baseline_end_cycle for late-baseline detection (used for k bounds constraint)
    baseline_end_cycle = baseline_cycles_final[-1]

    # ── Find the exponential region by scanning from the LAST cycle ──────
    # The inflection (max slope) of the S-curve is the boundary between
    # exponential growth and primer-depletion plateau.  Searching from the
    # right avoids early-cycle noise spikes that can masquerade as growth.
    min_points = 5

    if len(fluorescence) >= 5:
        _raw_grad  = np.gradient(fluorescence)
        _kern      = np.ones(5) / 5.0
        _smooth_g  = np.convolve(_raw_grad, _kern, mode='same')
        # Scan from right to find inflection peak
        _best_val = _smooth_g[-1]
        _best_idx = len(_smooth_g) - 1
        _found_peak = False
        for _j in range(len(_smooth_g) - 2, -1, -1):
            if _smooth_g[_j] > _best_val:
                _best_val = _smooth_g[_j]
                _best_idx = _j
            elif _best_val > 0 and _smooth_g[_j] < _best_val * 0.5:
                _found_peak = True
                break
        inflection_idx = _best_idx if _found_peak else int(np.argmax(_smooth_g))
    else:
        inflection_idx = len(fluorescence) - 1

    print(f"  Inflection (max slope from right): cycle index {inflection_idx} "
          f"(cycle {cycles[inflection_idx]:.0f})")

    # Efficiency region: up to the inflection, starting where fluorescence
    # first rises above the baseline noise level.  Use the baseline median
    # + 5× baseline SD as the onset threshold so we detect real signal,
    # not baseline drift.
    _bl_median = np.median(fluorescence[:baseline_end_idx]) if baseline_end_idx > 0 else baseline
    _bl_sd     = np.std(fluorescence[:baseline_end_idx]) if baseline_end_idx >= 3 else F_range * 0.01
    onset_thresh = _bl_median + 5 * _bl_sd
    # Search only up to the inflection
    onset_candidates = np.where(fluorescence[:inflection_idx + 1] > onset_thresh)[0]
    if len(onset_candidates) > 0:
        exp_start_cycle = max(0, onset_candidates[0] - 1)
    else:
        # No clear onset — start a few cycles before inflection
        exp_start_cycle = max(0, inflection_idx - min_points - 3)

    # End at the inflection (or just past it — efficiency drops there)
    exp_end_upper = inflection_idx

    # Make sure we have enough points
    if exp_end_upper - exp_start_cycle < min_points:
        exp_start_cycle = max(0, exp_end_upper - min_points - 3)

    exp_region_upper = np.arange(exp_start_cycle, exp_end_upper + 1)

    # Perfect doubling: first ~6 cycles of the exponential region
    # (early growth before primer depletion reduces efficiency below 2)
    exp_end_lower = min(exp_start_cycle + min_points + 1, exp_end_upper)
    exp_region_lower = np.arange(exp_start_cycle, exp_end_lower + 1)

    # Ensure minimum points
    if len(exp_region_lower) < min_points:
        exp_region_lower = np.arange(
            max(0, exp_end_upper - min_points),
            min(exp_end_upper + 1, len(cycles)))
    if len(exp_region_upper) < min_points:
        exp_region_upper = np.arange(
            max(0, exp_end_upper - min_points - 3),
            min(exp_end_upper + 1, len(cycles)))
    
    threshold = fluorescence[baseline_end_idx] if baseline_end_idx < len(fluorescence) else baseline
    
    # Extract data for both regions
    cycles_lower = cycles[exp_region_lower]
    fluor_lower = fluorescence[exp_region_lower]
    
    cycles_upper = cycles[exp_region_upper]
    fluor_upper = fluorescence[exp_region_upper]
    
    # Shift cycles so they start at n=0 for numerical stability.
    # Use the start of the exponential region (not cycles[0]) so that
    # 2^n / E^n don't overflow for late-amplifying wells.
    cycle_offset = cycles[exp_region_lower[0]]
    cycles_lower_shifted = cycles_lower - cycle_offset
    cycles_upper_shifted = cycles_upper - cycle_offset

    try:
        # Model 1: Perfect doubling (lower bound)
        # F = (F_bg_intercept + F_bg_slope * n) + D0 * 2^n
        # Background can drift linearly, exponential growth on top
        def perfect_doubling(n, D0, F_bg_intercept, F_bg_slope):
            return (F_bg_intercept + F_bg_slope * n) + D0 * (2.0 ** n)
        
        # Initial guess - D0 is now in fluorescence units
        D0_guess = (F_range / 100) if F_range > 0 else 1e-3
        
        # Adaptive multi-start strategy for perfect doubling fit
        # Target R² ≥ 0.90, max 10 attempts with wide range of D0 guesses
        max_attempts = 10
        r2_threshold = 0.90
        best_r2_lower = -np.inf
        best_params_lower = None
        
        print(f"\n  Perfect Doubling Fit (cycles {cycles_lower[0]:.0f}-{cycles_lower[-1]:.0f}, target R² ≥ {r2_threshold}):")
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Vary D0 initial guess across wide range
                # For late baseline samples, need very small D0 (1e-12 to 1e-15)
                if attempt <= 5:
                    # Normal range
                    D0_init = D0_guess * (0.1 + (attempt-1) * 0.4)  # 0.1x, 0.5x, 0.9x, 1.3x, 1.7x
                else:
                    # Very small D0 for late baseline samples
                    D0_init = D0_guess * (10 ** (-(attempt-5) * 3))  # 1e-3x, 1e-6x, 1e-9x, 1e-12x, 1e-15x
                
                bg_int_init = bg_intercept_est + (attempt - 5) * intercept_uncertainty * 0.5
                bg_slope_init = bg_slope_est + (attempt - 5) * slope_uncertainty * 0.5
                
                params, _ = curve_fit(
                    perfect_doubling,
                    cycles_lower_shifted,
                    fluor_lower,
                    p0=[D0_init, bg_int_init, bg_slope_init],
                    bounds=(
                        [1e-15, bg_intercept_min, bg_slope_min],
                        [D0_bound_upper, bg_intercept_max, bg_slope_max]
                    ),
                    maxfev=5000
                )
                
                # Calculate R²
                pred = perfect_doubling(cycles_lower_shifted, *params)
                ss_res = np.sum((fluor_lower - pred) ** 2)
                ss_tot = np.sum((fluor_lower - np.mean(fluor_lower)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                print(f"    Attempt {attempt}: R² = {r2:.6f}")
                
                # Track best
                if r2 > best_r2_lower:
                    best_r2_lower = r2
                    best_params_lower = params
                
                # Stop if threshold met
                if r2 >= r2_threshold:
                    print(f"    ✓ Threshold met after {attempt} attempt(s)")
                    break
                    
            except Exception as e:
                print(f"    Attempt {attempt}: Failed ({str(e)})")
                continue
        else:
            if best_r2_lower < r2_threshold:
                print(f"    ⚠ Stopped after {max_attempts} attempts (best R² = {best_r2_lower:.6f})")
        
        if best_params_lower is None:
            raise ValueError("All perfect doubling fits failed")
        
        params_lower = best_params_lower
        D0_lower = params_lower[0]
        F_bg_intercept1 = params_lower[1]
        F_bg_slope1 = params_lower[2]
        
        # Generate fitted values for visualization
        fit_lower = perfect_doubling(cycles_lower_shifted, *params_lower)
        
        # Also generate perfect doubling fit extended over efficiency region for comparison
        fit_lower_extended = perfect_doubling(cycles_upper_shifted, *params_lower)
        
        # Model 2: With efficiency E (upper bound)
        # F = (F_bg_intercept + F_bg_slope * n) + D0 * E^n, where E ∈ (1, 2)
        def with_efficiency(n, D0, E, F_bg_intercept, F_bg_slope):
            return (F_bg_intercept + F_bg_slope * n) + D0 * (E ** n)
        
        # Adaptive multi-start strategy for efficiency fit
        # Target R² ≥ 0.99, max 5 attempts
        max_attempts = 5
        r2_threshold = 0.99
        best_r2_upper = -np.inf
        best_params_upper = None
        
        print(f"\n  Efficiency Fit (cycles {cycles_upper[0]:.0f}-{cycles_upper[-1]:.0f}, target R² ≥ {r2_threshold}):")
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Vary D0 and E initial guesses, keep background near estimates
                D0_init = D0_guess * (0.1 + (attempt-1) * 0.4)
                E_init = 1.3 + (attempt-1) * 0.15  # 1.3, 1.45, 1.6, 1.75, 1.9
                bg_int_init = bg_intercept_est + (attempt - 3) * intercept_uncertainty * 0.5
                bg_slope_init = bg_slope_est + (attempt - 3) * slope_uncertainty * 0.5
                
                params, _ = curve_fit(
                    with_efficiency,
                    cycles_upper_shifted,
                    fluor_upper,
                    p0=[D0_init, E_init, bg_int_init, bg_slope_init],
                    bounds=(
                        [1e-15, 1.0, bg_intercept_min, bg_slope_min],
                        [D0_bound_upper, 2.0, bg_intercept_max, bg_slope_max]
                    ),
                    maxfev=5000
                )
                
                # Calculate R²
                pred = with_efficiency(cycles_upper_shifted, *params)
                ss_res = np.sum((fluor_upper - pred) ** 2)
                ss_tot = np.sum((fluor_upper - np.mean(fluor_upper)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                print(f"    Attempt {attempt}: R² = {r2:.6f}")
                
                # Track best
                if r2 > best_r2_upper:
                    best_r2_upper = r2
                    best_params_upper = params
                
                # Stop if threshold met
                if r2 >= r2_threshold:
                    print(f"    ✓ Threshold met after {attempt} attempt(s)")
                    break
                    
            except Exception as e:
                print(f"    Attempt {attempt}: Failed ({str(e)})")
                continue
        else:
            if best_r2_upper < r2_threshold:
                print(f"    ⚠ Stopped after {max_attempts} attempts (best R² = {best_r2_upper:.6f})")
        
        if best_params_upper is None:
            raise ValueError("All efficiency fits failed")
        
        params_upper = best_params_upper
        D0_upper = params_upper[0]
        efficiency = params_upper[1]
        F_bg_intercept2 = params_upper[2]
        F_bg_slope2 = params_upper[3]
        
        # Generate fitted values for visualization
        fit_upper = with_efficiency(cycles_upper_shifted, *params_upper)
        
        # Calculate R² for both fits (on fitted region only)
        # R² = 1 - (SS_res / SS_tot)
        def calculate_r_squared(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        r2_lower = calculate_r_squared(fluor_lower, fit_lower)
        r2_upper = calculate_r_squared(fluor_upper, fit_upper)
        
        # Average the background estimates
        F_bg_intercept = (F_bg_intercept1 + F_bg_intercept2) / 2
        F_bg_slope = (F_bg_slope1 + F_bg_slope2) / 2
        F_bg_estimate = F_bg_intercept  # For compatibility, return intercept
        
        # Ensure proper ordering
        if D0_lower > D0_upper:
            D0_lower, D0_upper = D0_upper, D0_lower
        
        print(f"  D0 from exponential fits:")
        print(f"    Perfect doubling (lower): {D0_lower:.2e}")
        print(f"    Efficiency (upper): {D0_upper:.2e}")

        # Add some margin (10x on each side) - allow D0 to go very low if needed.
        # Upper cap scales with the fluorescence range so raw ABI data (F_range ~1e5)
        # is not arbitrarily capped at 100 RFU.
        D0_lower_bounded = D0_lower / 10
        D0_upper_bounded = min(D0_bound_upper, D0_upper * 10)
        
        print(f"  After adding margin (10x):")
        print(f"    Lower bound: {D0_lower_bounded:.2e}")
        print(f"    Upper bound: {D0_upper_bounded:.2e}")

        # Estimate k from efficiency fit
        try:
            k_estimate = estimate_k_from_exponential(
                D0_upper, efficiency, cycles_upper, P0_assumed=1.0
            )
        except Exception as e:
            k_estimate = None
            print(f"  Warning: k estimation failed ({str(e)})")

        # Store fit info for visualization
        fit_info = {
            'exp_cycles_lower': cycles_lower,
            'exp_fluorescence_lower': fluor_lower,
            'fit_lower': fit_lower,
            'fit_lower_extended': fit_lower_extended,  # Perfect doubling extended to efficiency range
            'exp_cycles_upper': cycles_upper,
            'exp_fluorescence_upper': fluor_upper,
            'fit_upper': fit_upper,
            'efficiency': efficiency,
            'threshold_cycle': cycles[baseline_end_idx] if baseline_end_idx < len(cycles) else cycles[-1],
            'threshold': threshold,
            'r2_lower': r2_lower,
            'r2_upper': r2_upper,
            # Background bounds for MAK2 fitting
            'bg_intercept_min': bg_intercept_min,
            'bg_intercept_max': bg_intercept_max,
            'bg_slope_min': bg_slope_min,
            'bg_slope_max': bg_slope_max,
            # Averaged background for plotting
            'bg_intercept': F_bg_intercept,
            'bg_slope': F_bg_slope,
            # Additional info for analytical MAK2 parameter estimation
            'D0_efficiency': D0_upper,
            'fitted_cycles_efficiency': cycles_upper,
            'baseline_end_cycle': baseline_end_cycle,  # For late-baseline detection
            'k_estimate': k_estimate  # For data-driven k bounds
        }
        
        print(f"\nEstimated D0 bounds from sliding window baseline detection:")
        if baseline_detected:
            print(f"  Baseline ends at cycle {baseline_end_idx}")
        else:
            print(f"  Baseline end at cycle {baseline_end_idx} (fallback: no slope change detected)")
        print(f"  Inflection at cycle index {inflection_idx} (cycle {cycles[inflection_idx]:.0f})")
        print(f"  Perfect doubling fit: cycle {cycles_lower[0]:.0f} to {cycles_lower[-1]:.0f}, R² = {r2_lower:.4f}")
        print(f"    Fitted background: intercept={F_bg_intercept1:.4f}, slope={F_bg_slope1:.6f}")
        
        # Check if parameters hit bounds
        if abs(F_bg_intercept1 - bg_intercept_min) < 0.0001 or abs(F_bg_intercept1 - bg_intercept_max) < 0.0001:
            print(f"    ⚠️  WARNING: Intercept at bound!")
        if abs(F_bg_slope1 - bg_slope_min) < 0.00001 or abs(F_bg_slope1 - bg_slope_max) < 0.00001:
            print(f"    ⚠️  WARNING: Slope at bound!")
            
        print(f"  Efficiency fit: cycle {cycles_upper[0]:.0f} to {cycles_upper[-1]:.0f} (to inflection), R² = {r2_upper:.4f}, E = {efficiency:.2f}")
        print(f"    Fitted background: intercept={F_bg_intercept2:.4f}, slope={F_bg_slope2:.6f}")
        
        # Check if parameters hit bounds
        if abs(F_bg_intercept2 - bg_intercept_min) < 0.0001 or abs(F_bg_intercept2 - bg_intercept_max) < 0.0001:
            print(f"    ⚠️  WARNING: Intercept at bound!")
        if abs(F_bg_slope2 - bg_slope_min) < 0.00001 or abs(F_bg_slope2 - bg_slope_max) < 0.00001:
            print(f"    ⚠️  WARNING: Slope at bound!")
        print(f"  Lower bound: {D0_lower_bounded:.2e} fluorescence units")
        print(f"  Upper bound: {D0_upper_bounded:.2e} fluorescence units")
        print(f"  Background estimate: {F_bg_estimate:.4f}")
        
        return (D0_lower_bounded, D0_upper_bounded, F_bg_estimate, fit_info)
        
    except Exception as e:
        print(f"Warning: Exponential fitting failed ({str(e)}), using default bounds")
        return (1e-4, 1.0, baseline if 'baseline' in locals() else 0.1, {})


def estimate_k_from_exponential(
    D0_eff: float,
    E: float,
    cycles: np.ndarray,
    P0_assumed: float = 1.0,
    use_cycle: Optional[int] = None
) -> float:
    """Solve for the MAK2 rate constant k that matches an observed efficiency.

    The k parameter has no closed-form relationship to the observable
    PCR efficiency E — k controls a continuous tradeoff between
    exponential growth and primer depletion, while E is what the
    well actually showed in the early cycles. This function inverts
    that relationship by:

    1. Picking a representative cycle in the exponential fit window.
    2. Computing the cumulative growth predicted by the efficiency
       exponential at that cycle (``D_prev * (E - 1)``).
    3. Numerically solving for the k that produces the same
       per-cycle growth in the MAK2 equation:

           growth_exp = k * P0 * ln(1 + D_prev / (k * P0))

    The solution is well-defined and unimodal in k (for fixed P0,
    D_prev, growth_exp), so we use ``scipy.optimize.fsolve`` with
    multiple initial guesses and take the median accepted root for
    robustness.

    P0 is **assumed** here (default 1.0) because it cannot be
    disentangled from k using only the exponential region — the
    plateau is what pins down P0 and we haven't gotten there yet.
    The k returned is a starting estimate; the full MAK2 optimization
    refines both jointly.

    Args:
        D0_eff: Fitted D0 from the efficiency exponential model.
        E: Fitted efficiency, conventionally in the range 1.3–1.9 for
            real qPCR data (E=2.0 means perfect doubling).
        cycles: Cycle numbers from the exponential fit window, used
            to pick the representative cycle.
        P0_assumed: Working assumption for the primer pool size; the
            optimizer rescales k after fitting the true P0 to the
            plateau. Default 1.0.
        use_cycle: Optional override for the representative cycle
            (index into ``cycles``). Default behavior is to pick the
            middle of the fit window, which balances numerical
            stability against signal-to-noise.

    Returns:
        Estimated k value. If every numerical solver attempt fails,
        falls back to an empirical scaling (``0.3 * (2 - E)``) — that
        formula has no theoretical justification beyond "small E
        means lots of depletion means large k, and vice versa," but
        it's robust enough to keep the pipeline running.
    """
    from scipy.optimize import fsolve

    # Pick the representative cycle. Mid-window is a balance:
    # earlier cycles have D ≈ D0 (very small), making the equation
    # numerically degenerate; later cycles are more affected by the
    # exponential's slight extrapolation error away from the data.
    if use_cycle is None:
        use_cycle = len(cycles) // 2
    else:
        use_cycle = min(use_cycle, len(cycles) - 1)

    n_cycle = cycles[use_cycle]

    # Reconstruct the "previous-cycle" DNA quantity from the fitted
    # exponential, then compute the per-cycle growth E predicts.
    D_prev = D0_eff * E**(n_cycle - 1)
    growth_exp = D_prev * (E - 1)

    # The function we want to zero: squared difference between the
    # exponential's predicted growth and MAK2's growth at this k.
    # Returns a huge sentinel for non-physical inputs so fsolve walks
    # away from them.
    def equation(k):
        if k <= 1e-8:
            return 1e10
        try:
            mak2_growth = k * P0_assumed * np.log(1 + D_prev / (k * P0_assumed))
            return (growth_exp - mak2_growth)**2
        except:
            return 1e10

    # Multi-start: fsolve is gradient-based and the squared-residual
    # surface has a flat tail at large k. Scattering initial guesses
    # across the realistic k range catches both shallow and steep
    # local-minima geometries.
    k_estimates = []
    for k_init in [0.01, 0.05, 0.1, 0.2, 0.5]:
        try:
            result = fsolve(equation, k_init, full_output=True)
            if result[2] == 1:  # fsolve says it converged
                k_est = result[0][0]
                error = equation(k_est)
                # Only accept genuine roots (squared residual ≈ 0) in
                # a physically reasonable range — otherwise fsolve may
                # have just stopped at a flat region.
                if error < 1e-10 and 1e-6 < k_est < 100:
                    k_estimates.append(k_est)
        except:
            continue

    if not k_estimates:
        # Empirical fallback when the numerics fail entirely: a linear
        # interpolation between "E ≈ 2 → no depletion → k ≈ 0" and
        # "E ≈ 1 → heavy depletion → k ≈ 0.3". Coarse but it keeps the
        # downstream optimizer from getting None.
        k_estimate = 0.3 * (2.0 - E)
        print(f"    Warning: Numerical solution failed, using empirical estimate")
    else:
        # Median across initial guesses suppresses any single bad
        # convergence; with this many starts a true root will be hit
        # multiple times.
        k_estimate = np.median(k_estimates)

    return k_estimate


def estimate_MAK2_params_from_exponential(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    P0_assumed: float = 1.0,
    verbose: bool = True
) -> Tuple[dict, dict]:
    """Build a data-driven prior (point estimates + box bounds) for the MAK2 optimizer.

    This is the function the tiered optimizer calls to obtain its
    starting point. It composes the lower-level primitives:

    1. Run ``estimate_D0_bounds`` to get bracket estimates of D0 from
       perfect-doubling and efficiency exponential fits, plus
       background bounds and the fitted efficiency E.
    2. Use the efficiency-fit D0 as the point estimate.
    3. Call ``estimate_k_from_exponential`` to invert E into a
       starting k (with P0 assumed).
    4. Set P0's point estimate to ``F_max`` — empirically, the
       primer pool that produces a given plateau scales linearly with
       the observed plateau across instruments and dye chemistries.
    5. Apply the **cycle-offset correction** to D0 (see inline
       comments below) so the bounds are expressed in the MAK2
       model's "DNA at cycle 0" reference frame, not the
       exponential-fit's "DNA at the start of the exponential
       region" frame. Without this correction, a late-amplifying
       well (exponential region starts at cycle 25) would have D0
       bounds that are 2^25 ≈ 33 million times too high.
    6. Set k bounds: lower = ``max(0.01, k_estimate / 2)``;
       upper from the empirically-fit relationship
       ``k_upper = 0.2 - 0.03 * log10(D0_upper)``, clipped to
       ``[0.3, 2.0]``. The negative D0–k correlation comes from the
       fact that high-D0 wells reach the plateau quickly (short
       exponential phase, minimal observable depletion → small k
       range explored), while low-D0 wells take many cycles to
       plateau (long exponential phase, cumulative depletion visible
       → large k range needed). The fixed ``[0.3, 2.0]`` clip
       prevents pathological bounds when D0 is itself extreme.
    7. Cap k_upper at 0.85 for late-baseline wells (baseline ends
       at cycle ≥21) to suppress an empirically observed oscillation
       in the optimizer when k drifts above ~1.0.

    Args:
        cycles: Per-cycle cycle-number array.
        fluorescence: Per-cycle fluorescence array.
        P0_assumed: Working P0 used during the analytical k estimation
            (passed to ``estimate_k_from_exponential``). The bounds
            returned for the *real* P0 are derived independently from
            ``F_max``. Default 1.0.
        verbose: Print step-by-step diagnostics. The optimizer passes
            ``True`` so per-well diagnostics appear in the batch log.

    Returns:
        ``(estimates, bounds)`` two-tuple of dicts with keys
        ``D0``, ``k``, ``P0``, ``F_bg_intercept``, ``F_bg_slope``.
        ``estimates`` values are scalars; ``bounds`` values are
        ``(lo, hi)`` pairs for the optimizer's box constraints.

    Raises:
        ValueError: If the upstream exponential fit failed entirely
            (returns an empty ``fit_info``). The optimizer treats
            this as a fall-through to its non-analytical bounds.
    """
    
    if verbose:
        print("\n=== Analytical MAK2 Parameter Estimation ===")
    
    # Step 1: Get exponential fits (this already exists and works well!)
    D0_lower, D0_upper, F_bg_est_scalar, fit_info = estimate_D0_bounds(cycles, fluorescence)

    # Check if exponential fitting succeeded (fit_info will be empty dict if it failed)
    if not fit_info or 'D0_efficiency' not in fit_info:
        raise ValueError("Exponential fitting failed - cannot estimate MAK2 parameters analytically")

    # Reconstruct F_bg dictionary from fit_info (estimate_D0_bounds returns scalar for compatibility)
    F_bg_est = {
        'intercept': fit_info['bg_intercept'],
        'slope': fit_info['bg_slope'],
        'SE_intercept': 0.0,  # Not used in this function
        'SE_slope': 0.0  # Not used in this function
    }

    # Extract efficiency exponential results
    D0_eff = fit_info['D0_efficiency']
    E = fit_info['efficiency']
    fitted_cycles = fit_info['fitted_cycles_efficiency']

    # ── Prepare cycle-offset correction (applied to D0 bounds later) ───
    # estimate_D0_bounds fits exponentials on *shifted* cycles (exp region
    # starts at n=0), so D0_lower/D0_upper represent DNA at the start of
    # the exponential region.  But the MAK2 model's simulate_to_cycle
    # uses *unshifted* cycle numbers with cycle_offset=0, so its D0 is
    # the DNA quantity at cycle 0.  For a well whose exponential region
    # starts at actual cycle C, D0_model ≈ D0_shifted / 2^C.
    # Without this correction, late amplifiers (C ≈ 20-30) have D0 bounds
    # that are millions of times too high.
    # NOTE: The correction is deferred until after k bounds are computed,
    # because the k_upper formula was calibrated against shifted-space D0.
    _exp_start = float(fit_info.get('exp_cycles_lower', [cycles[0]])[0])
    # MAK2 model's D0 is at cycle 0 (simulate_to_cycle uses cycle_offset=0),
    # so the shift is from cycle 0 to the exponential region start — NOT from
    # the first data cycle.  When the app passes a truncated fit window
    # (e.g. cycles 23-45), cycles[0]=23, but D0 still refers to cycle 0.
    _cycle_shift = _exp_start   # shift from model's cycle 0
    _d0_scale_down = 1.0
    if _cycle_shift > 0 and E > 1.0:
        _d0_scale_down = 2.0 ** _cycle_shift
        if verbose:
            print(f"  Cycle-offset: exp starts at cycle {_exp_start:.0f}, "
                  f"shift={_cycle_shift:.0f}, scale={_d0_scale_down:.2e} (deferred)")

    if verbose:
        print(f"\nExponential Fit Results:")
        print(f"  D0 = {D0_eff:.2e}")
        print(f"  Efficiency E = {E:.4f}")
        print(f"  Background: {F_bg_est['intercept']:.4f} + {F_bg_est['slope']:.6f} × n")
    
    # Step 2: D0 estimate (cycle-offset correction applied later)
    D0_estimate = D0_eff
    
    # Step 3: Solve for k analytically
    if verbose:
        print(f"\nSolving for k (assuming P0 = {P0_assumed})...")
    
    k_estimate = estimate_k_from_exponential(
        D0_eff, E, fitted_cycles, P0_assumed
    )
    
    if verbose:
        print(f"  k estimate: {k_estimate:.6f}")
    
    # Step 4: P0 estimate based on maximum fluorescence
    # In MAK2, P0 represents primer concentration and should scale with fluorescence
    # Empirically, P0 ~ F_max works well across different fluorescence scales
    F_max = np.max(fluorescence)
    F_min = np.min(fluorescence)
    F_range = F_max - F_min
    
    # P0 estimate: use max fluorescence as direct proxy
    P0_estimate = F_max
    
    if verbose:
        print(f"\nStep 4: P0 Estimate")
        print(f"  F_max: {F_max:.4f}, F_min: {F_min:.4f}")
        print(f"  P0 estimate: {P0_estimate:.2f} (= F_max)")
    
    # Step 5: Create tight bounds (±1 order of magnitude)
    estimates = {
        'D0': D0_estimate,
        'k': k_estimate,
        'P0': P0_estimate,
        'F_bg_intercept': F_bg_est['intercept'],
        'F_bg_slope': F_bg_est['slope']
    }
    
    # Ensure minimum margins for F_bg parameters to avoid collapsed bounds
    F_bg_int_margin = max(
        3 * F_bg_est['SE_intercept'],  # SE-based margin
        0.1 * abs(F_bg_est['intercept']),  # 10% of value
        0.05  # Absolute minimum
    )
    F_bg_slope_margin = max(
        5 * F_bg_est['SE_slope'],  # SE-based margin
        0.001  # Absolute minimum
    )

    # Use data-driven bounds for D0 and P0, hybrid approach for k
    # D0 bounds come from estimate_D0_bounds: perfect doubling fit (lower) and efficiency fit (upper)
    #
    # k bounds: HYBRID DATA-DRIVEN (adaptive lower, fixed upper)
    # - Lower bound: k_estimate/2 (data-driven, adapts to each sample)
    #   - Accounts for k_estimate underestimating true k
    #   - Provides instrument-independent adaptation
    # - Upper bound: 1.2 (fixed empirically-validated maximum)
    #   - Represents realistic qPCR upper limit (~55% primer depletion per cycle)
    #   - Prevents optimizer from exploring unrealistic parameter space
    #   - Tested: fully data-driven upper (k_est*100) degraded performance to 80.8%
    #
    # For F_bg_slope, use the constrained bounds from estimate_D0_bounds (which includes
    # late-baseline detection and slope constraint) instead of margin-based bounds

    # Calculate data-driven k bounds (both adaptive based on D0)
    if k_estimate is not None:
        k_lower = max(0.01, k_estimate / 2)

        # k_upper based on D0: strong negative correlation (r=-0.79, R²=0.63)
        # High D0 → low k (short exp phase, minimal observable depletion)
        # Low D0 → high k (long exp phase, cumulative depletion visible)
        # Formula: k_upper = 0.2 - 0.03 * log10(D0), clipped to [0.3, 2.0]
        log_D0 = np.log10(D0_upper)  # Use D0 from efficiency fit
        k_upper_D0 = 0.2 - 0.03 * log_D0
        k_upper = np.clip(k_upper_D0, 0.3, 2.0)  # Reasonable bounds

        # Guard against inconsistent analytical bounds.  When k_estimate is
        # unphysically large (e.g. dilution-series wells where the linear-fit
        # k underestimate spikes to >0.6) AND D0_upper is large enough to clip
        # k_upper to its 0.3 floor, k_lower can exceed k_upper and the
        # downstream optimizer rejects the bounds.  In that case the
        # analytical estimates disagree, so fall back to the wider
        # non-analytical range — preserving the optimizer's ability to
        # actually run instead of trusting a contradictory tight box.
        if k_lower >= k_upper:
            if verbose:
                print(f"  ⚠ Analytical k bounds inconsistent "
                      f"(k_lower={k_lower:.4f} ≥ k_upper={k_upper:.4f}); "
                      f"falling back to (0.05, 1.2)")
            k_lower = 0.05
            k_upper = 1.2
    else:
        k_lower = 0.05
        k_upper = 1.2  # Fallback if k_estimate unavailable

    # ── Apply deferred D0 cycle-offset correction ───────────────────
    # Now that k bounds are computed (using shifted-space D0), apply the
    # correction so D0 bounds match the MAK2 model's cycle-0 reference.
    if _d0_scale_down > 1.0:
        D0_lower /= _d0_scale_down
        D0_upper /= _d0_scale_down
        D0_estimate /= _d0_scale_down
        D0_lower = max(D0_lower, 1e-18)
        D0_upper = max(D0_upper, D0_lower * 10)
        D0_estimate = max(D0_estimate, D0_lower)
        estimates['D0'] = D0_estimate
        if verbose:
            print(f"  D0 cycle-offset correction applied (÷{_d0_scale_down:.2e}):")
            print(f"    Bounds: [{D0_lower:.2e}, {D0_upper:.2e}], estimate: {D0_estimate:.2e}")

    bounds = {
        'D0': (D0_lower, D0_upper),  # Cycle-offset–corrected bounds
        'k': (k_lower, k_upper),  # Fully data-driven k bounds
        'P0': (P0_estimate * 0.5, P0_estimate * 2),  # Tight: 0.5x to 2x of F_max
        'F_bg_intercept': (
            fit_info['bg_intercept_min'],  # Use fit_info bounds directly
            fit_info['bg_intercept_max']
        ),
        'F_bg_slope': (
            fit_info['bg_slope_min'],  # Use constrained bounds from estimate_D0_bounds
            fit_info['bg_slope_max']   # (includes late-baseline positive slope constraint)
        )
    }

    # For late-baseline samples, cap k_upper to prevent oscillation.
    # When baseline extends to cycle 21+, optimizer tends to oscillate with
    # very high k (1.2+).  Capping the upper bound at 0.85 helps.
    # IMPORTANT: Do NOT raise k_lower — very late amplifiers (like Rutledge
    # X6 wells) can have k as low as 0.10, and the data-driven k_lower
    # computed above (max(0.01, k_estimate/2)) is already appropriate.
    baseline_end_cycle = fit_info.get('baseline_end_cycle', 0)
    if baseline_end_cycle >= 21:
        old_k_bounds = bounds['k']
        bounds['k'] = (old_k_bounds[0], min(old_k_bounds[1], 0.85))  # Only cap upper
        if verbose:
            print(f"    → Capping k upper for late baseline: {old_k_bounds} → {bounds['k']}")

    # Note: We don't constrain P0 based on plateau here because:
    # 1. Plateau = P0/(1+k*P0) + F_bg_intercept depends on both P0 and k
    # 2. We don't know k before fitting, so using k_min is too conservative
    # 3. Hard constraints degrade performance significantly
    # Instead, the optimizer should prefer fits where model doesn't exceed data

    if verbose:
        print(f"\nData-Driven Bounds:")
        print(f"  D0: [{bounds['D0'][0]:.2e}, {bounds['D0'][1]:.2e}]  (from perfect doubling and efficiency fits)")
        print(f"  k:  [{bounds['k'][0]:.6f}, {bounds['k'][1]:.6f}]  (fixed realistic range)")
        print(f"  P0: [{bounds['P0'][0]:.2f}, {bounds['P0'][1]:.2f}]  (0.5× to 2× F_max)")
        print(f"  F_bg_slope: [{bounds['F_bg_slope'][0]:.6f}, {bounds['F_bg_slope'][1]:.6f}]")
    
    return estimates, bounds


