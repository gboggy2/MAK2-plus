"""Data processing utilities for qPCR plates.

Two helpers consumed by the per-well pipeline (``run_batch.py`` and the
Streamlit ``app.py``):

- ``estimate_baseline_end`` — algorithmic discovery of where the
  pre-amplification baseline ends, used as a fallback / sanity-check
  against the instrument-reported ``Baseline End`` from metadata.
- ``detect_no_signal_samples`` — plate-wide triage: identify wells
  with no real amplification (NTC controls, failed reactions) so the
  expensive MAK2 fit is not run on them.

Neither function knows anything about the MAK2 model itself; both
work directly on the raw fluorescence trace.
"""

import numpy as np
from typing import Tuple, Dict



def estimate_baseline_end(cycles, fluorescence, first_cycle_idx=2, window_size=12, n_sd=5, max_iter=3):
    """Locate the last cycle of the pre-amplification baseline.

    Iterative forward-projection: fit a line through the current
    baseline window, then look for the first cycle whose fluorescence
    exceeds the projected line by more than ``n_sd`` standard
    deviations of the in-window residuals. That cycle becomes the new
    baseline end; the procedure repeats up to ``max_iter`` times to
    refine.

    The 5σ default is deliberately conservative — qPCR baseline noise
    is typically ~0.5-1% of plateau fluorescence, so 5σ catches the
    onset of amplification reliably without false positives from
    single-cycle noise spikes. Lower thresholds occasionally trigger
    on early-cycle outliers.

    Used in two places: as the algorithmic baseline-end estimate that
    the per-well pipeline compares against the instrument's metadata
    value, and (in the Streamlit app) as the value reported in the
    ``bl_end_est`` result column for diagnostic display.

    Args:
        cycles: Per-cycle cycle-number array.
        fluorescence: Per-cycle fluorescence array (raw RFU or Rn).
        first_cycle_idx: 0-based index of the first cycle to include
            in the baseline window. Default 2 skips the first two
            cycles, which on most instruments are unreliable
            (initial dye equilibration).
        window_size: Initial size of the baseline window in cycles.
            Default 12 is large enough to give a meaningful linear
            regression but small enough that it won't accidentally
            contain amplification on early-amplifying wells.
        n_sd: Threshold in baseline standard deviations.
        max_iter: Maximum refinement passes. The procedure usually
            converges in 1–2 iterations; the cap exists only to
            guarantee termination.

    Returns:
        0-based index ``i`` such that ``cycles[:i]`` is the baseline
        region (i.e. ``i`` is exclusive — it points at the first
        amplifying cycle). Returns ``min(first_cycle_idx + window_size,
        n - 1)`` if no clear amplification is detected.
    """
    n = len(cycles)
    bl_start = first_cycle_idx
    bl_end = min(bl_start + window_size, n - 1)
    for _ in range(max_iter):
        bl_c = cycles[bl_start:bl_end]
        bl_f = fluorescence[bl_start:bl_end]
        if len(bl_c) < 3:
            # Window collapsed below the minimum needed for meaningful
            # regression — bail out with whatever we have.
            break
        coeffs = np.polyfit(bl_c, bl_f, 1)
        residuals = bl_f - np.polyval(coeffs, bl_c)
        # Floor the SD so a perfectly flat baseline (residuals ≈ 0)
        # doesn't make the threshold infinitely sensitive.
        sd = max(float(np.std(residuals)), 1e-10)
        new_bl_end = bl_end
        for i in range(bl_end, n):
            if fluorescence[i] > np.polyval(coeffs, cycles[i]) + n_sd * sd:
                new_bl_end = i
                break
        if new_bl_end == bl_end:
            # Converged: no further extension reduces the window's
            # purity claim.
            break
        bl_end = new_bl_end
    return bl_end


def detect_no_signal_samples(
    cycles: np.ndarray,
    all_samples: Dict[str, np.ndarray],
    min_range_pct: float = 2.0,
    min_r2: float = 0.80,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], dict]:
    """Triage a plate, separating wells that amplified from wells that didn't.

    Why this exists: the MAK2 optimizer is expensive (~0.3-1 s per
    well, longer for difficult fits) and produces meaningless
    parameters when run on a flat NTC trace or a failed well. A
    plate-wide screen up front saves time and prevents the result
    table from being polluted by garbage fits that pass R² gates by
    accident.

    Three-stage classification:

    1. **Range-only rejection.** Compute fluorescence range for every
       well; reject wells whose range is below ``min_range_pct`` of
       the plate's largest range. The relative comparison adapts
       automatically to instrument scale (Rn, RFU, normalized) and
       template concentration without per-plate tuning.

    2. **Exponential-fit screening.** For surviving wells, fit the
       efficiency exponential (via ``estimate_D0_bounds``) and reject
       wells whose R² falls below ``min_r2``, *unless* the
       fluorescence range is >10% of the plate max — in that case the
       well clearly amplified and a poor exponential fit just means
       the curve is unusual (background-subtracted data, very late
       amplifiers, or oddly shaped sigmoids). Substantial signal
       overrides poor fit.

    3. **Linear-vs-exponential comparison.** A linearly drifting
       baseline can produce a high range and a passable exponential
       R² simply because any monotone curve is locally well
       approximated by either model. Reject wells where the
       exponential R² isn't at least 0.05 better than a pure linear
       fit and the range is <10% of plate max — those wells are drift,
       not amplification.

    Args:
        cycles: Per-cycle cycle-number array, shared across all wells.
        all_samples: ``{sample_name: fluorescence_array}``. The
            fluorescence arrays must all have the same length as
            ``cycles``.
        min_range_pct: Range threshold as percent of plate max for
            stage 1 rejection. Default 2.0 catches obviously flat
            wells without rejecting genuine low-template amplifiers.
        min_r2: R² threshold for stage 2 rejection. Default 0.80 is
            permissive — the goal here is to filter NTCs, not to
            assess fit quality. The downstream MAK2 quality gates do
            the real assessment.
        verbose: Print per-well diagnostics. Set False for unit tests.

    Returns:
        ``(valid_samples, no_signal_samples, plate_stats)``:

        - ``valid_samples``: ``{name: fluorescence_array}``, ready to
          pass to the MAK2 optimizer.
        - ``no_signal_samples``: ``{name: diag_dict}`` where the
          diag_dict has a ``'reason'`` string and a ``'check_type'``
          tag identifying which stage made the rejection
          (``'range_only'``, ``'fit_failed'``, ``'poor_fit'``,
          ``'linear_drift'``, ``'error'``).
        - ``plate_stats``: dict of plate-wide aggregates used in the
          UI summary (max range, thresholds, sample counts).
    """
    from mak2_model import estimate_D0_bounds

    # Step 1: Calculate plate-wide statistics
    all_ranges = {}
    all_max_fluor = {}

    for name, fluor in all_samples.items():
        F_range = fluor.max() - fluor.min()
        all_ranges[name] = F_range
        all_max_fluor[name] = fluor.max()

    max_range_on_plate = max(all_ranges.values())
    max_fluor_on_plate = max(all_max_fluor.values())

    # Calculate thresholds
    range_threshold = max_range_on_plate * (min_range_pct / 100.0)
    borderline_threshold = range_threshold * 2.5  # 2.5x = borderline zone

    if verbose:
        print(f"\n=== Plate-wide Signal Detection ===")
        print(f"Max fluorescence range on plate: {max_range_on_plate:.4f}")
        print(f"No-signal threshold ({min_range_pct}% of max): {range_threshold:.4f}")
        print(f"Borderline zone: {range_threshold:.4f} - {borderline_threshold:.4f}")

    # Step 2: Classify samples by fluorescence range
    obvious_no_signal = {}
    borderline_samples = {}
    clear_signal_samples = {}

    for name, fluor in all_samples.items():
        F_range = all_ranges[name]
        F_range_pct = (F_range / max_range_on_plate) * 100

        if F_range < range_threshold:
            # Clearly no signal
            obvious_no_signal[name] = {
                'reason': f'Fluorescence range too small ({F_range:.4f}, {F_range_pct:.1f}% of max)',
                'F_range': F_range,
                'F_range_pct': F_range_pct,
                'check_type': 'range_only'
            }
        else:
            # Needs exponential vs linear check
            # (both borderline and clear signal samples)
            borderline_samples[name] = fluor

    # Step 3: For all remaining samples, check exponential fit quality
    # This includes both borderline (2-5% range) and seemingly clear (>5% range)
    # because linear drift can fool range-based detection
    if borderline_samples:
        if verbose:
            print(f"\n{len(borderline_samples)} samples - checking exponential vs linear fit...")

        for name, fluor in borderline_samples.items():
            F_range = all_ranges[name]
            F_range_pct = (F_range / max_range_on_plate) * 100

            try:
                D0_lower, D0_upper, _, fit_info = estimate_D0_bounds(cycles, fluor)

                if not fit_info or 'D0_efficiency' not in fit_info:
                    # Fit failed - likely no signal
                    obvious_no_signal[name] = {
                        'reason': f'Exponential fit failed (range {F_range:.4f}, {F_range_pct:.1f}% of max)',
                        'F_range': F_range,
                        'F_range_pct': F_range_pct,
                        'check_type': 'fit_failed'
                    }
                    continue

                # Get R² from efficiency fit (r2_upper is the efficiency exponential R²)
                r2_exp = fit_info.get('r2_upper', 0)
                D0 = fit_info['D0_efficiency']
                efficiency = fit_info.get('efficiency', None)

                # Check 1: Poor exponential fit R²
                # BUT: if the range is substantial (>10% of plate max),
                # the well clearly has signal regardless of exponential
                # fit quality.  Background-subtracted data and unusual
                # curve shapes can cause poor exponential R² even for
                # wells with obvious amplification.
                if r2_exp < min_r2 and F_range_pct < 10.0:
                    obvious_no_signal[name] = {
                        'reason': f'Poor exponential fit (R²={r2_exp:.3f}, range {F_range_pct:.1f}% of max)',
                        'F_range': F_range,
                        'F_range_pct': F_range_pct,
                        'R2_exp': r2_exp,
                        'D0': D0,
                        'check_type': 'poor_fit'
                    }
                    continue

                # Check 2: Linear vs Exponential comparison
                # Fit simple linear model: F = a + b*cycle
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(cycles, fluor)
                r2_linear = r_value ** 2

                # If linear fit is nearly as good as exponential, it's just drift
                # Use threshold: exponential must be at least 0.05 R² better
                r2_improvement = r2_exp - r2_linear

                if r2_improvement < 0.05 and F_range_pct < 10.0:  # Exponential barely better than linear
                    obvious_no_signal[name] = {
                        'reason': f'Linear drift only (R²_exp={r2_exp:.3f} vs R²_linear={r2_linear:.3f}, Δ={r2_improvement:.3f})',
                        'F_range': F_range,
                        'F_range_pct': F_range_pct,
                        'R2_exp': r2_exp,
                        'R2_linear': r2_linear,
                        'R2_improvement': r2_improvement,
                        'D0': D0,
                        'check_type': 'linear_drift'
                    }
                else:
                    # Passed both checks - real exponential signal
                    clear_signal_samples[name] = fluor
                    if verbose:
                        print(f"  ✓ {name}: R²_exp={r2_exp:.3f}, R²_linear={r2_linear:.3f}, Δ={r2_improvement:.3f}, D0={D0:.2e} (valid)")

            except Exception as e:
                # Error during fitting - likely problematic data
                obvious_no_signal[name] = {
                    'reason': f'Fitting error: {str(e)[:50]}',
                    'F_range': F_range,
                    'F_range_pct': F_range_pct,
                    'check_type': 'error'
                }

    # Step 4: Compile results
    plate_stats = {
        'max_range': max_range_on_plate,
        'max_fluor': max_fluor_on_plate,
        'range_threshold': range_threshold,
        'borderline_threshold': borderline_threshold,
        'n_total': len(all_samples),
        'n_signal': len(clear_signal_samples),
        'n_no_signal': len(obvious_no_signal),
        'n_borderline_passed': len(borderline_samples) - len([k for k in borderline_samples if k in obvious_no_signal])
    }

    if verbose:
        print(f"\n=== Results ===")
        print(f"✅ {len(clear_signal_samples)} samples with valid signal")
        print(f"⚠️  {len(obvious_no_signal)} samples with no detectable signal")

    return clear_signal_samples, obvious_no_signal, plate_stats


