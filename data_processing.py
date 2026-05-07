"""
Data processing utilities for qPCR data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict



def estimate_baseline_end(cycles, fluorescence, first_cycle_idx=2, window_size=12, n_sd=5, max_iter=3):
    """Estimate end of pre-amplification baseline region using forward projection.

    Fits a line to an early window of cycles, then scans forward to find the first
    cycle where fluorescence exceeds the projected baseline + n_sd * SD of residuals.
    Iterates to refine the window boundary.

    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers (1-indexed)
    fluorescence : np.ndarray
        Raw fluorescence values
    first_cycle_idx : int
        Index (0-based) of the first cycle to include in baseline window
    window_size : int
        Initial number of cycles to use for baseline fit
    n_sd : float
        Number of baseline SDs above projection to call as end-of-baseline
    max_iter : int
        Maximum iterations

    Returns
    -------
    int
        0-based index of the last baseline cycle (exclusive end, i.e. baseline is cycles[:result])
    """
    n = len(cycles)
    bl_start = first_cycle_idx
    bl_end = min(bl_start + window_size, n - 1)
    for _ in range(max_iter):
        bl_c = cycles[bl_start:bl_end]
        bl_f = fluorescence[bl_start:bl_end]
        if len(bl_c) < 3:
            break
        coeffs = np.polyfit(bl_c, bl_f, 1)
        residuals = bl_f - np.polyval(coeffs, bl_c)
        sd = max(float(np.std(residuals)), 1e-10)
        new_bl_end = bl_end
        for i in range(bl_end, n):
            if fluorescence[i] > np.polyval(coeffs, cycles[i]) + n_sd * sd:
                new_bl_end = i
                break
        if new_bl_end == bl_end:
            break
        bl_end = new_bl_end
    return bl_end  # index of last baseline cycle (exclusive)


def detect_no_signal_samples(
    cycles: np.ndarray,
    all_samples: Dict[str, np.ndarray],
    min_range_pct: float = 2.0,
    min_r2: float = 0.80,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], dict]:
    """
    Detect wells with no real qPCR signal using plate-wide fluorescence range.

    Strategy:
    1. Calculate max fluorescence range across ALL samples on plate
    2. Flag samples with range < X% of max range (e.g., 2%)
    3. For borderline cases (2-5% of max), check exponential fit R²

    This avoids fitting MAK2+ model to wells with no amplification signal
    (e.g., NTC controls, failed reactions, or extremely low/no template).

    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers (same for all samples)
    all_samples : dict
        Dictionary {sample_name: fluorescence_array}
    min_range_pct : float, optional
        Minimum fluorescence range as % of max plate range (default 2%)
    min_r2 : float, optional
        Minimum R² for exponential fit for borderline cases (default 0.80)
    verbose : bool, optional
        Print diagnostic information (default True)

    Returns
    -------
    valid_samples : dict
        Samples with detectable signal {name: fluorescence_array}
    no_signal_samples : dict
        Samples without signal {name: diagnostic_info_dict}
    plate_stats : dict
        Plate-wide statistics for reference

    Examples
    --------
    >>> valid, no_signal, stats = detect_no_signal_samples(cycles, all_samples)
    >>> print(f"Found {len(valid)} samples with signal")
    >>> print(f"Flagged {len(no_signal)} samples with no signal")
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


