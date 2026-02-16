"""
Data processing utilities for qPCR data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict



def export_results(
    cycles: np.ndarray,
    fluorescence_data: np.ndarray,
    fluorescence_fit: np.ndarray,
    params: dict,
    output_path: str
):
    """
    Export fitting results to CSV.
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence_data : np.ndarray
        Measured fluorescence
    fluorescence_fit : np.ndarray
        Fitted fluorescence
    params : dict
        Fitted parameters
    output_path : str
        Path for output CSV file
    """
    # Create DataFrame with data
    df = pd.DataFrame({
        'Cycle': cycles,
        'Fluorescence_Data': fluorescence_data,
        'Fluorescence_Fit': fluorescence_fit,
        'Residual': fluorescence_data - fluorescence_fit
    })
    
    # Add parameters as metadata in separate rows
    param_df = pd.DataFrame({
        'Parameter': list(params.keys()),
        'Value': list(params.values())
    })
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("# MAK2 Fitting Results\n")
        f.write("# Parameters:\n")
        param_df.to_csv(f, index=False)
        f.write("\n# Data:\n")
        df.to_csv(f, index=False)
    
    print(f"Results exported to {output_path}")


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
                if r2_exp < min_r2:
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

                if r2_improvement < 0.05:  # Exponential barely better than linear
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


if __name__ == "__main__":
    # Test example data generation
    examples = prepare_example_data()
    
    print("Generated example datasets:")
    for name, data in examples.items():
        print(f"\n{name}:")
        print(f"  Cycles: {len(data['cycles'])}")
        print(f"  Fluorescence range: {data['fluorescence'].min():.2f} - {data['fluorescence'].max():.2f}")
        print(f"  True params: D0={data['true_params']['D0']}, k={data['true_params']['k']}, P0={data['true_params']['P0']:.2e}")
