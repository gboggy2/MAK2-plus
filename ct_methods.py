"""
Common Ct (threshold cycle) calculation methods for qPCR.

This module implements standard Ct determination algorithms used by
commercial qPCR instruments and software packages.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Dict


def calculate_ct_threshold(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    threshold: Optional[float] = None,
    method: str = 'auto'
) -> Tuple[float, float]:
    """
    Calculate Ct using fixed fluorescence threshold method.

    This is the most common method used by instruments:
    - Set threshold in exponential phase (often 10× baseline SD)
    - Find cycle where fluorescence crosses threshold
    - Interpolate for sub-cycle precision

    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Fluorescence values
    threshold : float, optional
        Fluorescence threshold. If None, auto-calculated.
    method : str
        Method for auto threshold: 'auto', 'baseline_sd', 'midpoint'

    Returns
    -------
    ct : float
        Threshold cycle (fractional)
    threshold_used : float
        Threshold value used
    """
    # Auto-calculate threshold if not provided
    if threshold is None:
        if method == 'auto' or method == 'baseline_sd':
            # Common: 10× standard deviation of baseline (first 5-10 cycles)
            baseline = fluorescence[:min(10, len(fluorescence)//3)]
            baseline_mean = np.mean(baseline)
            baseline_sd = np.std(baseline)
            threshold = baseline_mean + 10 * baseline_sd

        elif method == 'midpoint':
            # Alternative: midpoint between min and max
            threshold = (np.min(fluorescence) + np.max(fluorescence)) / 2

        else:
            raise ValueError(f"Unknown method: {method}")

    # Find where fluorescence crosses threshold
    above_threshold = fluorescence >= threshold

    if not np.any(above_threshold):
        # Never crosses threshold
        return np.nan, threshold

    # Find first crossing point
    crossing_idx = np.where(above_threshold)[0][0]

    if crossing_idx == 0:
        # Already above threshold at cycle 1
        return cycles[0], threshold

    # Linear interpolation between points bracketing threshold
    # y = mx + b, solve for x when y = threshold
    c1, c2 = cycles[crossing_idx - 1], cycles[crossing_idx]
    f1, f2 = fluorescence[crossing_idx - 1], fluorescence[crossing_idx]

    if f2 == f1:  # Avoid division by zero
        ct = c1
    else:
        # Linear interpolation: ct = c1 + (threshold - f1) / (f2 - f1) * (c2 - c1)
        ct = c1 + (threshold - f1) * (c2 - c1) / (f2 - f1)

    return ct, threshold


def calculate_ct_second_derivative(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    window: int = 5
) -> float:
    """
    Calculate Ct using second derivative maximum method.

    The Ct is defined as the cycle where the second derivative
    (curvature) is maximum - this corresponds to the inflection point
    of the amplification curve.

    Used by: Bio-Rad CFX Manager (optional mode)

    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Fluorescence values
    window : int
        Smoothing window size for derivatives

    Returns
    -------
    ct : float
        Ct at maximum second derivative
    """
    # Smooth data with moving average
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(fluorescence, kernel, mode='valid')
        cycles_smooth = cycles[window//2:-(window//2)]
    else:
        smoothed = fluorescence
        cycles_smooth = cycles

    # First derivative (rate of change)
    first_deriv = np.gradient(smoothed)

    # Second derivative (acceleration/curvature)
    second_deriv = np.gradient(first_deriv)

    # Find maximum of second derivative
    max_idx = np.argmax(second_deriv)
    ct = cycles_smooth[max_idx]

    return ct


def calculate_ct_regression(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    window_size: int = 5
) -> Tuple[float, Dict]:
    """
    Calculate Ct using linear regression on log-phase.

    Method:
    1. Find exponential phase using first derivative
    2. Fit linear regression to log(fluorescence) in exp phase
    3. Extrapolate to baseline threshold

    Used by: Some research-grade analysis (LinRegPCR software)

    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Fluorescence values
    window_size : int
        Size of window for exponential phase detection

    Returns
    -------
    ct : float
        Ct from regression
    info : dict
        Regression statistics (slope=efficiency, R²)
    """
    # Remove any non-positive values for log transform
    valid_mask = fluorescence > 0
    cycles_valid = cycles[valid_mask]
    fluor_valid = fluorescence[valid_mask]

    if len(fluor_valid) < 5:
        return np.nan, {'efficiency': np.nan, 'r2': np.nan}

    # Find exponential phase: where log(F) is most linear
    log_fluor = np.log(fluor_valid)

    # Sliding window to find most linear region
    best_r2 = -np.inf
    best_window = None

    for i in range(len(cycles_valid) - window_size):
        x = cycles_valid[i:i+window_size]
        y = log_fluor[i:i+window_size]

        # Linear regression
        coeffs = np.polyfit(x, y, deg=1)
        y_pred = np.polyval(coeffs, x)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

        if r2 > best_r2:
            best_r2 = r2
            best_window = (i, i+window_size, coeffs)

    if best_window is None:
        return np.nan, {'efficiency': np.nan, 'r2': np.nan}

    # Use best linear region
    start, end, coeffs = best_window
    slope, intercept = coeffs

    # Calculate efficiency from slope
    # E = 10^(1/slope) - 1 (since log10 of 2^n)
    # Or for natural log: E = e^slope - 1
    efficiency = np.exp(slope) - 1

    # Calculate Ct: where regression line crosses baseline threshold
    baseline = np.mean(fluor_valid[:min(5, len(fluor_valid)//4)])
    log_baseline = np.log(baseline)

    # Solve: log_baseline = slope * ct + intercept
    ct = (log_baseline - intercept) / slope

    return ct, {
        'efficiency': efficiency,
        'r2': best_r2,
        'slope': slope,
        'intercept': intercept
    }


def calculate_all_ct_methods(
    cycles: np.ndarray,
    fluorescence: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Ct using all common methods for comparison.

    Returns
    -------
    results : dict
        Dictionary with Ct values from each method
    """
    results = {}

    # Method 1: Threshold (auto)
    ct_thresh, threshold = calculate_ct_threshold(cycles, fluorescence, method='auto')
    results['threshold_auto'] = ct_thresh
    results['threshold_value'] = threshold

    # Method 2: Threshold (midpoint)
    ct_mid, _ = calculate_ct_threshold(cycles, fluorescence, method='midpoint')
    results['threshold_midpoint'] = ct_mid

    # Method 3: Second derivative maximum
    ct_d2 = calculate_ct_second_derivative(cycles, fluorescence)
    results['second_derivative'] = ct_d2

    # Method 4: Regression
    ct_reg, reg_info = calculate_ct_regression(cycles, fluorescence)
    results['regression'] = ct_reg
    results['regression_efficiency'] = reg_info.get('efficiency', np.nan)
    results['regression_r2'] = reg_info.get('r2', np.nan)

    return results


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd

    # Load example data
    df = pd.read_csv('example_data/Boggy.csv')

    # Test on F1.1 (high concentration, clean curve)
    sample = 'F1.1'
    cycles = df['Cycles'].values
    fluorescence = df[sample].values

    print(f"Ct Methods Comparison for {sample}")
    print("=" * 60)

    results = calculate_all_ct_methods(cycles, fluorescence)

    for method, value in results.items():
        if 'efficiency' in method or 'r2' in method or 'threshold_value' in method:
            print(f"{method:30s}: {value:.4f}")
        else:
            print(f"{method:30s}: {value:.2f}")

    print("\n" + "=" * 60)
    print("\nTesting on all samples...")

    # Compare across dilution series
    comparison_data = []
    for col in df.columns[1:]:  # Skip 'Cycles' column
        cts = calculate_all_ct_methods(df['Cycles'].values, df[col].values)
        comparison_data.append({
            'Sample': col,
            'Ct_Threshold': cts['threshold_auto'],
            'Ct_SecondDeriv': cts['second_derivative'],
            'Ct_Regression': cts['regression'],
            'Efficiency': cts['regression_efficiency']
        })

    results_df = pd.DataFrame(comparison_data)
    print(results_df.to_string(index=False))

    # Calculate Ct differences
    print("\n" + "=" * 60)
    print("Method Comparison (ΔCt from threshold method):")
    results_df['ΔCt_SecondDeriv'] = results_df['Ct_SecondDeriv'] - results_df['Ct_Threshold']
    results_df['ΔCt_Regression'] = results_df['Ct_Regression'] - results_df['Ct_Threshold']

    print(f"\nMean difference (Second Deriv): {results_df['ΔCt_SecondDeriv'].mean():.3f} ± {results_df['ΔCt_SecondDeriv'].std():.3f}")
    print(f"Mean difference (Regression):   {results_df['ΔCt_Regression'].mean():.3f} ± {results_df['ΔCt_Regression'].std():.3f}")
