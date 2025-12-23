"""
MAK2 Model with Primer Depletion
A mechanistic model of PCR for qPCR data fitting.

Based on Boggy & Woolf (2010) with extensions for primer depletion.
"""

import numpy as np
from typing import Tuple, Optional


class MAK2Model:
    """
    Implements the MAK2 mechanistic PCR model with primer depletion.
    
    The model tracks:
    - DNA concentration at each cycle
    - Primer depletion across cycles
    - Background fluorescence (linear model)
    """
    
    def __init__(self):
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
        """
        Simulate PCR for n_cycles using the MAK2 model with primer depletion.
        
        Note: D0 is now in FLUORESCENCE UNITS (not molecules), representing
        the initial fluorescence contribution from template DNA. This combines
        the previous D0 (molecules) × F_scale into a single observable parameter.
        
        Parameters
        ----------
        D0 : float
            Initial DNA fluorescence (fluorescence units, not molecules)
            This is the directly observable quantity = template × fluorophore_per_template
        k : float
            PCR characteristic constant (k_a / 2k_b)
        P0 : float
            Initial primer concentration (in same units as D0)
        n_cycles : int
            Number of PCR cycles to simulate
        F_bg_intercept : float
            Background fluorescence intercept
        F_bg_slope : float
            Background fluorescence slope (per cycle)
            
        Returns
        -------
        cycles : np.ndarray
            Array of cycle numbers (0, 1, 2, ..., n_cycles-1)
        D : np.ndarray
            DNA fluorescence at each cycle
        F : np.ndarray
            Total fluorescence (DNA signal + background) at each cycle
        """
        # Initialize arrays
        cycles = np.arange(n_cycles)
        D = np.zeros(n_cycles)
        P = np.zeros(n_cycles)
        F = np.zeros(n_cycles)
        
        # Set initial conditions
        D[0] = D0
        P[0] = P0
        
        # Simulate each cycle
        for n in range(1, n_cycles):
            # Calculate effective rate constant for this cycle
            k_eff = k * P[n-1]
            
            # Prevent division by zero or negative log arguments
            if k_eff <= 0 or D[n-1] <= 0:
                D[n] = D[n-1]
                P[n] = P[n-1]
                continue
            
            # MAK2 equation with primer concentration
            # D_n = D_{n-1} + k_eff * ln(1 + D_{n-1} / k_eff)
            log_arg = 1 + D[n-1] / k_eff
            
            if log_arg <= 0:
                D[n] = D[n-1]
                P[n] = P[n-1]
                continue
                
            D_increment = k_eff * np.log(log_arg)
            D[n] = D[n-1] + D_increment
            
            # Update primer concentration (primers consumed by amplification)
            # Each DNA molecule produced consumes primers
            primers_consumed = D_increment
            P[n] = max(0, P[n-1] - primers_consumed)
        
        # Calculate fluorescence: DNA signal + background
        # D is already in fluorescence units, so just add background
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
        """
        Simulate PCR and return fluorescence at specific cycle numbers.
        Useful for fitting to data where cycles may not start at 0.
        
        Parameters
        ----------
        D0 : float
            Initial DNA fluorescence (fluorescence units)
        k : float
            PCR characteristic constant
        P0 : float
            Initial primer concentration
        cycles : np.ndarray
            Array of cycle numbers to evaluate
        F_bg_intercept : float
            Background fluorescence intercept
        F_bg_slope : float
            Background fluorescence slope
        cycle_offset : float
            Cycle at which amplification begins (lag phase)
            
        Returns
        -------
        F : np.ndarray
            Fluorescence at each requested cycle
        """
        # Shift cycles by offset (amplification starts at cycle_offset)
        effective_cycles = np.maximum(0, cycles - cycle_offset)
        max_cycle = int(np.max(effective_cycles)) + 1
        
        _, _, F_all = self.simulate_cycles(
            D0, k, P0, max_cycle, F_bg_intercept, F_bg_slope
        )
        
        # Interpolate or extract values at requested effective cycles
        F_result = np.zeros_like(cycles, dtype=float)
        for i, eff_cyc in enumerate(effective_cycles):
            if eff_cyc < 0:
                # Before amplification starts - just background
                F_result[i] = F_bg_intercept + F_bg_slope * cycles[i]
            else:
                idx = int(eff_cyc)
                if idx < len(F_all):
                    F_result[i] = F_all[idx]
                else:
                    F_result[i] = F_all[-1]
        
        return F_result


def calculate_amplification_efficiency(D: np.ndarray) -> np.ndarray:
    """
    Calculate cycle-by-cycle amplification efficiency.
    
    Efficiency at cycle n is defined as:
    E_n = (D_n - D_{n-1}) / D_{n-1}
    
    Parameters
    ----------
    D : np.ndarray
        DNA concentration at each cycle
        
    Returns
    -------
    efficiency : np.ndarray
        Amplification efficiency at each cycle (starts at cycle 1)
    """
    efficiency = np.zeros(len(D) - 1)
    for n in range(1, len(D)):
        if D[n-1] > 0:
            efficiency[n-1] = (D[n] - D[n-1]) / D[n-1]
    return efficiency


def find_plateau_onset(fluorescence: np.ndarray, threshold: float = 0.05) -> int:
    """
    Find the onset of the plateau phase where amplification stops.
    
    The plateau is detected when the slope drops below a threshold
    relative to the maximum slope observed.
    
    Parameters
    ----------
    fluorescence : np.ndarray
        Fluorescence values at each cycle
    threshold : float
        Fraction of max slope to use as cutoff (default: 0.05 = 5%)
        
    Returns
    -------
    plateau_cycle : int
        Cycle number where plateau begins (or last cycle if no plateau)
    """
    # Calculate first derivative (slope)
    slope = np.diff(fluorescence)
    
    if len(slope) == 0:
        return len(fluorescence) - 1
    
    # Find maximum slope
    max_slope = np.max(slope)
    
    # Find where slope drops below threshold of max slope
    cutoff = threshold * max_slope
    
    # Find first cycle where slope is consistently below cutoff
    plateau_idx = len(slope)  # Default to end
    for i in range(len(slope)):
        if slope[i] < cutoff:
            # Check if it stays low for next few cycles
            if i + 3 < len(slope):
                if np.all(slope[i:i+3] < cutoff):
                    plateau_idx = i
                    break
            else:
                plateau_idx = i
                break
    
    # Add 1 because diff loses one element
    return min(plateau_idx + 1, len(fluorescence) - 1)


def find_fluorescence_threshold_cycle(
    fluorescence: np.ndarray, 
    threshold_pct: float = 85.0
) -> int:
    """
    Find the cycle where fluorescence reaches a percentage of maximum.
    
    This is useful for truncating data before deep plateau effects
    (enzyme degradation, dNTP depletion, etc.) that aren't modeled
    by primer depletion alone.
    
    Parameters
    ----------
    fluorescence : np.ndarray
        Fluorescence values at each cycle
    threshold_pct : float
        Percentage of maximum fluorescence (default: 85.0%)
        
    Returns
    -------
    threshold_cycle : int
        First cycle where fluorescence exceeds threshold_pct of maximum
        Returns last cycle if threshold never reached
    """
    if len(fluorescence) == 0:
        return 0
    
    max_F = np.max(fluorescence)
    threshold_value = (threshold_pct / 100.0) * max_F
    
    # Find first cycle exceeding threshold
    threshold_cycles = np.where(fluorescence >= threshold_value)[0]
    
    if len(threshold_cycles) > 0:
        return int(threshold_cycles[0])
    else:
        # Threshold never reached - use all data
        return len(fluorescence) - 1


def find_slope_threshold_cycle(
    fluorescence: np.ndarray,
    slope_pct: float = 10.0
) -> int:
    """
    Find the cycle where slope drops below a percentage of maximum slope.
    
    This detects when amplification efficiency decreases significantly,
    indicating transition into plateau phase. Looks for where the slope
    first drops below the threshold AFTER reaching maximum slope.
    
    Parameters
    ----------
    fluorescence : np.ndarray
        Fluorescence values at each cycle
    slope_pct : float
        Percentage of maximum slope (default: 10.0%)
        
    Returns
    -------
    threshold_cycle : int
        First cycle (after max slope) where slope drops below threshold
        Returns last cycle if threshold never reached
    """
    if len(fluorescence) < 2:
        return len(fluorescence) - 1
    
    # Calculate slope (first derivative)
    slope = np.diff(fluorescence)
    
    if len(slope) == 0:
        return len(fluorescence) - 1
    
    # Find maximum slope and its position
    max_slope_idx = np.argmax(slope)
    max_slope = slope[max_slope_idx]
    
    if max_slope <= 0:
        return len(fluorescence) - 1
    
    # Calculate threshold
    threshold_value = (slope_pct / 100.0) * max_slope
    
    # Find first cycle AFTER max slope where slope drops below threshold
    # and stays below for at least 2 more cycles
    for i in range(max_slope_idx, len(slope)):
        if slope[i] < threshold_value:
            # Check if it stays low
            if i + 2 < len(slope):
                if np.all(slope[i:i+3] < threshold_value):
                    return i + 1  # +1 because diff loses one element
            else:
                return i + 1
    
    return len(fluorescence) - 1


def find_truncation_cycle(fluorescence: np.ndarray) -> int:
    """
    Legacy function - now returns fluorescence threshold cycle.
    
    DEPRECATED: Use find_fluorescence_threshold_cycle() or 
    find_slope_threshold_cycle() instead.
    
    Parameters
    ----------
    fluorescence : np.ndarray
        Fluorescence values at each cycle
        
    Returns
    -------
    threshold_cycle : int
        Cycle where fluorescence reaches 85% of maximum
    """
    return find_fluorescence_threshold_cycle(fluorescence, threshold_pct=85.0)


def estimate_D0_bounds(
    cycles: np.ndarray,
    fluorescence: np.ndarray
) -> tuple:
    """
    Estimate bounds on initial DNA fluorescence (D0) by fitting exponentials.
    
    Uses two exponential models starting from cycle 1:
    1. Perfect doubling: F = F_bg + D0 * 2^n (lower bound)
    2. With efficiency: F = F_bg + D0 * E^n (upper bound, E~1.8)
    
    Note: D0 is now in fluorescence units directly, not molecules.
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Fluorescence measurements
        
    Returns
    -------
    D0_lower : float
        Lower bound on D0 (from perfect doubling model)
    D0_upper : float
        Upper bound on D0 (from efficiency model)
    F_bg_estimate : float
        Estimated background fluorescence
    fit_info : dict
        Additional info for visualization (cycles, fitted values, etc.)
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
        
        # IMPROVED: For strong drifting baselines, detrend the data first
        # Fit linear trend to early cycles and subtract it
        baseline_fluor_cycles = min(10, len(fluorescence) // 4)
        early_cycles = cycles[:baseline_fluor_cycles]
        early_fluor = fluorescence[:baseline_fluor_cycles]
        
        # Fit linear trend to early data
        if len(early_fluor) >= 2:
            trend_coeffs = np.polyfit(early_cycles, early_fluor, 1)
            trend_slope = trend_coeffs[0]
            trend_intercept = trend_coeffs[1]
            
            # Detrend ALL fluorescence data
            fluorescence_detrended = fluorescence - (trend_intercept + trend_slope * cycles)
            
            # Calculate baseline stats from detrended data
            baseline_fluor_values = fluorescence_detrended[:baseline_fluor_cycles]
            baseline_fluor_median = np.median(baseline_fluor_values)
            baseline_fluor_std = np.std(baseline_fluor_values)
            
            # Fluorescence must be at least 5σ above baseline (more conservative for detrended)
            fluor_threshold = baseline_fluor_median + 5 * baseline_fluor_std
        else:
            # Fallback: no detrending
            baseline_fluor_values = fluorescence[:baseline_fluor_cycles]
            baseline_fluor_median = np.median(baseline_fluor_values)
            baseline_fluor_std = np.std(baseline_fluor_values)
            fluor_threshold = baseline_fluor_median + 3 * baseline_fluor_std
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
    
    # Find first cycle with real signal (for information only)
    min_signal_threshold = max(0.001, F_min + 0.01 * F_range)
    real_signal_idx = np.where(fluorescence > min_signal_threshold)[0]
    
    if len(real_signal_idx) > 0:
        first_signal_cycle = real_signal_idx[0]
        print(f"  First cycle with real signal (>{min_signal_threshold:.4f}): cycle {first_signal_cycle}")
    else:
        first_signal_cycle = 0
        print(f"  Warning: No cycles above threshold {min_signal_threshold:.4f}")
    
    # Start exponential fitting from cycle 0 for all data types
    # The model with background fitting can handle baseline cycles
    exp_start_cycle = 0
    
    # Now fit exponentials: from cycle 0 through cycles into exponential phase
    # Efficiency: up to 30% of fluorescence range
    min_points = 5
    threshold_30 = F_min + 0.30 * F_range
    above_30 = np.where(fluorescence > threshold_30)[0]
    
    if len(above_30) > 0:
        exp_end_upper = min(above_30[0], len(cycles) - 1)
    else:
        exp_end_upper = min(baseline_end_idx + 10, len(cycles) - 1)
    
    # Make sure we have enough range from exp_start_cycle
    if exp_end_upper - exp_start_cycle < min_points:
        exp_end_upper = min(exp_start_cycle + min_points + 3, len(cycles) - 1)
    
    exp_region_upper = np.arange(exp_start_cycle, exp_end_upper + 1)
    
    # Perfect doubling: 4 cycles BEFORE efficiency endpoint
    # This captures the early exponential phase before efficiency < 2
    exp_end_lower = max(exp_start_cycle + min_points, exp_end_upper - 4)
    exp_region_lower = np.arange(exp_start_cycle, exp_end_lower + 1)
    
    # Ensure minimum points
    if len(exp_region_lower) < min_points:
        exp_region_lower = np.arange(exp_start_cycle, min(exp_start_cycle + min_points, len(cycles)))
    if len(exp_region_upper) < min_points:
        exp_region_upper = np.arange(exp_start_cycle, min(exp_start_cycle + min_points + 3, len(cycles)))
    
    threshold = fluorescence[baseline_end_idx] if baseline_end_idx < len(fluorescence) else baseline
    
    # Extract data for both regions
    cycles_lower = cycles[exp_region_lower]
    fluor_lower = fluorescence[exp_region_lower]
    
    cycles_upper = cycles[exp_region_upper]
    fluor_upper = fluorescence[exp_region_upper]
    
    # Shift cycles so they start at n=0 for numerical stability
    # Always use cycles[0] as offset for consistent behavior
    cycle_offset = cycles[0]
    cycles_lower_shifted = cycles_lower - cycle_offset
    cycles_upper_shifted = cycles_upper - cycle_offset
    
    F_min = fluorescence.min()
    
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
                        [1e-15, bg_intercept_min, bg_slope_min],  # Lower D0 bound for numerical stability
                        [10.0, bg_intercept_max, bg_slope_max]
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
                        [1e-15, 1.0, bg_intercept_min, bg_slope_min],  # Lower D0 bound for numerical stability
                        [10.0, 2.0, bg_intercept_max, bg_slope_max]
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
        
        # Add some margin (10x on each side) - NO minimum floor
        D0_lower_bounded = D0_lower / 10
        D0_upper_bounded = min(100.0, D0_upper * 10)  # Cap at reasonable max
        
        print(f"  After adding margin (10x):")
        print(f"    Lower bound: {D0_lower_bounded:.2e}")
        print(f"    Upper bound: {D0_upper_bounded:.2e}")
        
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
            'bg_slope': F_bg_slope
        }
        
        print(f"\nEstimated D0 bounds from sliding window baseline detection:")
        if baseline_detected:
            print(f"  Baseline ends at cycle {baseline_end_idx}")
        else:
            print(f"  Baseline end at cycle {baseline_end_idx} (fallback: no slope change detected)")
        print(f"  30% fluorescence threshold: {threshold_30:.4f} (reached at cycle {exp_end_upper})")
        print(f"  Perfect doubling fit: cycle {cycles_lower[0]:.0f} to {cycles_lower[-1]:.0f} (efficiency - 4), R² = {r2_lower:.4f}")
        print(f"    Fitted background: intercept={F_bg_intercept1:.4f}, slope={F_bg_slope1:.6f}")
        
        # Check if parameters hit bounds
        if abs(F_bg_intercept1 - bg_intercept_min) < 0.0001 or abs(F_bg_intercept1 - bg_intercept_max) < 0.0001:
            print(f"    ⚠️  WARNING: Intercept at bound!")
        if abs(F_bg_slope1 - bg_slope_min) < 0.00001 or abs(F_bg_slope1 - bg_slope_max) < 0.00001:
            print(f"    ⚠️  WARNING: Slope at bound!")
            
        print(f"  Efficiency fit: cycle {cycles_upper[0]:.0f} to {cycles_upper[-1]:.0f} (to 30% threshold), R² = {r2_upper:.4f}, E = {efficiency:.2f}")
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
    """
    Analytically estimate k by matching MAK2 growth to efficiency exponential.
    
    Strategy:
    1. Efficiency exponential tells us: D_n = D0 × E^n
    2. MAK2 growth per cycle: ΔD = k × P × ln(1 + D/(k×P))
    3. Match them at a specific cycle to solve for k
    
    Parameters
    ----------
    D0_eff : float
        Initial DNA from efficiency exponential fit
    E : float
        Efficiency from exponential fit (typically 1.3-1.9)
    cycles : np.ndarray
        Cycle numbers that were fit
    P0_assumed : float
        Assumed P0 value (default 1.0, will be refined during full MAK2 fit)
    use_cycle : int, optional
        Specific cycle to use for matching. If None, uses middle of fitted region.
        
    Returns
    -------
    k_estimate : float
        Estimated k value
    """
    from scipy.optimize import fsolve
    
    # Use middle cycle of exponential fit region for matching
    if use_cycle is None:
        use_cycle = len(cycles) // 2
    else:
        use_cycle = min(use_cycle, len(cycles) - 1)
    
    n_cycle = cycles[use_cycle]
    
    # DNA concentration at cycle n-1 (from exponential)
    D_prev = D0_eff * E**(n_cycle - 1)
    
    # Expected growth from exponential
    growth_exp = D_prev * (E - 1)
    
    # Solve for k where MAK2 growth matches exponential growth
    # growth_exp = k × P0 × ln(1 + D_prev / (k × P0))
    
    def equation(k):
        if k <= 1e-8:
            return 1e10
        try:
            mak2_growth = k * P0_assumed * np.log(1 + D_prev / (k * P0_assumed))
            return (growth_exp - mak2_growth)**2
        except:
            return 1e10
    
    # Try multiple initial guesses
    k_estimates = []
    for k_init in [0.01, 0.05, 0.1, 0.2, 0.5]:
        try:
            result = fsolve(equation, k_init, full_output=True)
            if result[2] == 1:  # Solution converged
                k_est = result[0][0]
                if 1e-6 < k_est < 100:  # Sanity check
                    k_estimates.append(k_est)
        except:
            continue
    
    if not k_estimates:
        # Fallback: use empirical relationship
        # E close to 2 → minimal depletion → small k
        # E close to 1 → strong depletion → large k
        k_estimate = 0.3 * (2.0 - E)
        print(f"    Warning: Numerical solution failed, using empirical estimate")
    else:
        k_estimate = np.median(k_estimates)
    
    return k_estimate


def estimate_MAK2_params_from_exponential(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    P0_assumed: float = 1.0,
    verbose: bool = True
) -> Tuple[dict, dict]:
    """
    Analytically estimate MAK2 parameters from exponential fits.
    
    This provides data-driven initial guesses and tight bounds for MAK2 optimization.
    
    Strategy:
    1. Fit efficiency exponential: F = D0 × E^n + bg (already very accurate)
    2. D0 estimate: Use D0 from exponential fit directly
    3. k estimate: Solve for k where MAK2 growth matches exponential growth
    4. P0 estimate: Assume typical value (1.0), will be refined in plateau region
    5. Return tight bounds (±1 order of magnitude) around estimates
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Fluorescence values
    P0_assumed : float
        Assumed P0 for k estimation (default 1.0)
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    estimates : dict
        Point estimates for D0, k, P0, F_bg_intercept, F_bg_slope
    bounds : dict
        Tight bounds for each parameter (±1 order of magnitude)
    """
    
    if verbose:
        print("\n=== Analytical MAK2 Parameter Estimation ===")
    
    # Step 1: Get exponential fits (this already exists and works well!)
    D0_lower, D0_upper, F_bg_est, fit_info = estimate_D0_bounds(cycles, fluorescence)
    
    # Extract efficiency exponential results
    D0_eff = fit_info['D0_efficiency']
    E = fit_info['efficiency']
    fitted_cycles = fit_info['fitted_cycles_efficiency']
    
    if verbose:
        print(f"\nExponential Fit Results:")
        print(f"  D0 = {D0_eff:.2e}")
        print(f"  Efficiency E = {E:.4f}")
        print(f"  Background: {F_bg_est['intercept']:.4f} + {F_bg_est['slope']:.6f} × n")
    
    # Step 2: D0 estimate is direct
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
    
    bounds = {
        'D0': (D0_estimate / 10, D0_estimate * 10),
        'k': (k_estimate / 10, k_estimate * 10),
        'P0': (P0_estimate * 0.5, P0_estimate * 2),  # Tight: 0.5x to 2x of F_max
        'F_bg_intercept': (
            max(0, F_bg_est['intercept'] - 3 * F_bg_est['SE_intercept']),
            F_bg_est['intercept'] + 3 * F_bg_est['SE_intercept']
        ),
        'F_bg_slope': (
            F_bg_est['slope'] - 5 * F_bg_est['SE_slope'],
            F_bg_est['slope'] + 5 * F_bg_est['SE_slope']
        )
    }
    
    if verbose:
        print(f"\nData-Driven Bounds:")
        print(f"  D0: [{bounds['D0'][0]:.2e}, {bounds['D0'][1]:.2e}]  (100× range)")
        print(f"  k:  [{bounds['k'][0]:.6f}, {bounds['k'][1]:.6f}]  (100× range)")
        print(f"  P0: [{bounds['P0'][0]:.2f}, {bounds['P0'][1]:.2f}]  (0.5× to 2× F_max)")
    
    return estimates, bounds


if __name__ == "__main__":
    # Example usage
    model = MAK2Model()
    
    # Simulate with typical parameters
    D0 = 100  # Initial template molecules
    k = 0.5   # PCR constant
    P0 = 1e6  # Initial primer concentration
    n_cycles = 40
    
    cycles, D, F = model.simulate_cycles(
        D0=D0,
        k=k,
        P0=P0,
        n_cycles=n_cycles,
        F_bg_intercept=0.1,
        F_bg_slope=0.001
    )
    
    print("MAK2 Model Simulation")
    print(f"D0 = {D0}, k = {k}, P0 = {P0}")
    print(f"\nFirst 10 cycles:")
    print("Cycle\tDNA\tFluorescence")
    for i in range(10):
        print(f"{i}\t{D[i]:.2f}\t{F[i]:.4f}")
    
    # Calculate efficiency
    efficiency = calculate_amplification_efficiency(D)
    print(f"\nAmplification efficiency (first 10 cycles):")
    for i in range(10):
        print(f"Cycle {i+1}: {efficiency[i]:.4f}")
    
    # Find truncation point
    trunc_cycle = find_truncation_cycle(F)
    print(f"\nTruncation cycle (max slope increase): {trunc_cycle}")
