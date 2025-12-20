"""
Optimizer for fitting MAK2 model to qPCR data.
Uses adaptive multi-start Levenberg-Marquardt optimization.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
from mak2_model import (
    MAK2Model, 
    find_fluorescence_threshold_cycle,
    find_slope_threshold_cycle,
    estimate_D0_bounds,
    estimate_MAK2_params_from_exponential
)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class MAK2Optimizer:
    """
    Fits MAK2 model parameters to qPCR data using adaptive multi-start
    Trust Region Reflective (TRF) optimization.
    
    The optimizer automatically retries with different initial guesses until
    a good fit (R² ≥ threshold) is achieved, or max attempts is reached.
    
    TRF is a robust bounded optimization method similar to Levenberg-Marquardt
    but with support for parameter bounds.
    """
    
    def __init__(self, model: Optional[MAK2Model] = None):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        model : MAK2Model, optional
            MAK2 model instance (creates new one if not provided)
        """
        self.model = model or MAK2Model()
        self.optimal_params = None
        self.cycles_fit = None
        self.fluorescence_fit = None
        self.metrics = None
        self.n_attempts = None
    
    def fit(
        self,
        cycles: np.ndarray,
        fluorescence: np.ndarray,
        truncation_method: str = 'fluorescence',
        max_fluorescence_pct: float = 85.0,
        max_slope_pct: float = 50.0,
        auto_truncate: bool = True,
        truncate_cycle: Optional[int] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        max_attempts: int = 5,
        r2_threshold: float = 0.999,
        verbose: bool = False,
        method: str = None  # Ignored - kept for compatibility
    ) -> Dict[str, float]:
        """
        Fit MAK2 model to qPCR data using adaptive multi-start optimization.
        
        The optimizer tries different random initial guesses until R² ≥ r2_threshold
        or max_attempts is reached. Most good curves fit well on the first attempt.
        
        Parameters
        ----------
        cycles : np.ndarray
            Cycle numbers
        fluorescence : np.ndarray
            Fluorescence measurements
        truncation_method : str
            'fluorescence' or 'slope' (default: 'fluorescence')
        max_fluorescence_pct : float
            Truncate at this % of max fluorescence (default: 85%)
        max_slope_pct : float
            Truncate when slope drops below this % of max (default: 50%)
        auto_truncate : bool
            Apply automatic truncation (default: True)
        truncate_cycle : int, optional
            Manual truncation cycle (overrides auto_truncate)
        bounds : dict, optional
            Parameter bounds as {'param': (min, max)}
            If not provided, uses data-driven bounds from exponential fits
        max_attempts : int
            Maximum optimization attempts (default: 5)
        r2_threshold : float
            Stop when R² exceeds this value (default: 0.999)
        verbose : bool
            Print fitting progress (default: False)
        method : str, optional
            Ignored (kept for backward compatibility)
            
        Returns
        -------
        params : dict
            Fitted parameters: D0, k, P0, F_bg_intercept, F_bg_slope
        """
        
        # Apply truncation
        if truncate_cycle is not None:
            # Manual truncation
            trunc_idx = np.where(cycles <= truncate_cycle)[0]
            if len(trunc_idx) == 0:
                raise ValueError(f"truncate_cycle {truncate_cycle} is before all data points")
            cycles_fit = cycles[trunc_idx]
            fluorescence_fit = fluorescence[trunc_idx]
            if verbose:
                print(f"Manually truncated at cycle {truncate_cycle}")
                
        elif auto_truncate:
            # Auto-truncate
            if truncation_method == 'fluorescence':
                trunc_idx = find_fluorescence_threshold_cycle(
                    fluorescence, 
                    threshold_pct=max_fluorescence_pct
                )
            elif truncation_method == 'slope':
                trunc_idx = find_slope_threshold_cycle(
                    fluorescence,
                    slope_pct=max_slope_pct
                )
            else:
                raise ValueError(f"Unknown truncation method: {truncation_method}")
            
            cycles_fit = cycles[:trunc_idx + 1]
            fluorescence_fit = fluorescence[:trunc_idx + 1]
            
            if verbose:
                print(f"Auto-truncated at cycle {cycles_fit[-1]:.0f}")
                print(f"Using {len(cycles_fit)}/{len(cycles)} cycles for fitting")
        else:
            # No truncation
            cycles_fit = cycles
            fluorescence_fit = fluorescence
        
        # Use analytical parameter estimation from exponential fits
        if bounds is None:
            bounds = {}
        
        # Try analytical estimation for data-driven bounds and initial guess
        use_analytical_init = 'D0' not in bounds  # Only if user didn't provide custom bounds
        
        if use_analytical_init:
            try:
                estimates, analytical_bounds = estimate_MAK2_params_from_exponential(
                    cycles_fit, fluorescence_fit, 
                    P0_assumed=1.0,
                    verbose=verbose
                )
                
                # Use analytical bounds as defaults (user bounds override)
                for param in analytical_bounds:
                    if param not in bounds:
                        bounds[param] = analytical_bounds[param]
                
                # Store estimates for smart initial guess
                self.analytical_estimates = estimates
                
            except Exception as e:
                # Fallback to old method if analytical estimation fails
                if verbose:
                    print(f"\nWarning: Analytical estimation failed ({str(e)})")
                    print("Falling back to exponential fit bounds...")
                
                try:
                    D0_lower, D0_upper, F_bg_estimate_scalar, _ = estimate_D0_bounds(
                        cycles_fit, fluorescence_fit
                    )
                    
                    # Convert scalar F_bg_estimate to dict format
                    F_bg_est = {
                        'intercept': F_bg_estimate_scalar,
                        'SE_intercept': F_bg_estimate_scalar * 0.05,  # 5% uncertainty
                        'slope': 0.0,
                        'SE_slope': 0.001
                    }
                    
                    if verbose:
                        print(f"\nD0 bounds from exponential fits:")
                        print(f"  Lower: {D0_lower:.2e}")
                        print(f"  Upper: {D0_upper:.2e}")
                except Exception as e2:
                    # Double fallback if both fail
                    if verbose:
                        print(f"\nWarning: Exponential fits failed ({str(e2)})")
                        print("Using default bounds based on data range")
                    
                    F_max = np.max(fluorescence_fit)
                    F_min = np.min(fluorescence_fit)
                    F_range = F_max - F_min
                    
                    D0_lower = F_range / 1e6
                    D0_upper = F_range / 10
                    
                    F_bg_est = {
                        'intercept': F_min,
                        'SE_intercept': F_range * 0.01,
                        'slope': 0.0,
                        'SE_slope': 0.0001
                    }
                
                # Set default bounds (old method)
                default_bounds = {
                    'D0': (D0_lower, D0_upper),
                    'k': (1e-4, 10.0),
                    'P0': (0.5, 100.0),
                    'F_bg_intercept': (
                        max(0, F_bg_est['intercept'] - 3*F_bg_est['SE_intercept']),
                        F_bg_est['intercept'] + 3*F_bg_est['SE_intercept']
                    ),
                    'F_bg_slope': (
                        F_bg_est['slope'] - 5*F_bg_est['SE_slope'],
                        F_bg_est['slope'] + 5*F_bg_est['SE_slope']
                    )
                }
                
                # Update with any user-provided bounds
                default_bounds.update(bounds)
                bounds = default_bounds
                self.analytical_estimates = None
        else:
            # User provided custom bounds - don't use analytical estimation
            # Ensure all required bounds are present
            self.analytical_estimates = None
            if 'k' not in bounds:
                bounds['k'] = (1e-4, 10.0)
            if 'P0' not in bounds:
                bounds['P0'] = (0.5, 100.0)
            if 'F_bg_intercept' not in bounds:
                F_min = np.min(fluorescence_fit)
                F_max = np.max(fluorescence_fit)
                F_range = F_max - F_min
                bounds['F_bg_intercept'] = (max(0, F_min - 0.1*F_range), F_min + 0.1*F_range)
            if 'F_bg_slope' not in bounds:
                bounds['F_bg_slope'] = (-0.001, 0.001)
        
        if verbose:
            print(f"\n=== MAK2 Model Fitting ===")
            print(f"Target R²: ≥ {r2_threshold}")
            print(f"Max attempts: {max_attempts}")
        
        # Adaptive multi-start optimization with bounds adjustment
        best_params = None
        best_r2 = -np.inf
        n_bounds_adjustments = 0  # Track number of adjustments
        max_bounds_adjustments = 3  # Allow up to 3 adjustments
        
        for attempt in range(1, max_attempts + 1):
            try:
                params, r2 = self._fit_attempt(
                    cycles_fit, 
                    fluorescence_fit, 
                    bounds, 
                    seed=attempt
                )
                
                if verbose:
                    print(f"  Attempt {attempt}: R² = {r2:.6f}, SSR = {params['ssr']:.6f}")
                
                # Calculate normalized SSR for quality check
                F_max = np.max(fluorescence_fit)
                F_min = np.min(fluorescence_fit)
                F_range = F_max - F_min
                
                # Normalized SSR: SSR / (F_range^2 * n_points)
                # Typical good fits have normalized SSR < 0.0001
                n_points = len(fluorescence_fit)
                normalized_ssr = params['ssr'] / (F_range**2 * n_points)
                
                # Flag if SSR is suspiciously high (likely local minimum)
                ssr_threshold = 0.0002  # Tunable threshold
                ssr_too_high = normalized_ssr > ssr_threshold
                
                if verbose and ssr_too_high:
                    print(f"    ⚠️  High normalized SSR: {normalized_ssr:.6f} > {ssr_threshold:.6f}")
                    print(f"       Likely local minimum - will retry with adjusted bounds")
                
                # Check if any parameters are at their bounds (within tolerance)
                at_bound_info = []
                tolerance = 0.01  # 1% of bound value
                
                # Check k at upper bound (most common issue)
                if abs(params['k'] - bounds['k'][1]) / bounds['k'][1] < tolerance:
                    at_bound_info.append('k at upper bound')
                
                # Check k at lower bound
                if abs(params['k'] - bounds['k'][0]) / bounds['k'][0] < tolerance:
                    at_bound_info.append('k at lower bound')
                
                # Check P0 at lower bound
                if abs(params['P0'] - bounds['P0'][0]) / bounds['P0'][0] < tolerance:
                    at_bound_info.append('P0 at lower bound')
                
                # Check D0 at bounds
                if abs(params['D0'] - bounds['D0'][0]) / bounds['D0'][0] < tolerance:
                    at_bound_info.append('D0 at lower bound')
                if abs(params['D0'] - bounds['D0'][1]) / bounds['D0'][1] < tolerance:
                    at_bound_info.append('D0 at upper bound')
                
                # If parameters at bound and R² is poor, adjust bounds and retry
                if at_bound_info and r2 < r2_threshold and attempt < max_attempts and n_bounds_adjustments < max_bounds_adjustments:
                    if verbose:
                        print(f"    ⚠ {', '.join(at_bound_info)}")
                    
                    # Adjust bounds based on what's hitting
                    if 'k at upper bound' in at_bound_info:
                        # k maxing out usually means P0 is too low
                        old_P0_min = bounds['P0'][0]
                        bounds['P0'] = (old_P0_min * 10, bounds['P0'][1])
                        if verbose:
                            print(f"    → Increasing P0 lower bound: {old_P0_min:.3f} → {bounds['P0'][0]:.3f}")
                        n_bounds_adjustments += 1
                        continue  # Don't count this as a valid attempt
                    
                    elif 'P0 at lower bound' in at_bound_info:
                        # P0 hitting lower bound - increase it
                        old_P0_min = bounds['P0'][0]
                        bounds['P0'] = (old_P0_min * 5, bounds['P0'][1])
                        if verbose:
                            print(f"    → Increasing P0 lower bound: {old_P0_min:.3f} → {bounds['P0'][0]:.3f}")
                        n_bounds_adjustments += 1
                        continue
                    
                    elif 'D0 at upper bound' in at_bound_info:
                        # D0 hitting upper bound - widen it
                        old_D0_max = bounds['D0'][1]
                        bounds['D0'] = (bounds['D0'][0], old_D0_max * 10)
                        if verbose:
                            print(f"    → Increasing D0 upper bound: {old_D0_max:.2e} → {bounds['D0'][1]:.2e}")
                        n_bounds_adjustments += 1
                        continue
                    
                    elif 'k at lower bound' in at_bound_info:
                        # k hitting lower bound - decrease it
                        old_k_min = bounds['k'][0]
                        bounds['k'] = (old_k_min / 10, bounds['k'][1])
                        if verbose:
                            print(f"    → Decreasing k lower bound: {old_k_min:.6f} → {bounds['k'][0]:.6f}")
                        n_bounds_adjustments += 1
                        continue
                
                # If SSR is high and k is small, likely stuck in local minimum
                # Try increasing k lower bound to force larger k values
                if ssr_too_high and params['k'] < bounds['k'][1] * 0.1 and n_bounds_adjustments < max_bounds_adjustments:
                    if verbose:
                        print(f"    ⚠ High SSR + small k → Likely local minimum")
                        print(f"    → Increasing k lower bound to escape")
                    old_k_min = bounds['k'][0]
                    old_k_max = bounds['k'][1]
                    # Shift k bounds upward significantly
                    bounds['k'] = (old_k_min * 10, old_k_max * 2)
                    if verbose:
                        print(f"    → New k bounds: [{bounds['k'][0]:.6f}, {bounds['k'][1]:.6f}]")
                    n_bounds_adjustments += 1
                    continue
                
                # Track best result
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = params
                
                # Stop if threshold met and no parameters at bound
                if r2 >= r2_threshold and not at_bound_info:
                    if verbose:
                        print(f"  ✓ Threshold met after {attempt} attempt(s)")
                    self.n_attempts = attempt
                    break
                
                # Warn if threshold met but parameters at bound
                if r2 >= r2_threshold and at_bound_info:
                    if verbose:
                        print(f"  ⚠ R² threshold met but: {', '.join(at_bound_info)}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Attempt {attempt}: Failed - {type(e).__name__}: {str(e)}")
                # Store the last error for debugging
                last_error = e
                continue
        else:
            # Max attempts reached
            self.n_attempts = max_attempts
            if verbose and best_r2 < r2_threshold:
                print(f"  ⚠ Stopped after {max_attempts} attempts (best R² = {best_r2:.6f})")
        
        if best_params is None:
            error_msg = "All optimization attempts failed"
            if 'last_error' in locals():
                error_msg += f". Last error: {type(last_error).__name__}: {str(last_error)}"
            raise RuntimeError(error_msg)
        
        # Final check: warn if best fit has parameters at bounds
        if verbose and best_params is not None:
            final_at_bound = []
            tolerance = 0.01
            
            if abs(best_params['k'] - bounds['k'][1]) / bounds['k'][1] < tolerance:
                final_at_bound.append(f"k={best_params['k']:.4f} at upper bound")
            if abs(best_params['k'] - bounds['k'][0]) / bounds['k'][0] < tolerance:
                final_at_bound.append(f"k={best_params['k']:.6f} at lower bound")
            if abs(best_params['P0'] - bounds['P0'][0]) / bounds['P0'][0] < tolerance:
                final_at_bound.append(f"P0={best_params['P0']:.4f} at lower bound")
            if abs(best_params['D0'] - bounds['D0'][0]) / bounds['D0'][0] < tolerance:
                final_at_bound.append(f"D0={best_params['D0']:.2e} at lower bound")
            if abs(best_params['D0'] - bounds['D0'][1]) / bounds['D0'][1] < tolerance:
                final_at_bound.append(f"D0={best_params['D0']:.2e} at upper bound")
            
            if final_at_bound:
                print(f"\n  ⚠️ WARNING: Best fit has parameters at bounds:")
                for warning in final_at_bound:
                    print(f"    - {warning}")
                print(f"  → Fit may be unreliable (R² = {best_r2:.6f})")
        
        # Additional quality check: SSR relative to signal
        if verbose:
            F_max = np.max(fluorescence_fit)
            F_min = np.min(fluorescence_fit)
            F_range = F_max - F_min
            ssr = best_params['ssr']
            
            # SSR should be << signal range squared
            # Typical good fit: SSR < 0.01 * F_range^2
            ssr_threshold = 0.01 * (F_range ** 2)
            ssr_ratio = ssr / (F_range ** 2)
            
            if ssr > ssr_threshold:
                print(f"\n  ⚠️ WARNING: High SSR relative to signal:")
                print(f"    - SSR = {ssr:.6f}")
                print(f"    - Signal range = {F_range:.4f}")
                print(f"    - SSR/(F_range²) = {ssr_ratio:.4f} (threshold: 0.01)")
                print(f"    - Likely local minimum with poor parameters")
        
        # Automatic retry if SSR too high (likely local minimum)
        F_max = np.max(fluorescence_fit)
        F_min = np.min(fluorescence_fit)
        F_range = F_max - F_min
        ssr = best_params['ssr']
        ssr_threshold = 0.01 * (F_range ** 2)
        
        if ssr > ssr_threshold and 'retry_attempted' not in locals():
            if verbose:
                print(f"\n  🔄 Attempting retry with adjusted initial conditions...")
                print(f"    - Increasing k bounds (shift upward by 5×)")
                print(f"    - Decreasing P0 bounds (shift downward by 0.5×)")
            
            # Adjust bounds to escape local minimum
            # Increase k (local minimum often has k too small)
            old_k_bounds = bounds['k']
            bounds['k'] = (old_k_bounds[0] * 5, old_k_bounds[1] * 3)
            
            # Decrease P0 (compensate for larger k)
            old_P0_bounds = bounds['P0']
            bounds['P0'] = (old_P0_bounds[0] * 0.5, old_P0_bounds[1] * 0.7)
            
            if verbose:
                print(f"    - Old k bounds: [{old_k_bounds[0]:.6f}, {old_k_bounds[1]:.6f}]")
                print(f"    - New k bounds: [{bounds['k'][0]:.6f}, {bounds['k'][1]:.6f}]")
                print(f"    - Old P0 bounds: [{old_P0_bounds[0]:.2f}, {old_P0_bounds[1]:.2f}]")
                print(f"    - New P0 bounds: [{bounds['P0'][0]:.2f}, {bounds['P0'][1]:.2f}]")
            
            # Mark that we've attempted retry to avoid infinite loop
            retry_attempted = True
            
            # Store old best for comparison
            old_ssr = ssr
            old_params = best_params.copy()
            
            # Retry optimization with adjusted bounds
            retry_best_params = None
            retry_best_r2 = -np.inf
            
            for attempt in range(1, max_attempts + 1):
                try:
                    params, r2 = self._fit_attempt(
                        cycles_fit, 
                        fluorescence_fit, 
                        bounds, 
                        seed=attempt + 100  # Different seed from first round
                    )
                    
                    if verbose:
                        print(f"    Retry attempt {attempt}: R² = {r2:.6f}, SSR = {params['ssr']:.6f}")
                    
                    if r2 > retry_best_r2:
                        retry_best_r2 = r2
                        retry_best_params = params
                    
                    if r2 >= r2_threshold:
                        break
                        
                except Exception as e:
                    if verbose:
                        print(f"    Retry attempt {attempt}: Failed - {type(e).__name__}")
                    continue
            
            # Use retry result if it's better (lower SSR)
            if retry_best_params is not None:
                retry_ssr = retry_best_params['ssr']
                if verbose:
                    print(f"\n  📊 Comparing results:")
                    print(f"    Original: SSR = {old_ssr:.6f}, k = {old_params['k']:.6f}")
                    print(f"    Retry:    SSR = {retry_ssr:.6f}, k = {retry_best_params['k']:.6f}")
                
                if retry_ssr < old_ssr:
                    if verbose:
                        print(f"  ✅ Retry improved fit (SSR: {old_ssr:.6f} → {retry_ssr:.6f})")
                    best_params = retry_best_params
                    best_r2 = retry_best_r2
                else:
                    if verbose:
                        print(f"  ↩️  Keeping original fit (retry did not improve SSR)")
            else:
                if verbose:
                    print(f"  ↩️  Retry failed, keeping original fit")
        
        # Store results
        self.optimal_params = best_params
        self.cycles_fit = cycles_fit
        self.fluorescence_fit = fluorescence_fit
        
        # Calculate metrics
        self.metrics = self.calculate_fit_metrics()
        
        return best_params
    
    def _fit_attempt(
        self,
        cycles: np.ndarray,
        fluorescence: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
        seed: int
    ) -> Tuple[Dict[str, float], float]:
        """
        Single fitting attempt with random initial guess.
        
        Parameters
        ----------
        cycles : np.ndarray
            Cycle numbers
        fluorescence : np.ndarray
            Fluorescence values
        bounds : dict
            Parameter bounds
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        params : dict
            Fitted parameters
        r2 : float
            R² of fit
        """
        np.random.seed(seed)
        
        # Use analytical estimates for first attempt (seed=1), random for others
        if seed == 1 and hasattr(self, 'analytical_estimates') and self.analytical_estimates is not None:
            # Smart initial guess from analytical estimation
            D0_init = self.analytical_estimates['D0']
            k_init = self.analytical_estimates['k']
            P0_init = self.analytical_estimates['P0']
            F_bg_int_init = self.analytical_estimates['F_bg_intercept']
            F_bg_slope_init = self.analytical_estimates['F_bg_slope']
        else:
            # Random initial guess within bounds
            D0_init = 10**(np.random.uniform(
                np.log10(bounds['D0'][0]), 
                np.log10(bounds['D0'][1])
            ))
            k_init = np.random.uniform(bounds['k'][0], bounds['k'][1])
            P0_init = np.random.uniform(bounds['P0'][0], bounds['P0'][1])
            F_bg_int_init = np.random.uniform(
                bounds['F_bg_intercept'][0], 
                bounds['F_bg_intercept'][1]
            )
            F_bg_slope_init = np.random.uniform(
                bounds['F_bg_slope'][0], 
                bounds['F_bg_slope'][1]
            )
        
        p0 = [D0_init, k_init, P0_init, F_bg_int_init, F_bg_slope_init]
        
        # Prepare bounds for curve_fit
        lower_bounds = [
            bounds['D0'][0],
            bounds['k'][0],
            bounds['P0'][0],
            bounds['F_bg_intercept'][0],
            bounds['F_bg_slope'][0]
        ]
        upper_bounds = [
            bounds['D0'][1],
            bounds['k'][1],
            bounds['P0'][1],
            bounds['F_bg_intercept'][1],
            bounds['F_bg_slope'][1]
        ]
        
        # Fit using Trust Region Reflective (supports bounds, similar performance to LM)
        popt, _ = curve_fit(
            lambda n, D0, k, P0, bg_int, bg_slope: self.model.simulate_to_cycle(
                D0, k, P0, n, bg_int, bg_slope
            ),
            cycles,
            fluorescence,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            method='trf',  # Trust Region Reflective (supports bounds)
            maxfev=10000
        )
        
        # Calculate R² and SSR
        y_pred = self.model.simulate_to_cycle(
            popt[0], popt[1], popt[2], cycles, popt[3], popt[4]
        )
        r2 = calculate_r2(fluorescence, y_pred)
        
        # Calculate SSR for quality checks
        residuals = fluorescence - y_pred
        ssr = np.sum(residuals**2)
        
        params = {
            'D0': popt[0],
            'k': popt[1],
            'P0': popt[2],
            'F_bg_intercept': popt[3],
            'F_bg_slope': popt[4],
            'ssr': ssr  # Add SSR to params
        }
        
        return params, r2
    
    def calculate_fit_metrics(self) -> Dict[str, float]:
        """
        Calculate fit quality metrics.
        
        Returns
        -------
        metrics : dict
            r_squared, rmse, mae, mape, nrmse, ssr, aic, bic, reduced_chi_sq,
            n_points, n_params, dof
        """
        if self.optimal_params is None:
            raise ValueError("No fitted parameters. Run fit() first.")
        
        # Predict
        y_pred = self.predict(self.cycles_fit)
        y_true = self.fluorescence_fit
        
        # R²
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # NRMSE (normalized by range)
        data_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / data_range if data_range > 0 else 0.0
        
        # SSR
        ssr = ss_res
        
        # AIC and BIC (information criteria)
        n = len(y_true)
        k = 5  # Number of parameters (D0, k, P0, F_bg_intercept, F_bg_slope)
        
        # AIC = 2k + n*ln(SSR/n)
        aic = 2*k + n*np.log(ssr/n) if ssr > 0 else np.inf
        
        # BIC = k*ln(n) + n*ln(SSR/n)
        bic = k*np.log(n) + n*np.log(ssr/n) if ssr > 0 else np.inf
        
        # Reduced chi-squared (SSR per degree of freedom)
        dof = n - k  # degrees of freedom
        reduced_chi_sq = ssr / dof if dof > 0 else np.inf
        
        return {
            'r_squared': r_squared,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'nrmse': nrmse,
            'ssr': ssr,
            'aic': aic,
            'bic': bic,
            'reduced_chi_sq': reduced_chi_sq,
            'n_points': n,
            'n_params': k,
            'dof': dof
        }
    
    def predict(
        self,
        cycles: np.ndarray,
        params: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Predict fluorescence at given cycles using fitted parameters.
        
        Parameters
        ----------
        cycles : np.ndarray
            Cycle numbers to predict
        params : dict, optional
            Parameters to use (default: use fitted parameters)
            
        Returns
        -------
        fluorescence : np.ndarray
            Predicted fluorescence values
        """
        if params is None:
            if self.optimal_params is None:
                raise ValueError("No fitted parameters available. Run fit() first.")
            params = self.optimal_params
        
        F_pred = self.model.simulate_to_cycle(
            D0=params['D0'],
            k=params['k'],
            P0=params['P0'],
            cycles=cycles,
            F_bg_intercept=params['F_bg_intercept'],
            F_bg_slope=params['F_bg_slope']
        )
        
        return F_pred
    
    def plot_fit(self):
        """
        Create a plot of the fitted model vs data.
        
        Returns a Plotly figure object.
        """
        if self.optimal_params is None:
            raise ValueError("No fitted parameters. Run fit() first.")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Predict
        y_pred = self.predict(self.cycles_fit)
        residuals = self.fluorescence_fit - y_pred
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("MAK2 Fit", "Residuals"),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )
        
        # Data points
        fig.add_trace(
            go.Scatter(
                x=self.cycles_fit,
                y=self.fluorescence_fit,
                mode='markers',
                name='Data',
                marker=dict(size=8, color='blue')
            ),
            row=1, col=1
        )
        
        # Model fit
        fig.add_trace(
            go.Scatter(
                x=self.cycles_fit,
                y=y_pred,
                mode='lines',
                name='MAK2 Fit',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Residuals
        fig.add_trace(
            go.Scatter(
                x=self.cycles_fit,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6, color='green'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Layout
        fig.update_xaxes(title_text="Cycle", row=2, col=1)
        fig.update_yaxes(title_text="Fluorescence", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=f"R² = {self.metrics['r_squared']:.6f}"
        )
        
        return fig
