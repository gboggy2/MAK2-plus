"""
Bootstrap uncertainty quantification for MAK2+ model parameters.

Implements parametric bootstrap by resampling residuals from the fitted model,
leveraging analytical parameter estimation for efficient refitting.

Author: Greg Boggy, PhD
Date: December 12, 2024
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time
from tqdm import tqdm

from mak2_model import MAK2Model
from optimizer import MAK2Optimizer


@dataclass
class BootstrapResults:
    """Container for bootstrap uncertainty analysis results."""
    
    # Point estimates (from original fit)
    D0_point: float
    k_point: float
    P0_point: float
    
    # Bootstrap distributions (raw samples)
    D0_samples: np.ndarray
    k_samples: np.ndarray
    P0_samples: np.ndarray
    
    # Confidence intervals
    D0_ci: Tuple[float, float]
    k_ci: Tuple[float, float]
    P0_ci: Tuple[float, float]
    
    # Derived quantities
    efficiency_point: float
    efficiency_ci: Tuple[float, float]
    
    # Metadata
    n_bootstrap: int
    confidence_level: float
    n_successful: int
    runtime_seconds: float
    
    def summary_dict(self) -> Dict:
        """Return summary statistics as dictionary."""
        return {
            'D0': {
                'estimate': self.D0_point,
                'ci_lower': self.D0_ci[0],
                'ci_upper': self.D0_ci[1],
                'std': np.std(self.D0_samples)
            },
            'k': {
                'estimate': self.k_point,
                'ci_lower': self.k_ci[0],
                'ci_upper': self.k_ci[1],
                'std': np.std(self.k_samples)
            },
            'P0': {
                'estimate': self.P0_point,
                'ci_lower': self.P0_ci[0],
                'ci_upper': self.P0_ci[1],
                'std': np.std(self.P0_samples)
            },
            'efficiency': {
                'estimate': self.efficiency_point,
                'ci_lower': self.efficiency_ci[0],
                'ci_upper': self.efficiency_ci[1]
            },
            'metadata': {
                'n_bootstrap': self.n_bootstrap,
                'n_successful': self.n_successful,
                'success_rate': self.n_successful / self.n_bootstrap,
                'confidence_level': self.confidence_level,
                'runtime_seconds': self.runtime_seconds
            }
        }


class BootstrapAnalyzer:
    """
    Bootstrap uncertainty quantification for MAK2 model parameters.
    
    Uses parametric bootstrap by resampling residuals from fitted model.
    Leverages analytical parameter estimation for fast refitting.
    """
    
    def __init__(
        self,
        model: MAK2Model,
        optimizer: MAK2Optimizer,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Initialize bootstrap analyzer.
        
        Parameters
        ----------
        model : MAK2Model
            MAK2 model instance with configuration
        optimizer : MAK2Optimizer
            Optimizer instance for fitting
        n_bootstrap : int, default=1000
            Number of bootstrap iterations
        confidence_level : float, default=0.95
            Confidence level for intervals (e.g., 0.95 for 95% CI)
        random_seed : int, optional
            Random seed for reproducibility
        show_progress : bool, default=True
            Show progress bar during bootstrap
        """
        self.model = model
        self.optimizer = optimizer
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.show_progress = show_progress
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_bootstrap(
        self,
        cycles: np.ndarray,
        fluorescence: np.ndarray,
        original_params: Dict[str, float],
        original_fit: np.ndarray
    ) -> BootstrapResults:
        """
        Run parametric bootstrap uncertainty analysis.
        
        Parameters
        ----------
        cycles : np.ndarray
            Cycle numbers
        fluorescence : np.ndarray
            Measured fluorescence values
        original_params : dict
            Original fitted parameters {'D0', 'k', 'P0'}
        original_fit : np.ndarray
            Original model predictions
        
        Returns
        -------
        BootstrapResults
            Complete bootstrap results with CIs
        """
        start_time = time.time()
        
        # Calculate residuals from original fit
        residuals = fluorescence - original_fit
        
        # Storage for bootstrap samples
        D0_samples = []
        k_samples = []
        P0_samples = []
        
        # Track errors for debugging
        error_counts = {}
        last_errors = []
        
        # Calculate original efficiency for reference
        efficiency_point = self._calculate_efficiency(
            original_params['D0'],
            original_params['k'],
            original_params['P0']
        )
        
        # Run bootstrap iterations
        iterator = range(self.n_bootstrap)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Bootstrap iterations")
        
        # Create random number generator with explicit state
        rng = np.random.RandomState()
        
        for i in iterator:
            try:
                # Resample residuals with replacement - use explicit RNG
                resampled_residuals = rng.choice(
                    residuals,
                    size=len(residuals),
                    replace=True
                )
                
                # Generate bootstrap sample by adding resampled residuals
                bootstrap_fluorescence = original_fit + resampled_residuals
                
                # Use tight bounds based on original fit for fast convergence
                # This is faster than letting analytical estimation run on noisy bootstrap samples
                bootstrap_bounds = {
                    'D0': (original_params['D0'] / 10, original_params['D0'] * 10),
                    'k': (max(1e-4, original_params['k'] / 10), min(10.0, original_params['k'] * 10)),
                    'P0': (max(0.5, original_params['P0'] / 5), min(100.0, original_params['P0'] * 5)),
                    'F_bg_intercept': (
                        original_params['F_bg_intercept'] - 1.0,
                        original_params['F_bg_intercept'] + 1.0
                    ),
                    'F_bg_slope': (
                        original_params['F_bg_slope'] - 0.01,
                        original_params['F_bg_slope'] + 0.01
                    )
                }
                
                # Fit with provided bounds (skips analytical estimation for speed)
                bootstrap_result = self.optimizer.fit(
                    cycles,
                    bootstrap_fluorescence,
                    auto_truncate=False,
                    bounds=bootstrap_bounds,
                    verbose=False,
                    max_attempts=2,  # Bootstrap samples similar to original, 2 attempts enough
                    r2_threshold=0.90  # Lower threshold OK for bootstrap
                )
                
                # If we got here, fit succeeded - store parameters
                D0_samples.append(bootstrap_result['D0'])
                k_samples.append(bootstrap_result['k'])
                P0_samples.append(bootstrap_result['P0'])
                
            except Exception as e:
                # Track error types (minimal tracking for speed)
                error_type = type(e).__name__
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                # Only capture first error for debugging
                if len(last_errors) == 0:
                    last_errors.append(f"{error_type}: {str(e)}")
                
                continue
        
        # Convert to arrays
        D0_samples = np.array(D0_samples)
        k_samples = np.array(k_samples)
        P0_samples = np.array(P0_samples)
        
        # Check if we have enough successful iterations
        n_successful = len(D0_samples)
        if n_successful < 10:
            error_msg = (
                f"Bootstrap failed: only {n_successful}/{self.n_bootstrap} iterations succeeded.\n"
                f"Error summary:\n"
            )
            for error_type, count in error_counts.items():
                error_msg += f"  - {error_type}: {count} times\n"
            if last_errors:
                error_msg += f"\nFIRST ERROR WITH FULL TRACEBACK:\n{last_errors[0]}\n"
            error_msg += "\nThis usually means the original fit was poor or the data is problematic."
            raise ValueError(error_msg)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        D0_ci = (
            np.percentile(D0_samples, lower_percentile),
            np.percentile(D0_samples, upper_percentile)
        )
        k_ci = (
            np.percentile(k_samples, lower_percentile),
            np.percentile(k_samples, upper_percentile)
        )
        P0_ci = (
            np.percentile(P0_samples, lower_percentile),
            np.percentile(P0_samples, upper_percentile)
        )
        
        # Calculate efficiency CI
        efficiency_samples = np.array([
            self._calculate_efficiency(D0, k, P0)
            for D0, k, P0 in zip(D0_samples, k_samples, P0_samples)
        ])
        efficiency_ci = (
            np.percentile(efficiency_samples, lower_percentile),
            np.percentile(efficiency_samples, upper_percentile)
        )
        
        runtime = time.time() - start_time
        
        return BootstrapResults(
            D0_point=original_params['D0'],
            k_point=original_params['k'],
            P0_point=original_params['P0'],
            D0_samples=D0_samples,
            k_samples=k_samples,
            P0_samples=P0_samples,
            D0_ci=D0_ci,
            k_ci=k_ci,
            P0_ci=P0_ci,
            efficiency_point=efficiency_point,
            efficiency_ci=efficiency_ci,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level,
            n_successful=len(D0_samples),
            runtime_seconds=runtime
        )
    
    def _calculate_efficiency(self, D0: float, k: float, P0: float) -> float:
        """
        Calculate PCR efficiency from MAK2 parameters.
        
        Efficiency is calculated at the exponential phase (low D/P ratio).
        """
        # At low D/P, MAK2 reduces to exponential: D_n+1 = D_n * (1 + k*P0)
        # Efficiency E = 1 + k*P0, so amplification factor = 2^E
        efficiency = 1 + k * P0
        return efficiency
    
    def plot_bootstrap_distributions(
        self,
        results: BootstrapResults,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Plot bootstrap distributions for all parameters.
        
        Parameters
        ----------
        results : BootstrapResults
            Bootstrap results to plot
        figsize : tuple, default=(15, 5)
            Figure size (width, height)
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # D0 distribution
        axes[0].hist(results.D0_samples, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(results.D0_point, color='red', linestyle='--', linewidth=2,
                       label=f'Point: {results.D0_point:.2e}')
        axes[0].axvline(results.D0_ci[0], color='green', linestyle=':', linewidth=2)
        axes[0].axvline(results.D0_ci[1], color='green', linestyle=':', linewidth=2,
                       label=f'95% CI')
        axes[0].set_xlabel('D0', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].set_title('D0 Bootstrap Distribution', fontsize=11)
        axes[0].legend(fontsize=8)
        # Only use log scale if data spans more than 1 order of magnitude
        if results.D0_samples.max() / results.D0_samples.min() > 10:
            try:
                axes[0].set_xscale('log')
            except:
                pass  # Skip log scale if it fails
        
        # k distribution
        axes[1].hist(results.k_samples, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(results.k_point, color='red', linestyle='--', linewidth=2,
                       label=f'Point: {results.k_point:.4f}')
        axes[1].axvline(results.k_ci[0], color='green', linestyle=':', linewidth=2)
        axes[1].axvline(results.k_ci[1], color='green', linestyle=':', linewidth=2,
                       label=f'95% CI')
        axes[1].set_xlabel('k', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        axes[1].set_title('k Bootstrap Distribution', fontsize=11)
        axes[1].legend(fontsize=8)
        # Only use log scale if data spans more than 1 order of magnitude
        if results.k_samples.max() / results.k_samples.min() > 10:
            try:
                axes[1].set_xscale('log')
            except:
                pass
        
        # P0 distribution
        axes[2].hist(results.P0_samples, bins=50, alpha=0.7, edgecolor='black')
        axes[2].axvline(results.P0_point, color='red', linestyle='--', linewidth=2,
                       label=f'Point: {results.P0_point:.2e}')
        axes[2].axvline(results.P0_ci[0], color='green', linestyle=':', linewidth=2)
        axes[2].axvline(results.P0_ci[1], color='green', linestyle=':', linewidth=2,
                       label=f'95% CI')
        axes[2].set_xlabel('P0', fontsize=10)
        axes[2].set_ylabel('Frequency', fontsize=10)
        axes[2].set_title('P0 Bootstrap Distribution', fontsize=11)
        axes[2].legend(fontsize=8)
        # Only use log scale if data spans more than 1 order of magnitude
        if results.P0_samples.max() / results.P0_samples.min() > 10:
            try:
                axes[2].set_xscale('log')
            except:
                pass
        
        plt.tight_layout()
        return fig


def bootstrap_parameter_uncertainty(
    cycles: np.ndarray,
    fluorescence: np.ndarray,
    original_params: Dict[str, float],
    original_fit: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
    show_progress: bool = True
) -> BootstrapResults:
    """
    Convenience function for bootstrap uncertainty analysis.
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Measured fluorescence values
    original_params : dict
        Original fitted parameters {'D0', 'k', 'P0'}
    original_fit : np.ndarray
        Original model predictions
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for intervals
    random_seed : int, optional
        Random seed for reproducibility
    show_progress : bool, default=True
        Show progress bar
    
    Returns
    -------
    BootstrapResults
        Complete bootstrap results with confidence intervals
    
    Example
    -------
    >>> # After fitting model
    >>> bootstrap_results = bootstrap_parameter_uncertainty(
    ...     cycles=cycles,
    ...     fluorescence=fluorescence,
    ...     original_params={'D0': 100, 'k': 0.1, 'P0': 1.0},
    ...     original_fit=fitted_curve,
    ...     n_bootstrap=1000
    ... )
    >>> print(bootstrap_results.summary_dict())
    """
    # Create model and optimizer instances
    model = MAK2Model()
    optimizer = MAK2Optimizer(model)
    
    # Create analyzer and run bootstrap
    analyzer = BootstrapAnalyzer(
        model=model,
        optimizer=optimizer,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
        show_progress=show_progress
    )
    
    return analyzer.run_bootstrap(
        cycles=cycles,
        fluorescence=fluorescence,
        original_params=original_params,
        original_fit=original_fit
    )
