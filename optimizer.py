"""Tiered MAK2 fitter with escalation, fallback, and quality gates.

This module turns a single per-well fluorescence array into fitted
``(D0, k, P0, F_bg_intercept, F_bg_slope)`` parameters by walking
through a sequence of optimisation strategies — cheaper / more
constrained at the top, slower / more exhaustive at the bottom —
and stopping at the first one that produces a fit clearing the R²
threshold.

Why a tier system instead of one optimisation method?
=====================================================

The MAK2 loss surface has shallow local minima around the true
optimum and a few sharp wrong basins (e.g. background-absorbs-D0
solutions that look great on R² but quantify nonsense). No single
optimiser handles every well well:

- **Trust Region Reflective (TRF)** is fast and well-behaved when
  given a reasonable starting point — it's perfect for "easy" wells
  with strong signal and clear sigmoid shape.
- **Latin Hypercube Sampling + TRF** rescues "hard" wells by seeding
  TRF from many corners of the parameter box, reducing the chance of
  getting stuck in a local minimum.
- **Differential Evolution** is the heavy hitter — global stochastic
  search that ignores starting points entirely, used only when LHS
  also fails to find a good fit.

Doing all three on every well would be wasteful (DE alone is ~10×
slower than TRF). The tier system tries the cheap option first and
escalates only when needed.

Tier overview
=============

The tiers are tagged in ``optimal_params['tier']`` so downstream
code (``run_batch.run_pass1``) can record which tier produced each
well's fit:

  - ``T1-Full``     — TRF with all 5 parameters free, single attempt.
  - ``T1-Fixed``    — TRF with background pinned to the per-well
                       pre-estimate (``fix_background=True``). Avoids
                       background-absorbs-D0 failure mode.
  - ``T1.5``        — TRF retry with adjusted bounds when the first
                       attempt's residuals show specific failure
                       patterns (k-stuck-at-bound, residuals-elbow,
                       etc.). Includes the well-specific pattern
                       blocks (X6.R4.2, X5.R3.1, …) that were tuned
                       against problematic Rutledge-dataset wells.
  - ``T2-LHS``      — Latin Hypercube multi-start: 3D LHS samples
                       (D0, k, P0) with background pinned, fed as
                       initial guesses to TRF.
  - ``T2.5``        — 5D LHS fallback: LHS samples include
                       background, background no longer pinned.
                       Last resort before DE.
  - ``T3-DE``       — Scipy ``differential_evolution`` with the
                       full 5D parameter box. Slow but global.
  - ``T4``          — Plateau-overshoot refit (post-fit correction
                       when the model exceeds the data plateau).

See ``fit()`` for the per-tier orchestration logic.

Quality gates
=============

After a fit clears the R² threshold, several gates check whether
the fit is *meaningful* (not just well-shaped):

  - **R² floor** (Gate 0): R² ≥ 0.999, relaxed to 0.997 for
    late amplifiers (where the fit window includes the last data
    cycle and there's less plateau information).
  - **Fit window width** (Gate 2): ≥ 12 cycles. Narrow windows
    don't constrain the sigmoid enough.
  - **Sigmoid vs linear** (Gate 2b): the MAK2 R² must beat a pure
    linear fit on the same window by ≥ 0.04. Catches the case where
    the optimiser produced a high-R² monotone non-sigmoid fit.
  - **Sigmoid shape** (Gate 3): the fitted curve's second derivative
    must change sign in-window — confirms there's an inflection.
  - **Plateau overshoot** (Tier 4): the fitted plateau height must
    not exceed the data plateau by more than a small tolerance.

Background separation (why ``fix_background`` exists)
=====================================================

Without baseline pinning, a wrong D0 can be hidden by a
correspondingly wrong background — the fit's R² stays high but the
quantification answer is meaningless. The baseline is therefore
estimated separately from the kinetic parameters via two-stage
fitting (see ``mak2_model.pre_estimate_background``) and pinned in
during the optimisation. ``fix_background=True`` is the production
default; ``fix_background=False`` is mostly for diagnostic A/B
comparison.

Public API
==========

  - ``calculate_r2`` — top-level R² helper used by tier-internal code.
  - ``MAK2Optimizer.fit`` — the only entry point you usually call.
  - ``MAK2Optimizer.calculate_fit_metrics`` — R²/RMSE/AIC/BIC/SSR
    on the fit window (called by callers after ``fit()``).
  - ``MAK2Optimizer.predict`` — model prediction at arbitrary cycles.
  - ``MAK2Optimizer.calculate_ct`` — threshold-cycle from the
    fitted curve.
  - ``MAK2Optimizer.check_plateau_overshoot`` — Tier 4 detection
    helper, also called externally by Tier 4 refit logic.

See ``mak2_model.py`` for the underlying biochemical model and
``CLAUDE.md``'s "Phase 1 Per-well Pipeline Unification" note for
the long-term plan to extract the orchestration around this fitter.
"""

# VERSION: 2.0.0 - LHS + Noise + Stuck Detection
print("="*80)
print("🔄 OPTIMIZER MODULE LOADING - VERSION 2.0.0 (LHS + NOISE + STUCK DETECTION)")
print("="*80)

import time
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import qmc
from typing import Tuple, Dict, Optional, Set
from mak2_model import (
    MAK2Model,
    find_slope_threshold_cycle,
    estimate_D0_bounds,
    estimate_MAK2_params_from_exponential
)
from config import RANDOM_SEED


# ── Per-call-site offsets for derived seeds ──────────────────────────────────
# Different stochastic loops in fit() must not share the same seed sequence —
# if they did, two LHS calls (or LHS + retry-loop inits) would explore the
# same corners of the parameter space, weakening the search. Each site below
# adds a unique offset to RANDOM_SEED so derived seeds stay disjoint without
# the user needing to think about it.
#
# In production (RANDOM_SEED is None), every seed below resolves to None and
# scipy/numpy pull fresh entropy on each call — i.e. the optimizer naturally
# varies run-to-run, exposing the optimizer's stochastic spread.
#
# In testing (RANDOM_SEED=<int>), each site gets a deterministic seed
# (RANDOM_SEED + offset + attempt), so the full pipeline is byte-reproducible.
_SEED_OFFSET_LHS_3D       = 0      # Tier 2 LHS sampler
_SEED_OFFSET_LHS_5D       = 100    # Tier 2.5 fallback LHS sampler
_SEED_OFFSET_DE           = 200    # Tier 3 differential evolution
_SEED_OFFSET_TIER1_RETRY  = 1000   # Tier 1.5 per-attempt seeds
_SEED_OFFSET_PATTERN      = 2000   # pattern-based retry seeds
_SEED_OFFSET_TIER2_SECOND = 3000   # Tier 2 secondary attempts
_SEED_OFFSET_TIER25_ATTEMPT = 4000  # Tier 2.5 per-attempt seeds


def _derive_seed(offset: int, attempt: int = 0) -> Optional[int]:
    """Combine ``RANDOM_SEED`` with a per-site offset and a per-attempt index.

    Returns ``None`` when ``RANDOM_SEED is None`` (production / unseeded
    mode) — both scipy and numpy treat ``seed=None`` as "use fresh
    entropy", giving the optimizer its natural stochastic variation.

    Returns a deterministic integer when ``RANDOM_SEED`` is set
    (testing / CI). The combination ``base + offset + attempt`` keeps
    every call site's seed sequence disjoint from every other call
    site's, so the LHS samplers and the per-attempt retry loops
    don't accidentally explore the same parameter-space corners.
    """
    if RANDOM_SEED is None:
        return None
    return RANDOM_SEED + offset + attempt


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination) for fitted-vs-observed arrays.

    Standard formula: ``R² = 1 - SS_res / SS_tot``. Returns 0 when
    ``SS_tot`` is zero (constant data) — the alternative would be
    NaN, but downstream tier-escalation code interprets 0 as "fit
    isn't useful, escalate" which is the correct outcome anyway.
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class MAK2Optimizer:
    """Tiered MAK2 fitter with retry escalation and quality gates.

    Composes a ``MAK2Model`` (the forward simulator) with the multi-tier
    optimisation strategy described in this module's docstring. The
    class is instantiated per-fit because it accumulates per-fit state
    (``optimal_params``, ``cycles_fit``, ``tier_log``, etc.) that
    callers read after ``fit()`` returns; do not share one instance
    across concurrent fits.

    Typical usage::

        opt = MAK2Optimizer()
        params = opt.fit(cycles, fluorescence, fixed_background_values=...)
        metrics = opt.calculate_fit_metrics()
        ct_info = opt.calculate_ct(method='threshold')

    State that ``fit()`` populates on the instance (read by callers
    after fit returns):

      - ``optimal_params``: the best ``{D0, k, P0, F_bg_intercept,
        F_bg_slope}`` parameter dict, plus tier-tagging metadata.
      - ``cycles_fit`` / ``fluorescence_fit``: the (possibly
        truncated) cycle / fluor arrays the fit was actually run on.
        Saved so ``calculate_fit_metrics``, ``predict``, and
        ``calculate_ct`` can reuse them without the caller passing
        them back in.
      - ``metrics``: R²/RMSE/AIC/BIC/SSR computed at fit-end.
      - ``n_attempts``: how many TRF attempts were used.
      - ``tier_log``: list of per-tier timing/improvement dicts —
        feeds the offline benchmarking in ``benchmark_tiers.py``.
    """

    def __init__(self, model: Optional[MAK2Model] = None):
        """Construct an optimizer holding a forward-simulator instance.

        Args:
            model: A ``MAK2Model`` instance to use for forward
                simulation. Optional; a fresh one is created if not
                supplied. Sharing a single ``MAK2Model`` across many
                ``MAK2Optimizer`` instances is fine — the model is
                stateless.
        """
        self.model = model or MAK2Model()
        self.optimal_params = None
        self.cycles_fit = None
        self.fluorescence_fit = None
        self.metrics = None
        self.n_attempts = None
        self.tier_log = []  # one dict per tier: timing, R², improvement, etc.
        # Set True to prevent the Tier 4 plateau-overshoot refit from
        # recursing — the refit itself calls fit() internally.
        self._skip_overshoot_refit = False
    
    def fit(
        self,
        cycles: np.ndarray,
        fluorescence: np.ndarray,
        cycles_after_max: int = 3,
        auto_truncate: bool = True,
        truncate_cycle: Optional[int] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        max_attempts: int = 10,
        r2_threshold: float = 0.999,
        verbose: bool = False,
        fix_background: bool = True,  # Fix background with adaptive fallback
        fixed_background_values: Optional[Dict[str, float]] = None,  # Exact bg values to fix
        disabled_tiers: Optional[Set[str]] = None  # For ablation testing
    ) -> Dict[str, float]:
        """Run the full tiered MAK2 fit and return the best parameters.

        This is the only method most callers need. It internally
        walks the tier escalation described in the module docstring,
        stopping at the first tier whose result clears the R²
        threshold and the quality gates. Side-effects (per-instance
        state populated for downstream calls): see the class
        docstring's "State that ``fit()`` populates" section.

        Pipeline:

          1. **Truncate** the input array if ``auto_truncate`` (default)
             — cuts off cycles past the inflection + ``cycles_after_max``,
             so the fit doesn't get distracted by the noisy tail.
          2. **Compute bounds.** Either uses caller-supplied
             ``bounds`` or runs ``estimate_MAK2_params_from_exponential``
             on the truncated window to derive data-driven box bounds.
          3. **Pin background** if ``fix_background`` (default). Reads
             ``fixed_background_values`` if supplied (caller-known
             baseline regression), else uses the analytical estimates.
          4. **Tier 1 / 1.5** — TRF with the chosen bounds, with
             pattern-based retry if the residuals show known failure
             shapes.
          5. **Tier 2 / 2.5** — LHS-seeded TRF (3D then 5D fallback).
          6. **Tier 3** — Differential Evolution.
          7. **Tier 4** — Plateau-overshoot refit if the chosen fit
             exceeds the data plateau.
          8. **Quality gates** post-fit (R² floor, window width,
             sigmoid-vs-linear discriminator, sigmoid-shape check).

        Args:
            cycles: Per-cycle cycle-number array.
            fluorescence: Per-cycle fluorescence array (same length
                as ``cycles``). Raw RFU or normalised Rn — the model
                is scale-aware via the data-driven bounds.
            cycles_after_max: Truncation offset past the inflection
                cycle. Default 3 includes a few plateau cycles for
                P0 estimation but excludes the bulk of the noisy
                tail.
            auto_truncate: When True, apply automatic truncation via
                ``find_slope_threshold_cycle``. When False, the fit
                uses every cycle (only safe if the input is already
                truncated by the caller).
            truncate_cycle: Manual truncation cycle (override). When
                set, ``auto_truncate`` is ignored.
            bounds: Caller-supplied parameter bounds dict. Keys are
                ``D0``, ``k``, ``P0``, ``F_bg_intercept``,
                ``F_bg_slope``; values are ``(lo, hi)`` tuples. Any
                missing parameter gets data-driven bounds. When
                ``None``, every bound is derived.
            max_attempts: Cap on TRF retry attempts. Default 10. Most
                wells finish in 1–2 attempts; the cap matters only
                for difficult wells.
            r2_threshold: Stop-early R² for tier escalation. Default
                0.999 — qPCR fits with R² > 0.999 are essentially
                noise-limited and not worth further optimisation.
            verbose: Print per-attempt diagnostic information.
                Production callers (``run_batch.run_pass1``) pass
                False to keep the log readable.
            fix_background: When True (production default), pins
                background to the per-well pre-estimate so the
                kinetic optimiser can't trade D0 against background.
                When False, fits all 5 parameters jointly.
            fixed_background_values: Exact background to pin when
                ``fix_background=True``. Comes from the caller's own
                ``pre_estimate_background`` regression on the
                metadata baseline window. Falls back to the
                analytical estimate when ``None``.
            disabled_tiers: Set of tier-name strings to skip. Used
                only for ablation testing in ``benchmark_tiers.py``;
                production callers don't pass this.

        Returns:
            Dict with the fitted parameters and tier-tagging metadata.
            Keys include ``D0``, ``k``, ``P0``, ``F_bg_intercept``,
            ``F_bg_slope``, plus internal flags like ``de_used``,
            ``fallback_succeeded``, ``used_fixed_background`` that
            tell ``run_pass1`` which tier produced the result.

        Raises:
            ValueError: If ``truncate_cycle`` is before all data,
                or if computed bounds are invalid (e.g. the
                pre-existing k-bounds bug for some Boggy wells —
                see the spawned bug-fix task).
            RuntimeError: If every tier and every attempt failed —
                rare in practice, indicates either a pathological
                well or a bug in the upstream bounds derivation.
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
            # Auto-truncate at max slope + cycles_after_max
            trunc_idx = find_slope_threshold_cycle(
                fluorescence,
                cycles_after_max=cycles_after_max
            )
            
            cycles_fit = cycles[:trunc_idx + 1]
            fluorescence_fit = fluorescence[:trunc_idx + 1]
            
            if verbose:
                print(f"Auto-truncated at cycle {cycles_fit[-1]:.0f}")
                print(f"Using {len(cycles_fit)}/{len(cycles)} cycles for fitting")
        else:
            # No truncation
            cycles_fit = cycles
            fluorescence_fit = fluorescence
        
        # Initialize disabled_tiers and tier_log
        if disabled_tiers is None:
            disabled_tiers = set()
        self.tier_log = []
        self._fit_deadline = time.perf_counter() + 20.0  # 20-second timeout
        self._fit_timed_out = False
        _tier1_start = time.perf_counter()

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
                    D0_lower, D0_upper, F_bg_estimate_scalar, fit_info = estimate_D0_bounds(
                        cycles_fit, fluorescence_fit
                    )
                    
                    # Convert scalar F_bg_estimate to dict format
                    F_bg_est = {
                        'intercept': F_bg_estimate_scalar,
                        'SE_intercept': abs(F_bg_estimate_scalar) * 0.05,  # 5% uncertainty (use abs!)
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

                    # D0 is initial template - can be very small for low-template samples
                    # Allow D0 to go very low (no floor) but cap upper bound reasonably
                    D0_lower = F_range / 1000  # Allow very low D0 values
                    D0_upper = F_range / 3  # Allow D0 up to 33% of range for more flexibility
                    
                    F_bg_est = {
                        'intercept': F_min,
                        'SE_intercept': F_range * 0.01,
                        'slope': 0.0,
                        'SE_slope': 0.0001
                    }
                
                # Set default bounds (old method)
                # For P0: estimate from maximum fluorescence
                # At plateau: F_max ≈ D0 + P0 + F_bg
                # Since D0 << P0 typically: P0 ≈ F_max - F_bg
                F_max = np.max(fluorescence_fit)
                F_bg_estimate_at_plateau = F_bg_est['intercept'] + F_bg_est['slope'] * len(fluorescence_fit)
                P0_estimate = F_max - F_bg_estimate_at_plateau
                
                # Set bounds with reasonable margin
                # Lower: allow 10% of estimate (primer could be in excess)
                # Upper: allow 3x estimate (some primer depletion possible)
                P0_lower = max(0.05, P0_estimate * 0.1)
                P0_upper = max(P0_estimate * 3.0, 10.0)
                
                # Ensure minimum range for F_bg parameters to avoid collapsed bounds
                # Use at least 10% of intercept value or absolute minimum 0.05
                se_based_margin = 3 * F_bg_est['SE_intercept']
                percent_based_margin = 0.1 * abs(F_bg_est['intercept'])
                F_bg_int_margin = max(se_based_margin, percent_based_margin, 0.05)

                F_bg_slope_margin = max(5 * F_bg_est['SE_slope'], 0.001)

                if verbose:
                    print(f"  F_bg margins: intercept ±{F_bg_int_margin:.6f}, slope ±{F_bg_slope_margin:.6f}")

                default_bounds = {
                    'D0': (D0_lower, D0_upper),
                    'k': (0.01, 1.2),  # Realistic qPCR range with some flexibility
                    'P0': (P0_lower, P0_upper),  # Data-driven bounds!
                    'F_bg_intercept': (
                        F_bg_est['intercept'] - F_bg_int_margin,  # Allow negative
                        F_bg_est['intercept'] + F_bg_int_margin
                    ),
                    'F_bg_slope': (
                        F_bg_est['slope'] - F_bg_slope_margin,
                        F_bg_est['slope'] + F_bg_slope_margin
                    )
                }
                
                if verbose:
                    print(f"\nP0 bounds from data:")
                    print(f"  F_max: {F_max:.4f}")
                    print(f"  F_bg (at plateau): {F_bg_estimate_at_plateau:.4f}")
                    print(f"  P0 estimate: {P0_estimate:.4f}")
                    print(f"  P0 bounds: [{P0_lower:.4f}, {P0_upper:.4f}]")
                
                # Update with any user-provided bounds
                default_bounds.update(bounds)
                bounds = default_bounds
                self.analytical_estimates = None
        else:
            # User provided custom D0 bounds — keep them, but still run
            # analytical estimation for INITIAL GUESSES (not bounds).
            # Without this, seed=1 in _fit_attempt falls through to random
            # init, missing the exponential-phase D0/k estimate that is
            # critical for convergence on both early-Ct and late-Ct wells.
            try:
                estimates, _ = estimate_MAK2_params_from_exponential(
                    cycles_fit, fluorescence_fit,
                    P0_assumed=1.0,
                    verbose=verbose
                )
                self.analytical_estimates = estimates
            except Exception:
                self.analytical_estimates = None

            # Heuristic fallback: if analytical estimation failed, estimate
            # D0 from the sigmoid shape.  For a standard qPCR curve the
            # signal rises from baseline to plateau over ~10 doublings, so
            # D0 ≈ F_range / 2^(midpoint_cycle - first_cycle).
            # This gives seed=1 a fighting chance instead of random init.
            if self.analytical_estimates is None and len(cycles_fit) >= 5:
                try:
                    _F_max = float(np.max(fluorescence_fit))
                    _F_min = float(np.min(fluorescence_fit))
                    _F_range = _F_max - _F_min
                    _F_mid = (_F_max + _F_min) / 2.0
                    # Find the cycle closest to the midpoint fluorescence
                    _mid_idx = int(np.argmin(np.abs(fluorescence_fit - _F_mid)))
                    _mid_cycle = float(cycles_fit[_mid_idx])
                    _first_cycle = float(cycles_fit[0])
                    _n_doublings = _mid_cycle - _first_cycle
                    if _n_doublings > 0 and _F_range > 0:
                        _D0_est = _F_range / (2.0 ** _n_doublings)
                        # Reasonable k and P0 from data
                        _k_est = 0.3  # typical qPCR efficiency
                        _P0_est = _F_range * 1.5  # primer > signal range
                        self.analytical_estimates = {
                            'D0': np.clip(_D0_est, bounds['D0'][0], bounds['D0'][1]),
                            'k': np.clip(_k_est, bounds.get('k', (0.05, 1.2))[0],
                                         bounds.get('k', (0.05, 1.2))[1]),
                            'P0': np.clip(_P0_est, bounds.get('P0', (0.05, 10))[0],
                                          bounds.get('P0', (0.05, 10))[1]),
                            'F_bg_intercept': _F_min,
                            'F_bg_slope': 0.0,
                        }
                        print(f"  📐 Heuristic D0 estimate: {_D0_est:.2e} "
                              f"(midpoint at cycle {_mid_cycle:.0f}, "
                              f"{_n_doublings:.0f} doublings from start)")
                except Exception:
                    pass  # heuristic failed, fall back to LHS
            # Ensure all required bounds are present
            if 'k' not in bounds:
                bounds['k'] = (0.01, 1.2)  # Realistic qPCR range with some flexibility
            if 'P0' not in bounds:
                # Set P0 bounds from maximum fluorescence
                F_max = np.max(fluorescence_fit)
                F_min = np.min(fluorescence_fit)
                # Rough estimate: P0 ≈ F_max - F_min (assuming F_bg ≈ F_min)
                P0_estimate = F_max - F_min
                P0_lower = max(0.05, P0_estimate * 0.1)
                P0_upper = max(P0_estimate * 3.0, 10.0)
                bounds['P0'] = (P0_lower, P0_upper)
            if 'F_bg_intercept' not in bounds:
                F_min = np.min(fluorescence_fit)
                F_max = np.max(fluorescence_fit)
                F_range = F_max - F_min
                bounds['F_bg_intercept'] = (F_min - 0.1*F_range, F_min + 0.1*F_range)  # Allow negative
            if 'F_bg_slope' not in bounds:
                bounds['F_bg_slope'] = (-0.001, 0.001)
        
        # Handle fixed background parameters
        if fix_background:
            if fixed_background_values is not None:
                # Use exact values passed by caller (from linear regression)
                fixed_F_bg_intercept = fixed_background_values['F_bg_intercept']
                fixed_F_bg_slope = fixed_background_values['F_bg_slope']
            elif self.analytical_estimates is not None:
                # Use values from analytical estimation
                fixed_F_bg_intercept = self.analytical_estimates['F_bg_intercept']
                fixed_F_bg_slope = self.analytical_estimates['F_bg_slope']
            else:
                # Use midpoint of bounds as best estimate
                fixed_F_bg_intercept = (bounds['F_bg_intercept'][0] + bounds['F_bg_intercept'][1]) / 2
                fixed_F_bg_slope = (bounds['F_bg_slope'][0] + bounds['F_bg_slope'][1]) / 2

            self.fixed_background = {
                'F_bg_intercept': fixed_F_bg_intercept,
                'F_bg_slope': fixed_F_bg_slope
            }

            if verbose:
                print(f"\n🔒 FIXED BACKGROUND (from exponential fits):")
                print(f"   F_bg_intercept = {fixed_F_bg_intercept:.6f}")
                print(f"   F_bg_slope = {fixed_F_bg_slope:.6f}")
                print(f"   Fitting only D0, k, P0 (3 parameters)")
        else:
            self.fixed_background = None

        if verbose:
            print(f"\n=== MAK2 Model Fitting ===")
            print(f"Target R²: ≥ {r2_threshold}")

        # Increase attempts for low-template wells (wide D0 bounds)
        # These are harder to fit due to poor exponential estimates
        D0_range_log = np.log10(bounds['D0'][1]) - np.log10(bounds['D0'][0])
        if D0_range_log > 5:  # Very wide bounds (>100,000x range)
            max_attempts = max(max_attempts, 10)  # At least 10 attempts
            if verbose:
                print(f"  Wide D0 bounds detected ({D0_range_log:.1f} orders of magnitude)")
                print(f"  Increasing attempts to {max_attempts} for better coverage")
        
        if verbose:
            print(f"Max attempts: {max_attempts}")
            print(f"\nParameter bounds:")
            print(f"  D0: [{bounds['D0'][0]:.2e}, {bounds['D0'][1]:.2e}]")
            print(f"  k: [{bounds['k'][0]:.4f}, {bounds['k'][1]:.4f}]")
            print(f"  P0: [{bounds['P0'][0]:.4f}, {bounds['P0'][1]:.4f}]")
            print(f"  F_bg_intercept: [{bounds['F_bg_intercept'][0]:.4f}, {bounds['F_bg_intercept'][1]:.4f}]")
            print(f"  F_bg_slope: [{bounds['F_bg_slope'][0]:.6f}, {bounds['F_bg_slope'][1]:.6f}]")
            print(f"\nData statistics:")
            print(f"  F_max: {np.max(fluorescence_fit):.4f}")
            print(f"  F_min: {np.min(fluorescence_fit):.4f}")
            print(f"  F_range: {np.max(fluorescence_fit) - np.min(fluorescence_fit):.4f}")

        # Preserve original bounds for LHS sampling
        # Bounds may get adjusted during attempts, but LHS should sample from original space
        original_bounds = {k: v for k, v in bounds.items()}

        # Generate LHS samples for D0, k, and P0 (for attempts 2+)
        # Attempt 1 uses analytical estimates, attempts 2+ use LHS
        # ADAPTIVE LHS: Generate more samples for better parameter space coverage
        # This helps escape local minima by exploring more starting points
        print("🎲 GENERATING ADAPTIVE LATIN HYPERCUBE SAMPLES")

        # Adaptive sampling: scale with max_attempts for better coverage
        # Base: 20 samples per attempt (much more than previous 1 sample per attempt)
        # This provides ~200 LHS points for 10 attempts vs ~9 previously
        n_lhs_samples = 20 * (max_attempts - 1)  # Attempt 1 uses analytical, rest use LHS

        if n_lhs_samples > 0:
            sampler = qmc.LatinHypercube(d=3, seed=_derive_seed(_SEED_OFFSET_LHS_3D))  # 3D: D0, k, P0
            lhs_samples = sampler.random(n=n_lhs_samples)
            print(f"✅ Generated {n_lhs_samples} LHS samples ({20} per attempt × {max_attempts-1} attempts)")
            print(f"   This provides {n_lhs_samples}x more coverage than previous approach")

            # Scale to ORIGINAL bounds in log space for D0, linear for k and P0
            # Column 0 = D0 (log space), Column 1 = k, Column 2 = P0
            log_D0_min = np.log10(original_bounds['D0'][0])
            log_D0_max = np.log10(original_bounds['D0'][1])
            D0_lhs = 10**(lhs_samples[:, 0] * (log_D0_max - log_D0_min) + log_D0_min)
            k_lhs = lhs_samples[:, 1] * (original_bounds['k'][1] - original_bounds['k'][0]) + original_bounds['k'][0]
            P0_lhs = lhs_samples[:, 2] * (original_bounds['P0'][1] - original_bounds['P0'][0]) + original_bounds['P0'][0]

            # Override attempt 2 to explore HIGH P0 + LOW k + UPPER D0 corner
            # This region gives maximum plateau: P0/(1+k*P0) is highest when P0 high, k low
            # D0 should be in upper range for better signal
            if n_lhs_samples >= 1:
                D0_lhs[0] = 10**(log_D0_min + 0.7 * (log_D0_max - log_D0_min))  # 70% into D0 range (log space)
                k_lhs[0] = original_bounds['k'][0] + 0.1 * (original_bounds['k'][1] - original_bounds['k'][0])  # 10% into k range (low k)
                P0_lhs[0] = original_bounds['P0'][0] + 0.85 * (original_bounds['P0'][1] - original_bounds['P0'][0])  # 85% into P0 range (high P0)
                print(f"📍 Biased attempt 2 toward high-P0 low-k upper-D0 corner for maximum plateau")

            # Override attempt 3 with heuristic D0 estimate if available.
            # This ensures the LHS pool always includes at least one sample
            # near the sigmoid-derived D0, even if analytical estimation failed.
            if (n_lhs_samples >= 2
                    and hasattr(self, 'analytical_estimates')
                    and self.analytical_estimates is not None):
                D0_lhs[1] = self.analytical_estimates['D0']
                k_lhs[1] = self.analytical_estimates['k']
                P0_lhs[1] = self.analytical_estimates['P0']
                print(f"📍 Biased attempt 3 toward analytical/heuristic estimate: "
                      f"D0={D0_lhs[1]:.2e}, k={k_lhs[1]:.4f}, P0={P0_lhs[1]:.2e}")

            # Evaluate all LHS samples to find best starting points
            # This is the key to escaping local minima: we test many points
            # and optimize from the most promising ones
            print(f"\n📊 EVALUATING {n_lhs_samples} LHS SAMPLES...")
            lhs_scores = []

            # Use fixed background values for LHS evaluation when available,
            # otherwise fall back to bounds midpoint
            if hasattr(self, 'fixed_background') and self.fixed_background is not None:
                _lhs_bg_int = self.fixed_background['F_bg_intercept']
                _lhs_bg_slope = self.fixed_background['F_bg_slope']
            else:
                _lhs_bg_int = (original_bounds['F_bg_intercept'][0] + original_bounds['F_bg_intercept'][1]) / 2
                _lhs_bg_slope = (original_bounds['F_bg_slope'][0] + original_bounds['F_bg_slope'][1]) / 2

            for i, (d0, k, p0) in enumerate(zip(D0_lhs, k_lhs, P0_lhs)):
                try:
                    # Quick evaluation: compute SSR for this parameter combination
                    # without full optimization (just initial guess quality)
                    test_params = {
                        'D0': d0,
                        'k': k,
                        'P0': p0,
                        'F_bg_intercept': _lhs_bg_int,
                        'F_bg_slope': _lhs_bg_slope
                    }

                    # Calculate SSR for this guess
                    # Use simulate_to_cycle (not simulate_cycles) so predictions
                    # align with actual cycle numbers, not 0-indexed offsets.
                    predicted_fit = self.model.simulate_to_cycle(
                        D0=test_params['D0'],
                        k=test_params['k'],
                        P0=test_params['P0'],
                        cycles=cycles_fit,
                        F_bg_intercept=test_params['F_bg_intercept'],
                        F_bg_slope=test_params['F_bg_slope']
                    )

                    ssr = np.sum((fluorescence_fit - predicted_fit) ** 2)

                    lhs_scores.append((ssr, i, d0, k, p0))
                except:
                    # If evaluation fails, assign worst score
                    lhs_scores.append((np.inf, i, d0, k, p0))

            # Sort by SSR (lower is better)
            lhs_scores.sort()

            # Use top samples for attempts 2+
            # Number of starting points = max_attempts - 1 (attempt 1 uses analytical)
            n_starts = min(max_attempts - 1, len(lhs_scores))
            best_lhs_indices = [lhs_scores[i][1] for i in range(n_starts)]

            # Extract best LHS samples for optimization attempts
            D0_lhs_best = np.array([D0_lhs[i] for i in best_lhs_indices])
            k_lhs_best = np.array([k_lhs[i] for i in best_lhs_indices])
            P0_lhs_best = np.array([P0_lhs[i] for i in best_lhs_indices])

            print(f"✅ Selected top {n_starts} LHS samples (from {n_lhs_samples} evaluated)")

            # Always print top LHS — critical for diagnosing convergence failures
            print(f"\nTop {min(5, n_starts)} LHS starting points (by SSR):")
            for i in range(min(5, n_starts)):
                ssr, idx, d0, k, p0 = lhs_scores[i]
                print(f"  Rank {i+1}: D0={d0:.2e}, k={k:.4f}, P0={p0:.4f}, SSR={ssr:.6f}")
            # Also show worst to verify spread
            if len(lhs_scores) > 5:
                worst_ssr = lhs_scores[-1][0]
                best_ssr = lhs_scores[0][0]
                print(f"  SSR spread: best={best_ssr:.2e}, worst={worst_ssr:.2e}, ratio={worst_ssr/max(best_ssr,1e-30):.1f}x")

            # Use the best ones for optimization
            D0_lhs = D0_lhs_best
            k_lhs = k_lhs_best
            P0_lhs = P0_lhs_best

        else:
            D0_lhs = None
            k_lhs = None
            P0_lhs = None

        # Adaptive multi-start optimization with bounds adjustment
        best_params = None
        best_r2 = -np.inf
        n_bounds_adjustments = 0  # Track number of adjustments
        max_bounds_adjustments = 3  # Allow up to 3 adjustments

        for attempt in range(1, max_attempts + 1):
            if time.perf_counter() >= self._fit_deadline:
                self._fit_timed_out = True
                break
            try:
                # For attempts 2+, pass LHS samples for D0, k, and P0
                lhs_D0 = D0_lhs[attempt - 2] if attempt > 1 and D0_lhs is not None else None
                lhs_k = k_lhs[attempt - 2] if attempt > 1 and k_lhs is not None else None
                lhs_P0 = P0_lhs[attempt - 2] if attempt > 1 and P0_lhs is not None else None

                # Use original bounds for LHS attempts to avoid clipping LHS samples
                # Use adjusted bounds only for attempt 1 (analytical estimates)
                bounds_to_use = original_bounds if (lhs_D0 is not None or lhs_k is not None or lhs_P0 is not None) else bounds

                params, r2 = self._fit_attempt(
                    cycles_fit,
                    fluorescence_fit,
                    bounds_to_use,
                    seed=_derive_seed(_SEED_OFFSET_TIER1_RETRY, attempt),
                    lhs_D0=lhs_D0,
                    lhs_k=lhs_k,
                    lhs_P0=lhs_P0
                )
                
                if verbose:
                    print(f"  Attempt {attempt}: R² = {r2:.6f}, SSR = {params['ssr']:.6f}")

                # Detect stuck optimization: check if fitted params are too close to initial guess
                k_change = abs(params['k'] - params['k_init']) / max(abs(params['k_init']), 0.01)
                P0_change = abs(params['P0'] - params['P0_init']) / max(abs(params['P0_init']), 0.01)

                # Always print for debugging
                print(f"    DEBUG: k change = {k_change*100:.1f}%, P0 change = {P0_change*100:.1f}%")

                # If both k and P0 barely moved (<5% relative change), optimizer is stuck
                if k_change < 0.05 and P0_change < 0.05 and attempt < max_attempts:
                    print(f"    ⚠️  Stuck optimization detected!")
                    print(f"       k: {params['k_init']:.4f} → {params['k']:.4f} ({k_change*100:.1f}% change)")
                    print(f"       P0: {params['P0_init']:.4f} → {params['P0']:.4f} ({P0_change*100:.1f}% change)")
                    print(f"       Triggering immediate retry with random initialization")
                    # Skip rest of checks, force continue to next attempt
                    continue

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
                        # k hitting lower bound — do NOT widen below the
                        # physiological minimum (0.05).  Lowering k further
                        # leads to degenerate fits (slow exponential instead
                        # of proper sigmoid).  Instead, just let the optimizer
                        # continue with the current bounds.
                        if verbose:
                            print(f"    ⚠ k at lower bound ({bounds['k'][0]:.6f}) — keeping bound (physiological minimum)")
                        n_bounds_adjustments += 1
                        continue
                
                # If R² is already very good (≥0.99), accept it even with minor issues
                # Continue trying remaining attempts to find the best fit
                if r2 >= 0.99:
                    if verbose:
                        print(f"    ✓ R² = {r2:.6f} is excellent, accepting fit despite minor warnings")
                    # Track as best but continue to try remaining attempts
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = params
                    # Continue to next attempt instead of breaking
                else:
                    # If SSR is high and k is small, likely stuck in local minimum
                    # BUT: For low-template wells (wide D0 bounds), k SHOULD be small!
                    # Don't increase k bounds if D0 bounds span >5 orders of magnitude
                    D0_range_log = np.log10(bounds['D0'][1]) - np.log10(bounds['D0'][0])
                    is_low_template = D0_range_log > 5  # >100,000x range

                    if ssr_too_high and params['k'] < bounds['k'][1] * 0.1 and n_bounds_adjustments < max_bounds_adjustments and not is_low_template:
                        if verbose:
                            print(f"    ⚠ High SSR + small k → Likely local minimum")
                            print(f"    → Increasing k lower bound to escape")
                        old_k_min = bounds['k'][0]
                        old_k_max = bounds['k'][1]
                        # Shift k bounds upward significantly, but cap at 1.5 (unrealistic beyond this)
                        new_k_max = min(old_k_max * 2, 1.5)
                        new_k_min = old_k_min * 10

                        # Prevent inverted bounds
                        if new_k_min >= new_k_max:
                            # If multiplication would invert, just use moderate increase
                            new_k_min = min(old_k_min * 2, 0.8)
                            if new_k_min >= new_k_max:
                                # Still inverted, give up on this adjustment
                                if verbose:
                                    print(f"    → Cannot increase k bounds further without inversion, skipping")
                                continue

                        bounds['k'] = (new_k_min, new_k_max)
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

        # --- End Tier 1 ---
        _tier1_elapsed = time.perf_counter() - _tier1_start
        self.tier_log.append({
            'tier': 'tier1_multistart',
            'fired': True,
            'time_seconds': _tier1_elapsed,
            'r2_before': None,
            'r2_after': best_r2,
            'improved': True,
        })

        # ── Tier 1.5: Residual-pattern-driven retry ──────────────────────────
        # When Tier 1 produces a fit that's "good but not great" (R² below the
        # threshold but above ~0.99), the residual pattern often points at a
        # specific failure mode — k stuck at the lower bound, plateau
        # overshoot, baseline elbow, etc. Each pattern is matched against
        # a tuned bounds adjustment; the optimizer then re-runs Tier 1 with
        # the corrected bounds.
        #
        # IMPORTANT — Phase 1 review candidate: many of the pattern blocks
        # below are tagged with comments referencing specific Rutledge wells
        # (X6.R4.2, X6.R2.1, X6.R5.4, X5.R1.4 …). The actual `if` conditions
        # test residual *shape*, not literal sample names — but the comments
        # honestly record that the heuristics were tuned against those wells.
        # During Phase 1 unification, this block should be reviewed: do the
        # patterns generalise to other instruments / chemistries, or are they
        # overfit to the qpcR R-package fixtures? See CLAUDE.md.
        _tier1_5_start = time.perf_counter()
        _tier1_5_fired = False

        # Tier 1.5 only runs when Tier 1 actually produced a parameter set
        # that didn't clear the R² threshold AND the per-fit deadline hasn't
        # expired. Disabled-tiers checking is for ablation testing in
        # benchmark_tiers.py.
        if (best_params is not None and best_r2 < r2_threshold
                and 'tier1.5' not in disabled_tiers
                and time.perf_counter() < self._fit_deadline):
            _tier1_5_fired = True
            # Predict fluorescence for all fitted cycles
            predicted_F = self.model.simulate_to_cycle(
                best_params['D0'], best_params['k'], best_params['P0'],
                cycles_fit, best_params['F_bg_intercept'], best_params['F_bg_slope']
            )

            # Divide data into regions for residual pattern analysis
            n_points = len(cycles_fit)
            n_baseline = max(3, int(n_points * 0.3))  # First 30% - baseline region
            n_plateau = max(3, int(n_points * 0.2))   # Last 20% - plateau region
            # Middle region is exponential/elbow

            baseline_data = fluorescence_fit[:n_baseline]
            baseline_pred = predicted_F[:n_baseline]
            baseline_residuals = baseline_data - baseline_pred

            elbow_data = fluorescence_fit[n_baseline:-n_plateau]
            elbow_pred = predicted_F[n_baseline:-n_plateau]
            elbow_residuals = elbow_data - elbow_pred

            plateau_data = fluorescence_fit[-n_plateau:]
            plateau_pred = predicted_F[-n_plateau:]
            plateau_residuals = plateau_data - plateau_pred

            # Calculate mean residuals in each region
            mean_baseline_residual = np.mean(baseline_residuals)
            mean_elbow_residual = np.mean(elbow_residuals) if len(elbow_residuals) > 0 else 0
            mean_plateau_residual = np.mean(plateau_residuals)

            # Analyze residual patterns to determine retry strategy
            # Pattern 1: Positive baseline residuals → increase background
            baseline_too_low = mean_baseline_residual > 0.01

            # Pattern 2: All positive residuals → systematic undershoot, likely background issue
            # Combined with baseline being low, this suggests global shift needed
            all_positive = (mean_baseline_residual > 0.005 and
                           mean_elbow_residual > 0.005 and
                           mean_plateau_residual > 0.005)

            # Pattern 3: Negative elbow, positive plateau → transition too late, decrease k & D0
            late_transition = mean_elbow_residual < -0.01 and mean_plateau_residual > 0.01

            # Pattern 4: Positive elbow, negative plateau → transition too early, increase k & D0
            early_transition = mean_elbow_residual > 0.01 and mean_plateau_residual < -0.01

            # Pattern 5: Positive plateau only → plateau saturation, decrease k
            final_residual = plateau_residuals[-1]
            final_2_mean = np.mean(plateau_residuals[-2:])
            plateau_saturation = (
                final_residual > 0.02 and
                final_2_mean > 0.01 and
                plateau_pred[-1] < plateau_data[-1] and
                not baseline_too_low and
                not all_positive  # Exclude if global background is the issue
            )

            # Pattern 6: Negative plateau residuals → model overshoots plateau
            # Model predicts too high at plateau → P0 too high or k too low
            # Fix: decrease P0, increase k, adjust background slope if needed
            plateau_overshoot = (
                mean_plateau_residual < -0.01 and
                plateau_pred[-1] > plateau_data[-1] and
                not plateau_saturation  # Mutually exclusive
            )

            # Pattern 7: Increasing residuals (plateau >> elbow >> baseline)
            # Suggests background slope is too low (flat when should be rising)
            # Check if plateau residuals significantly higher than baseline
            increasing_residuals = (
                mean_plateau_residual > mean_baseline_residual + 0.01 and
                mean_plateau_residual > 0.01 and
                not all_positive  # If all positive, use global undershoot instead
            )

            # Pattern 8: k stuck at lower bound (critical!)
            # When k hits lower bound, model is trying to achieve higher plateau by minimizing depletion
            # Real fix: increase k bounds AND increase background slope (make more positive)
            # Check if k is within 10% of lower bound
            k_at_lower_bound = abs(best_params['k'] - bounds['k'][0]) / bounds['k'][0] < 0.10

            # If k is stuck at lower bound, likely need both higher k and higher slope
            k_stuck_at_bound = (
                k_at_lower_bound and
                best_r2 < 0.995  # Only trigger if fit is poor
            )

            if verbose:
                print(f"\n  🔍 Residual Pattern Analysis:")
                print(f"    - Mean baseline residual: {mean_baseline_residual:+.4f}")
                print(f"    - Mean elbow residual: {mean_elbow_residual:+.4f}")
                print(f"    - Mean plateau residual: {mean_plateau_residual:+.4f}")
                print(f"    - Final residual: {final_residual:+.4f}")
                print(f"  Detected patterns:")
                print(f"    - Baseline too low: {baseline_too_low}")
                print(f"    - All positive (global undershoot): {all_positive}")
                print(f"    - Late transition: {late_transition}")
                print(f"    - Early transition: {early_transition}")
                print(f"    - Plateau saturation: {plateau_saturation}")
                print(f"    - Plateau overshoot: {plateau_overshoot}")
                print(f"    - Increasing residuals (slope issue): {increasing_residuals}")
                print(f"    - k stuck at lower bound: {k_stuck_at_bound}")

            # Apply retry strategy based on dominant pattern
            retry_needed = baseline_too_low or all_positive or late_transition or early_transition or plateau_saturation or plateau_overshoot or increasing_residuals

            if retry_needed:
                # Use ORIGINAL bounds, not modified bounds from SSR retry
                P0_old_lower, P0_old_upper = original_bounds['P0']
                D0_old_lower, D0_old_upper = original_bounds['D0']
                k_old_lower, k_old_upper = original_bounds['k']
                bg_int_old_lower, bg_int_old_upper = original_bounds['F_bg_intercept']
                bg_slope_old_lower, bg_slope_old_upper = original_bounds['F_bg_slope']

                # Initialize default slope bounds (will be overridden by specific patterns if needed)
                bg_slope_new_lower = bg_slope_old_lower
                bg_slope_new_upper = bg_slope_old_upper

                # Determine bounds adjustments based on residual pattern
                if k_stuck_at_bound:
                    # X6.R4.2 pattern: k at lower bound (0.05) → model trying to minimize depletion
                    # Real fix: INCREASE k bounds significantly + INCREASE background slope
                    # When k=0.05, model wants higher plateau but can't get it with low k
                    # User feedback: "increased slope and increased k"
                    if verbose:
                        print(f"\n  🔍 K STUCK AT LOWER BOUND DETECTED:")
                        print(f"    - k = {best_params['k']:.6f} (bound: {bounds['k'][0]:.6f})")
                        print(f"    - R² = {best_r2:.4f}")
                        print(f"    → INCREASING k bounds significantly (force k higher)")
                        print(f"    → INCREASING background slope (shift upward)")
                        print(f"    → Adjusting P0 bounds")

                    # Force k to explore MUCH HIGHER values
                    # If k is at 0.05, we want to push it to 0.2-0.8 range
                    k_new_lower = max(0.15, k_old_lower * 3.0)  # At least 3× higher
                    k_new_upper = max(k_old_upper, 1.0)  # Ensure upper bound is high enough
                    k_sample_range = (0.4, 0.8)  # Sample from higher range

                    # Increase background slope significantly (make more positive/upward)
                    # Estimate needed slope increase from signal range
                    signal_range = fluorescence_fit[-1] - fluorescence_fit[0]
                    slope_increase = signal_range / len(cycles_fit) * 0.02  # 2% of average rise per cycle

                    bg_slope_new_lower = bg_slope_old_lower + slope_increase * 0.5
                    bg_slope_new_upper = bg_slope_old_upper + slope_increase * 2.0

                    # Keep background intercept moderate
                    bg_int_new_lower = bg_int_old_lower
                    bg_int_new_upper = bg_int_old_upper

                    # Moderate D0/P0 adjustments
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper * 2.0
                    D0_sample_range = (0.3, 0.7)

                    # P0 adjustments depend on whether we need higher or lower plateau
                    # Start with moderate range
                    P0_new_lower = P0_old_lower * 0.8
                    P0_new_upper = P0_old_upper * 1.5
                    P0_sample_range = (0.3, 0.7)

                elif increasing_residuals:
                    # X6.R2.1 pattern: residuals increase from baseline → elbow → plateau
                    # Suggests background slope is too low (should be rising more)
                    # Fix: increase background slope bounds significantly
                    if verbose:
                        print(f"\n  🔍 INCREASING RESIDUALS DETECTED (SLOPE TOO LOW):")
                        print(f"    - Baseline residual: {mean_baseline_residual:+.4f}")
                        print(f"    - Plateau residual: {mean_plateau_residual:+.4f}")
                        print(f"    - Difference: {mean_plateau_residual - mean_baseline_residual:+.4f}")
                        print(f"    → Increasing background slope bounds")
                        print(f"    → Moderate parameter adjustments")

                    # Shift background slope UPWARD - this is the key fix
                    slope_shift = (mean_plateau_residual - mean_baseline_residual) / len(cycles_fit)
                    bg_slope_new_lower = max(bg_slope_old_lower, bg_slope_old_lower + slope_shift * 0.5)
                    bg_slope_new_upper = bg_slope_old_upper + slope_shift * 1.5

                    # Keep intercept bounds moderate
                    bg_int_new_lower = bg_int_old_lower
                    bg_int_new_upper = bg_int_old_upper

                    # Moderate D0/k/P0 adjustments
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper * 2.0
                    D0_sample_range = (0.3, 0.7)

                    k_new_lower = max(0.01, k_old_lower * 0.7)
                    k_new_upper = k_old_upper
                    k_sample_range = (0.1, 0.5)

                    P0_new_lower = P0_old_lower
                    P0_new_upper = P0_old_upper * 1.5

                elif all_positive:
                    # X6.R5.4 pattern: all residuals positive → global background too low
                    # Fix: significantly increase background, moderate adjustments to other params
                    if verbose:
                        print(f"\n  🔍 GLOBAL UNDERSHOOT DETECTED:")
                        print(f"    → Significantly increasing background bounds")
                        print(f"    → Moderate D0 sampling (20%-60%)")
                        print(f"    → Moderate k sampling (15%-55%)")

                    # Shift background up by mean of all residuals
                    global_shift = (mean_baseline_residual + mean_elbow_residual + mean_plateau_residual) / 3
                    bg_int_new_lower = bg_int_old_lower + global_shift
                    bg_int_new_upper = bg_int_old_upper + global_shift

                    # Keep D0/k bounds same, sample from moderate ranges
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper * 2.0
                    D0_sample_range = (0.2, 0.6)  # Moderate range

                    k_new_lower = k_old_lower
                    k_new_upper = k_old_upper
                    k_sample_range = (0.15, 0.55)  # Moderate range

                    # Moderate P0 increase
                    P0_new_lower = P0_old_lower
                    P0_new_upper = P0_old_upper * 1.5

                elif baseline_too_low and late_transition:
                    # X6.R5.4 pattern: positive baseline + negative elbow + positive plateau
                    # Fix: increase background, decrease D0 (shift lower), increase k (shift higher)
                    if verbose:
                        print(f"\n  🔍 BASELINE + LATE TRANSITION DETECTED:")
                        print(f"    → Increasing background bounds")
                        print(f"    → Decreasing D0 (sample from lower range)")
                        print(f"    → Increasing k (sample from higher range)")

                    # Increase background intercept bounds
                    bg_int_shift = mean_baseline_residual  # Shift up by average baseline error
                    bg_int_new_lower = bg_int_old_lower + bg_int_shift
                    bg_int_new_upper = bg_int_old_upper + bg_int_shift

                    # Keep D0 bounds same but will sample from LOWER portion (10%-40%)
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper
                    D0_sample_range = (0.1, 0.4)  # Sample from lower 10%-40%

                    # Keep k bounds same but will sample from HIGHER portion (30%-70%)
                    k_new_lower = k_old_lower
                    k_new_upper = k_old_upper
                    k_sample_range = (0.3, 0.7)  # Sample from higher 30%-70%

                    # Moderate P0 increase
                    P0_new_lower = P0_old_lower
                    P0_new_upper = P0_old_upper * 1.5

                elif plateau_saturation:
                    # F6.2 pattern: positive plateau only (no baseline/elbow issues)
                    # Fix: decrease k (sharper transition), moderate D0, increase P0
                    if verbose:
                        print(f"\n  🔍 PLATEAU SATURATION DETECTED:")
                        print(f"    - Model plateau: {plateau_pred[-1]:.4f}")
                        print(f"    - Data plateau: {plateau_data[-1]:.4f}")
                        print(f"    → Decreasing k (sample from lower range)")
                        print(f"    → Moderate D0 (sample from medium range)")
                        print(f"    → Increasing P0 bounds")

                    # Keep background bounds unchanged
                    bg_int_new_lower = bg_int_old_lower
                    bg_int_new_upper = bg_int_old_upper

                    # Sample D0 from medium range (30%-80%)
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper * 3.0
                    D0_sample_range = (0.3, 0.8)

                    # Sample k from very low range (2%-45%)
                    k_new_lower = max(0.01, k_old_lower * 0.5)
                    k_new_upper = k_old_upper
                    k_sample_range = (0.02, 0.45)

                    # Increase P0 upper bound
                    P0_new_lower = P0_old_lower
                    P0_new_upper = P0_old_upper * 2.0

                elif plateau_overshoot:
                    # X5.R1.4 pattern: negative plateau residuals → model overshoots
                    # Model predicts too high at plateau
                    # Fix: DECREASE P0, INCREASE k (more depletion, lower plateau)
                    # Also consider increasing background slope if baseline trends up
                    if verbose:
                        print(f"\n  🔍 PLATEAU OVERSHOOT DETECTED:")
                        print(f"    - Model plateau: {plateau_pred[-1]:.4f}")
                        print(f"    - Data plateau: {plateau_data[-1]:.4f}")
                        print(f"    - Model OVERSHOOTS (predicts too high)")
                        print(f"    → Decreasing P0 (sample from lower range)")
                        print(f"    → Increasing k (sample from higher range)")

                    # Keep background bounds, but could shift slope upward if needed
                    bg_int_new_lower = bg_int_old_lower
                    bg_int_new_upper = bg_int_old_upper

                    # Sample D0 from moderate range
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper * 2.0
                    D0_sample_range = (0.3, 0.7)

                    # Sample k from HIGHER range (40%-80%) for more depletion
                    k_new_lower = k_old_lower
                    k_new_upper = k_old_upper * 1.2  # Allow slightly higher k
                    k_sample_range = (0.4, 0.8)

                    # Sample P0 from LOWER range (10%-50%) to reduce plateau
                    P0_new_lower = P0_old_lower * 0.5
                    P0_new_upper = P0_old_upper * 0.8
                    P0_sample_range = (0.1, 0.5)  # Will override default later

                else:
                    # Default: use plateau saturation strategy
                    if verbose:
                        print(f"\n  🔍 POOR FIT DETECTED, using default retry strategy")
                    bg_int_new_lower = bg_int_old_lower
                    bg_int_new_upper = bg_int_old_upper
                    D0_new_lower = D0_old_lower
                    D0_new_upper = D0_old_upper * 3.0
                    D0_sample_range = (0.3, 0.8)
                    k_new_lower = max(0.01, k_old_lower * 0.5)
                    k_new_upper = k_old_upper
                    k_sample_range = (0.02, 0.45)
                    P0_new_lower = P0_old_lower
                    P0_new_upper = P0_old_upper * 2.0

                if verbose:
                    print(f"    - Old k bounds: [{k_old_lower:.4f}, {k_old_upper:.4f}]")
                    print(f"    - New k bounds: [{k_new_lower:.4f}, {k_new_upper:.4f}]")
                    print(f"    - Old P0 bounds: [{P0_old_lower:.4f}, {P0_old_upper:.4f}]")
                    print(f"    - New P0 bounds: [{P0_new_lower:.4f}, {P0_new_upper:.4f}]")
                    print(f"    - Old D0 bounds: [{D0_old_lower:.2e}, {D0_old_upper:.2e}]")
                    print(f"    - New D0 bounds: [{D0_new_lower:.2e}, {D0_new_upper:.2e}]")
                    print(f"    - Old BG bounds: [{bg_int_old_lower:.4f}, {bg_int_old_upper:.4f}]")
                    print(f"    - New BG bounds: [{bg_int_new_lower:.4f}, {bg_int_new_upper:.4f}]")

                # Try up to 3 attempts with pattern-specific sampling
                pattern_retry_best_ssr = np.inf
                pattern_retry_best_params = None
                pattern_retry_best_r2 = -np.inf

                for retry_i in range(1, 4):
                    try:
                        # Sample P0 - check if custom range is defined (for plateau_overshoot)
                        if 'P0_sample_range' in locals():
                            p0_min_pct, p0_max_pct = P0_sample_range
                            p0_pct = p0_min_pct + (p0_max_pct - p0_min_pct) * (retry_i - 1) / 2
                            P0_sample = P0_new_lower + p0_pct * (P0_new_upper - P0_new_lower)
                        else:
                            # Default: sample from UPPER portion (70%-100%) for most patterns
                            P0_sample = P0_new_lower + (0.7 + 0.3 * (retry_i - 1) / 2) * (P0_new_upper - P0_new_lower)

                        # Sample D0 using pattern-specific range
                        log_D0_lower = np.log10(D0_new_lower)
                        log_D0_upper = np.log10(D0_new_upper)
                        d0_min_pct, d0_max_pct = D0_sample_range
                        d0_pct = d0_min_pct + (d0_max_pct - d0_min_pct) * (retry_i - 1) / 2
                        D0_sample = 10**(log_D0_lower + d0_pct * (log_D0_upper - log_D0_lower))

                        # Sample k using pattern-specific range
                        k_min_pct, k_max_pct = k_sample_range
                        k_pct = k_min_pct + (k_max_pct - k_min_pct) * (retry_i - 1) / 2
                        k_sample = k_new_lower + k_pct * (k_new_upper - k_new_lower)

                        if verbose:
                            print(f"    Pattern retry {retry_i}: D0_init = {D0_sample:.2e}, k_init = {k_sample:.4f}, P0_init = {P0_sample:.4f}")

                        # Use uniform weighting for all retries (no plateau emphasis)
                        retry_params, retry_r2 = self._fit_attempt(
                            cycles_fit,
                            fluorescence_fit,
                            {**bounds,
                             'k': (k_new_lower, k_new_upper),
                             'D0': (D0_new_lower, D0_new_upper),
                             'P0': (P0_new_lower, P0_new_upper),
                             'F_bg_intercept': (bg_int_new_lower, bg_int_new_upper),
                             'F_bg_slope': (bg_slope_new_lower, bg_slope_new_upper)},
                            seed=_derive_seed(_SEED_OFFSET_PATTERN, retry_i),
                            lhs_D0=D0_sample,
                            lhs_k=k_sample,
                            lhs_P0=P0_sample,
                        )

                        if verbose:
                            print(f"      → R² = {retry_r2:.6f}, SSR = {retry_params['ssr']:.6f}, k_final = {retry_params['k']:.6f}, P0_final = {retry_params['P0']:.4f}")

                        if retry_r2 > pattern_retry_best_r2:
                            pattern_retry_best_r2 = retry_r2
                            pattern_retry_best_params = retry_params
                            pattern_retry_best_ssr = retry_params['ssr']

                        # Don't stop early - try all retries to find best parameters

                    except Exception as e:
                        if verbose:
                            print(f"    Pattern retry {retry_i}: Failed - {type(e).__name__}: {str(e)}")
                        continue

                # Use retry result if better
                if pattern_retry_best_params and pattern_retry_best_ssr < best_params['ssr']:
                    if verbose:
                        print(f"\n  ✅ Pattern-based retry improved fit:")
                        print(f"    Original: R² = {best_r2:.6f}, SSR = {best_params['ssr']:.6f}")
                        print(f"    Best retry: R² = {pattern_retry_best_r2:.6f}, SSR = {pattern_retry_best_ssr:.6f}")
                        print(f"    Best k = {pattern_retry_best_params['k']:.6f}, P0 = {pattern_retry_best_params['P0']:.4f}")
                    best_params = pattern_retry_best_params
                    best_r2 = pattern_retry_best_r2
                elif verbose:
                    if pattern_retry_best_params:
                        print(f"\n  ↩️  Pattern-based retry did not improve fit:")
                        print(f"    Original: SSR = {best_params['ssr']:.6f}")
                        print(f"    Best retry: SSR = {pattern_retry_best_ssr:.6f}")
                    else:
                        print(f"\n  ❌ All pattern-based retries failed")

                # Propagate refined bounds to downstream tiers only when
                # the fit still needs improvement.  If Tier 1.5 already
                # brought R² above threshold the refined (narrower) bounds
                # can actually hurt downstream tiers (e.g. Tier 4 refit)
                # by restricting their search space unnecessarily.
                if best_r2 < r2_threshold:
                    bounds['k'] = (k_new_lower, k_new_upper)
                    bounds['D0'] = (D0_new_lower, D0_new_upper)
                    bounds['P0'] = (P0_new_lower, P0_new_upper)
                    bounds['F_bg_intercept'] = (bg_int_new_lower, bg_int_new_upper)
                    bounds['F_bg_slope'] = (bg_slope_new_lower, bg_slope_new_upper)

                    if verbose:
                        print(f"\n  📤 Propagating refined bounds to downstream tiers (R² {best_r2:.4f} < {r2_threshold})")
                elif verbose:
                    print(f"\n  ⏭️  Skipping bounds propagation (R² {best_r2:.4f} ≥ {r2_threshold})")

        # --- End Tier 1.5 ---
        _tier1_5_elapsed = time.perf_counter() - _tier1_5_start
        _tier1_5_r2_after = best_r2
        self.tier_log.append({
            'tier': 'tier1.5_residual_patterns',
            'fired': _tier1_5_fired,
            'time_seconds': _tier1_5_elapsed,
            'r2_before': self.tier_log[-1]['r2_after'] if self.tier_log else None,
            'r2_after': _tier1_5_r2_after,
            'improved': _tier1_5_fired and _tier1_5_r2_after > (self.tier_log[-1]['r2_after'] if self.tier_log else 0),
        })

        # ── Tier 2: LHS-seeded multi-start retry ─────────────────────────────
        # Tier 1.5's pattern-driven retry handled "wrong basin" failures from
        # the analytical starting point. Tier 2 handles the orthogonal
        # failure mode: the optimizer landed in the right region but a poor
        # basin within it. The mitigation is brute-force initial-guess
        # diversity — Latin Hypercube Sampling spreads N starting points
        # uniformly across the parameter box, each is fed to TRF, and the
        # best result wins.
        #
        # Activation criterion: SSR > 1% of the squared signal range. This
        # catches fits that are technically in the right basin (R² maybe
        # 0.998) but have a long-tail residual structure suggesting a
        # nearby better minimum exists.
        _tier2_start = time.perf_counter()
        _tier2_fired = False
        _tier2_r2_before = best_r2

        F_max = np.max(fluorescence_fit)
        F_min = np.min(fluorescence_fit)
        F_range = F_max - F_min
        ssr = best_params['ssr']
        ssr_threshold = 0.01 * (F_range ** 2)

        if (ssr > ssr_threshold and 'retry_attempted' not in locals()
                and 'tier2' not in disabled_tiers
                and time.perf_counter() < self._fit_deadline):
            _tier2_fired = True
            # Check if this is a low-template well (wide D0 bounds)
            D0_range_log = np.log10(bounds['D0'][1]) - np.log10(bounds['D0'][0])
            is_low_template = D0_range_log > 5  # >100,000x range (very conservative)
            
            if verbose:
                print(f"\n  🔄 Attempting retry with adjusted initial conditions...")
                if is_low_template:
                    print(f"    - Low-template well detected (D0 range: {D0_range_log:.1f} orders)")
                    print(f"    - Keeping k bounds narrow, adjusting D0/P0 only")
                else:
                    print(f"    - Increasing k bounds (shift upward by 5×)")
                    print(f"    - Decreasing P0 bounds (shift downward by 0.5×)")
            
            # Adjust bounds to escape local minimum
            if is_low_template:
                # For low-template wells: k should be small (0.02-0.2)
                # Don't increase k bounds, just reset to reasonable range
                bounds['k'] = (0.01, 0.5)  # Keep k modest but physiologically realistic
                
                # For P0: check if this is high-plateau sample
                # If F_max > 5, keep P0 upper bound high enough
                if F_max > 5.0:
                    # High plateau - keep P0 range wide
                    P0_lower_retry = max(0.05, bounds['P0'][0] * 0.5)
                    P0_upper_retry = max(bounds['P0'][1], F_max * 2.0)
                    bounds['P0'] = (P0_lower_retry, P0_upper_retry)
                else:
                    # Low plateau - allow smaller P0 (primer in excess)
                    bounds['P0'] = (0.05, 10.0)
                
                # Sample even closer to perfect doubling D0
                old_D0_bounds = bounds['D0']
                log_D0_lower = np.log10(old_D0_bounds[0])
                D0_range = np.log10(old_D0_bounds[1]) - log_D0_lower
                # Sample from lower 10% only (even tighter than before)
                bounds['D0'] = (old_D0_bounds[0], 10**(log_D0_lower + D0_range * 0.1))
                
                if verbose:
                    print(f"    - k bounds: [0.05, 0.5]")
                    print(f"    - P0 bounds: [0.05, 10.0] (allow small P0)")
                    print(f"    - D0 bounds (lower 10%): [{bounds['D0'][0]:.2e}, {bounds['D0'][1]:.2e}]")
            else:
                # Original logic for normal wells
                # Increase k (local minimum often has k too small), but cap at 1.5
                old_k_bounds = bounds['k']
                new_k_lower = old_k_bounds[0] * 5
                new_k_max = min(old_k_bounds[1] * 3, 1.5)

                # If lower would exceed upper, just shift the entire range up
                if new_k_lower >= new_k_max:
                    new_k_lower = min(old_k_bounds[0] * 2, 0.8)  # More modest increase
                    new_k_max = 1.5

                bounds['k'] = (new_k_lower, new_k_max)
                
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
                        seed=_derive_seed(_SEED_OFFSET_TIER2_SECOND, attempt)
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
        
        # --- End Tier 2 ---
        _tier2_elapsed = time.perf_counter() - _tier2_start
        self.tier_log.append({
            'tier': 'tier2_ssr_retry',
            'fired': _tier2_fired,
            'time_seconds': _tier2_elapsed,
            'r2_before': _tier2_r2_before,
            'r2_after': best_r2,
            'improved': _tier2_fired and best_r2 > _tier2_r2_before,
        })

        # Store results
        self.optimal_params = best_params
        self.cycles_fit = cycles_fit
        self.fluorescence_fit = fluorescence_fit

        # Calculate metrics
        self.metrics = self.calculate_fit_metrics()

        # Track whether we used fixed background
        used_fixed_bg = fix_background and hasattr(self, 'fixed_background') and self.fixed_background is not None
        best_params['used_fixed_background'] = used_fixed_bg
        best_params['fallback_attempted'] = False
        best_params['fallback_succeeded'] = False

        # --- Tier 2.5: Adaptive Background Fallback ---
        _tier2_5_start = time.perf_counter()
        _tier2_5_fired = False
        _tier2_5_r2_before = self.metrics['r_squared']

        # ADAPTIVE BACKGROUND STRATEGY
        # If we used fixed background but R² < 0.999, retry with full 5-parameter fit
        if (used_fixed_bg and self.metrics['r_squared'] < 0.999
                and 'tier2.5' not in disabled_tiers
                and time.perf_counter() < self._fit_deadline):
            _tier2_5_fired = True
            best_params['fallback_attempted'] = True
            # Always print fallback messages (even with verbose=False)
            print(f"\n🔄 ADAPTIVE FALLBACK: Fixed background R²={self.metrics['r_squared']:.4f} < 0.999")
            print(f"   Retrying with full 5-parameter fit...")

            try:
                # Save fixed background results
                fixed_bg_params = dict(best_params)  # Use dict() instead of .copy()
                fixed_bg_r2 = self.metrics['r_squared']

                # Clear fixed background and re-run optimization on SAME truncated data
                self.fixed_background = None

                # WIDEN BOUNDS for fallback to escape local minima
                # Increase D0 upper bound by 10x
                # Decrease k lower bound by 10x
                # Increase P0 upper bound by 2x
                print(f"   Widening bounds for better exploration...")
                fallback_bounds = dict(bounds)
                fallback_bounds['D0'] = (bounds['D0'][0], bounds['D0'][1] * 10)
                fallback_bounds['k'] = (max(0.01, bounds['k'][0] / 10), bounds['k'][1])
                fallback_bounds['P0'] = (bounds['P0'][0], bounds['P0'][1] * 2)
                print(f"   D0: [{fallback_bounds['D0'][0]:.2e}, {fallback_bounds['D0'][1]:.2e}]")
                print(f"   k: [{fallback_bounds['k'][0]:.4f}, {fallback_bounds['k'][1]:.4f}]")
                print(f"   P0: [{fallback_bounds['P0'][0]:.4f}, {fallback_bounds['P0'][1]:.4f}]")

                # Generate LHS samples for FULL 5D parameter space with WIDENED bounds
                # Use more samples for fallback since we need thorough exploration
                n_lhs_fallback = 40  # Reduced from 100 — most good fits found in top 10
                print(f"   Generating {n_lhs_fallback} LHS samples for 5D parameter space...")

                # qmc is already imported at the top
                sampler = qmc.LatinHypercube(d=5, seed=_derive_seed(_SEED_OFFSET_LHS_5D))  # 5D: D0, k, P0, F_bg_int, F_bg_slope
                lhs_samples_5d = sampler.random(n=n_lhs_fallback)

                # Scale to WIDENED bounds in log space for D0, linear for others
                log_D0_min = np.log10(fallback_bounds['D0'][0])
                log_D0_max = np.log10(fallback_bounds['D0'][1])
                D0_lhs_5d = 10**(lhs_samples_5d[:, 0] * (log_D0_max - log_D0_min) + log_D0_min)
                k_lhs_5d = lhs_samples_5d[:, 1] * (fallback_bounds['k'][1] - fallback_bounds['k'][0]) + fallback_bounds['k'][0]
                P0_lhs_5d = lhs_samples_5d[:, 2] * (fallback_bounds['P0'][1] - fallback_bounds['P0'][0]) + fallback_bounds['P0'][0]
                F_bg_int_lhs_5d = lhs_samples_5d[:, 3] * (fallback_bounds['F_bg_intercept'][1] - fallback_bounds['F_bg_intercept'][0]) + fallback_bounds['F_bg_intercept'][0]
                F_bg_slope_lhs_5d = lhs_samples_5d[:, 4] * (fallback_bounds['F_bg_slope'][1] - fallback_bounds['F_bg_slope'][0]) + fallback_bounds['F_bg_slope'][0]

                # Evaluate all LHS samples
                print(f"   Evaluating {n_lhs_fallback} LHS samples...")
                lhs_scores_5d = []
                for i in range(n_lhs_fallback):
                    try:
                        predicted = self.model.simulate_to_cycle(
                            D0=D0_lhs_5d[i],
                            k=k_lhs_5d[i],
                            P0=P0_lhs_5d[i],
                            cycles=cycles_fit,
                            F_bg_intercept=F_bg_int_lhs_5d[i],
                            F_bg_slope=F_bg_slope_lhs_5d[i]
                        )
                        ssr = np.sum((fluorescence_fit - predicted)**2)
                        lhs_scores_5d.append((ssr, i))
                    except:
                        lhs_scores_5d.append((np.inf, i))

                # Sort by SSR and take top candidates
                lhs_scores_5d.sort()
                n_starts_fallback = min(max_attempts, len(lhs_scores_5d))
                print(f"   Optimizing from top {n_starts_fallback} starting points...")

                # Re-run the optimization loop with best LHS starting points
                best_params_full = None
                best_r2_full = -np.inf

                for attempt in range(n_starts_fallback):
                    try:
                        idx = lhs_scores_5d[attempt][1]

                        # Use LHS sample as initial guess with WIDENED bounds
                        params_full, r2_full = self._fit_attempt(
                            cycles_fit,
                            fluorescence_fit,
                            fallback_bounds,  # Use widened bounds!
                            seed=_derive_seed(_SEED_OFFSET_TIER25_ATTEMPT, attempt),
                            lhs_D0=D0_lhs_5d[idx],
                            lhs_k=k_lhs_5d[idx],
                            lhs_P0=P0_lhs_5d[idx]
                        )

                        if r2_full > best_r2_full:
                            best_params_full = params_full
                            best_r2_full = r2_full
                            print(f"   Attempt {attempt+1}: R² = {r2_full:.4f}")

                        if r2_full >= r2_threshold:
                            print(f"   ✅ Threshold met!")
                            break

                    except Exception as e:
                        print(f"   Attempt {attempt+1} failed: {str(e)[:80]}")
                        continue

                # Compare fixed vs full fit
                if best_params_full is not None and best_r2_full > fixed_bg_r2:
                    print(f"   ✅ Full fit improved: R² {fixed_bg_r2:.4f} → {best_r2_full:.4f}")
                    best_params = best_params_full
                    best_params['fallback_succeeded'] = True
                    # Update stored results
                    self.optimal_params = best_params
                    self.metrics = self.calculate_fit_metrics()
                else:
                    print(f"   ↩️  Fixed background was better: keeping R² = {fixed_bg_r2:.4f}")
                    # Restore fixed background results (already stored)

            except Exception as e:
                print(f"   ⚠️  Fallback failed with error: {str(e)[:80]}")
                print(f"   Keeping fixed background result (R² = {self.metrics['r_squared']:.4f})")

        # --- End Tier 2.5 ---
        _tier2_5_elapsed = time.perf_counter() - _tier2_5_start
        _tier2_5_r2_after = self.metrics['r_squared']
        self.tier_log.append({
            'tier': 'tier2.5_adaptive_fallback',
            'fired': _tier2_5_fired,
            'time_seconds': _tier2_5_elapsed,
            'r2_before': _tier2_5_r2_before,
            'r2_after': _tier2_5_r2_after,
            'improved': _tier2_5_fired and _tier2_5_r2_after > _tier2_5_r2_before,
        })

        # ── Tier 3: Differential Evolution (global stochastic search) ────────
        # When LHS-seeded TRF still hasn't cleared 0.999, the local-minimum
        # problem is severe enough to need a method that ignores starting
        # points entirely. scipy's differential_evolution is a global
        # population-based stochastic optimiser — it maintains a population
        # of candidate parameter vectors and evolves them via mutation +
        # crossover. Slow (~1-3 s per well, 5-10× the cost of TRF) but
        # finds basins TRF simply can't reach from any single starting
        # point.
        #
        # Activation gate: R² in [0.95, 0.999). Below 0.95 the curve is
        # almost certainly non-amplifying or pathological — DE wastes
        # seconds and still fails. Above 0.999 the fit is already done.
        # The 0.95 floor was chosen empirically; setting it lower made
        # batch mode unacceptably slow without meaningfully recovering
        # additional wells.
        _tier3_start = time.perf_counter()
        _tier3_fired = False
        _tier3_r2_before = self.metrics['r_squared']

        if (0.95 <= self.metrics['r_squared'] < 0.999
                and 'tier3' not in disabled_tiers
                and time.perf_counter() < self._fit_deadline):
            _tier3_fired = True
            print(f"\n🌍 TIER 3: DIFFERENTIAL EVOLUTION")
            print(f"   Current R²={self.metrics['r_squared']:.4f} < 0.999")
            print(f"   Attempting global optimization with Differential Evolution...")

            try:
                # Use widened bounds for DE (or even wider)
                de_bounds_widened = dict(bounds)
                de_bounds_widened['D0'] = (bounds['D0'][0], bounds['D0'][1] * 20)  # 20x for DE
                de_bounds_widened['k'] = (max(0.01, bounds['k'][0] / 20), bounds['k'][1])     # 20x wider but floor at 0.01
                de_bounds_widened['P0'] = (bounds['P0'][0], bounds['P0'][1] * 3)   # 3x wider

                de_params, de_r2 = self._fit_with_differential_evolution(
                    cycles_fit,
                    fluorescence_fit,
                    de_bounds_widened,
                    verbose=False
                )

                if de_r2 > self.metrics['r_squared']:
                    print(f"   ✅ DE improved: R² {self.metrics['r_squared']:.4f} → {de_r2:.4f}")
                    self.optimal_params = de_params
                    self.optimal_params['de_used'] = True
                    self.metrics = self.calculate_fit_metrics()
                else:
                    print(f"   ↩️  DE did not improve: R² = {de_r2:.4f}")

            except Exception as e:
                print(f"   ⚠️  DE failed: {str(e)[:80]}")

        # --- End Tier 3 ---
        _tier3_elapsed = time.perf_counter() - _tier3_start
        _tier3_r2_after = self.metrics['r_squared']
        self.tier_log.append({
            'tier': 'tier3_differential_evolution',
            'fired': _tier3_fired,
            'time_seconds': _tier3_elapsed,
            'r2_before': _tier3_r2_before,
            'r2_after': _tier3_r2_after,
            'improved': _tier3_fired and _tier3_r2_after > _tier3_r2_before,
        })

        # ── Tier 4: Plateau-overshoot post-fit correction ────────────────────
        # All previous tiers optimise R² unconditionally. But a fit can have
        # excellent R² and still be wrong about P0 — specifically, if the
        # fitted plateau height exceeds the observed plateau, the model is
        # claiming "more primer was available than the data shows," which
        # propagates into a too-low D0 estimate via the k * P0 coupling.
        # Tier 4 detects this overshoot and re-fits with a P0 upper bound
        # capped at the current fitted P0, allowing the optimiser to find
        # a slightly worse-R² but more physically correct fit.
        #
        # The refit recursively calls fit() with a fresh optimizer instance.
        # _skip_overshoot_refit guards against infinite recursion: the
        # nested fit sees the flag set and returns its result directly
        # without entering Tier 4 again.
        _tier4_start = time.perf_counter()
        _tier4_fired = False
        _tier4_r2_before = self.metrics['r_squared']

        if self._skip_overshoot_refit:
            # Log Tier 4 as skipped (recursion guard)
            self.tier_log.append({
                'tier': 'tier4_overshoot_refit',
                'fired': False,
                'time_seconds': 0.0,
                'r2_before': _tier4_r2_before,
                'r2_after': _tier4_r2_before,
                'improved': False,
            })
            return self.optimal_params
        if 'tier4' in disabled_tiers:
            self.tier_log.append({
                'tier': 'tier4_overshoot_refit',
                'fired': False,
                'time_seconds': 0.0,
                'r2_before': _tier4_r2_before,
                'r2_after': _tier4_r2_before,
                'improved': False,
            })
            return self.optimal_params
        if time.perf_counter() >= self._fit_deadline:
            self._fit_timed_out = True
            return self.optimal_params
        try:
            overshoots, overshoot_ratio = self.check_plateau_overshoot(
                overshoot_threshold=0.0,
                verbose=verbose
            )

            if overshoots:
                _tier4_fired = True
                if verbose:
                    print(f"\n⚠️  Model overshoots plateau by {overshoot_ratio:.1%}")
                    print(f"   Refitting with constrained P₀ using full optimization pipeline...")

                # Save original results
                original_params = dict(self.optimal_params)
                original_metrics = dict(self.metrics)

                # Constrain P0: upper bound at current value, initial guess at 90%
                current_P0 = self.optimal_params['P0']
                refit_bounds = dict(bounds)
                refit_bounds['P0'] = (refit_bounds['P0'][0], current_P0)

                if verbose:
                    print(f"   Current P₀: {current_P0:.4e}")
                    print(f"   Refit P₀ bounds: [{refit_bounds['P0'][0]:.4e}, {refit_bounds['P0'][1]:.4e}]")

                try:
                    # Create a fresh optimizer and run full fit pipeline
                    # Skip Tier 4 on refit to prevent recursion
                    refit_model = MAK2Model()
                    refit_optimizer = MAK2Optimizer(refit_model)
                    refit_optimizer._skip_overshoot_refit = True
                    refit_optimizer.fit(
                        cycles, fluorescence,
                        cycles_after_max=cycles_after_max,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=refit_bounds,
                        max_attempts=max_attempts,
                        r2_threshold=r2_threshold,
                        verbose=False,
                        fix_background=fix_background
                    )

                    # Check if refit reduced overshoot
                    new_overshoots, new_overshoot_ratio = refit_optimizer.check_plateau_overshoot(
                        overshoot_threshold=0.0,
                        verbose=False
                    )
                    refit_r2 = refit_optimizer.metrics['r_squared']

                    # Accept refit if it reduced overshoot and R² is still good
                    if new_overshoot_ratio < overshoot_ratio and refit_r2 >= 0.995:
                        if verbose:
                            print(f"   ✅ Refit successful: overshoot {overshoot_ratio:.1%} → {new_overshoot_ratio:.1%}")
                            print(f"      R² = {refit_r2:.4f}, P₀ = {refit_optimizer.optimal_params['P0']:.4e}")
                        # Copy refit results into this optimizer
                        self.optimal_params = refit_optimizer.optimal_params
                        self.cycles_fit = refit_optimizer.cycles_fit
                        self.fluorescence_fit = refit_optimizer.fluorescence_fit
                        self.metrics = refit_optimizer.metrics
                        self.optimal_params['overshoot_refit'] = True
                    else:
                        if verbose:
                            print(f"   ↩️  Refit did not improve: overshoot {overshoot_ratio:.1%} → {new_overshoot_ratio:.1%}, R² = {refit_r2:.4f}")
                        # Keep original
                        self.optimal_params = original_params
                        self.metrics = original_metrics

                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Refit failed: {str(e)[:80]}")
                    self.optimal_params = original_params
                    self.metrics = original_metrics

        except Exception as e:
            if verbose:
                print(f"   ⚠️  Overshoot check failed: {str(e)[:80]}")

        # --- End Tier 4 ---
        _tier4_elapsed = time.perf_counter() - _tier4_start
        _tier4_r2_after = self.metrics['r_squared']
        self.tier_log.append({
            'tier': 'tier4_overshoot_refit',
            'fired': _tier4_fired,
            'time_seconds': _tier4_elapsed,
            'r2_before': _tier4_r2_before,
            'r2_after': _tier4_r2_after,
            'improved': _tier4_fired and _tier4_r2_after > _tier4_r2_before,
        })

        return self.optimal_params

    def _fit_with_differential_evolution(
        self,
        cycles: np.ndarray,
        fluorescence: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
        verbose: bool = False
    ) -> Tuple[Dict[str, float], float]:
        """Run scipy Differential Evolution on the MAK2 loss surface.

        DE is a global stochastic optimiser — it maintains a population
        of candidate parameter vectors and evolves them through
        mutation + crossover, ignoring gradient information entirely.
        Unlike TRF (which needs a starting guess and may converge to
        a nearby local minimum), DE explores the whole parameter box
        and reliably finds the global minimum, at the cost of being
        ~5-10× slower per call. Used only by Tier 3, when LHS-seeded
        TRF has failed to clear the R² threshold.

        Implementation notes worth knowing:

        - **D0 is optimised in log10 space** for the same reason the
          rest of the engine works in log space: D0 spans 6+ orders of
          magnitude across a dilution series and DE's mutation step
          works in fixed-magnitude jumps that are hopeless on a
          linear scale.
        - **strategy='best2bin'** + **popsize=30**: empirically more
          robust on noisy qPCR loss surfaces than the scipy default
          ('best1bin', popsize=15). The 5-parameter problem with
          this popsize gives 150 candidates per generation.
        - **polish=True** runs L-BFGS-B on DE's best solution at
          convergence, so the final answer matches gradient-method
          precision (rather than DE's coarser stochastic resolution).
        - **workers=1**: single-threaded. Multi-process workers would
          require the objective function to be picklable, which it
          isn't (closure over ``self.model``).

        Args:
            cycles: Cycle numbers (already truncated by caller).
            fluorescence: Per-cycle fluorescence (same length as
                ``cycles``).
            bounds: Parameter box. Keys are ``D0``, ``k``, ``P0``,
                ``F_bg_intercept``, ``F_bg_slope``; values are
                ``(lo, hi)`` tuples. ``D0`` bounds get log10'd
                internally.
            verbose: Print convergence diagnostics.

        Returns:
            ``(params, r2)`` where ``params`` is a 5-key dict
            (no tier-tagging metadata; that's added by the caller)
            and ``r2`` is the R² of the best DE solution against the
            input fluorescence.
        """
        if verbose:
            print("   Running Differential Evolution...")

        # Convert bounds to DE format: [(min, max), ...]
        # Use log space for D0 for better exploration
        de_bounds = [
            (np.log10(bounds['D0'][0]), np.log10(bounds['D0'][1])),  # log10(D0)
            (bounds['k'][0], bounds['k'][1]),                         # k
            (bounds['P0'][0], bounds['P0'][1]),                       # P0
            (bounds['F_bg_intercept'][0], bounds['F_bg_intercept'][1]),  # F_bg_intercept
            (bounds['F_bg_slope'][0], bounds['F_bg_slope'][1])        # F_bg_slope
        ]

        # Objective function: minimize sum of squared residuals
        def objective(x):
            log_D0, k, P0, F_bg_intercept, F_bg_slope = x
            D0 = 10 ** log_D0  # Convert back from log space

            try:
                y_pred = self.model.simulate_to_cycle(
                    D0, k, P0, cycles,
                    F_bg_intercept, F_bg_slope
                )

                ssr = np.sum((fluorescence - y_pred) ** 2)
                return ssr

            except Exception:
                return 1e10  # Return large penalty for invalid parameters

        # Run Differential Evolution
        result = differential_evolution(
            objective,
            de_bounds,
            strategy='best2bin',  # Robust strategy
            maxiter=200,          # Max iterations
            popsize=30,           # Population size (30 * 5 params = 150 candidates)
            tol=1e-7,             # Convergence tolerance
            mutation=(0.5, 1.5),  # Mutation factor range
            recombination=0.7,    # Crossover probability
            seed=_derive_seed(_SEED_OFFSET_DE),  # None in production (natural variation), deterministic in tests
            polish=True,          # Use L-BFGS-B to polish the best solution
            workers=1,            # Single-threaded (avoid pickling issues)
            updating='deferred'   # Evaluate full generation before updating
        )

        # Extract best parameters
        log_D0_best, k_best, P0_best, F_bg_int_best, F_bg_slope_best = result.x
        D0_best = 10 ** log_D0_best

        # Calculate R² for best solution
        y_pred = self.model.simulate_to_cycle(
            D0_best, k_best, P0_best, cycles,
            F_bg_int_best, F_bg_slope_best
        )

        r2 = calculate_r2(fluorescence, y_pred)

        # Package results
        params = {
            'D0': D0_best,
            'k': k_best,
            'P0': P0_best,
            'F_bg_intercept': F_bg_int_best,
            'F_bg_slope': F_bg_slope_best
        }

        if verbose:
            print(f"   DE converged: R² = {r2:.6f}")
            print(f"   Best params: D0={D0_best:.2e}, k={k_best:.4f}, P0={P0_best:.2e}")

        return params, r2

    def _fit_attempt(
        self,
        cycles: np.ndarray,
        fluorescence: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
        seed: int,
        lhs_D0: Optional[float] = None,
        lhs_k: Optional[float] = None,
        lhs_P0: Optional[float] = None,
        use_uniform_weighting: bool = False,
        plateau_weight_multiplier: float = 1.0
    ) -> Tuple[Dict[str, float], float]:
        """One TRF fit attempt with a configurable starting-point strategy.

        The actual workhorse of the multi-start strategy. Called many
        times per ``fit()``: once per Tier 1 attempt, once per Tier 1.5
        pattern retry, once per LHS sample in Tier 2 / 2.5. Each call
        runs a single ``scipy.optimize.curve_fit`` with Trust Region
        Reflective (TRF) — a bounded variant of Levenberg-Marquardt.

        Three starting-point modes (selected via constructor of the
        call site, not via this method's args):

        1. **Analytical-seeded** (``seed=1`` and analytical estimates
           available): use the analytical D0/k/P0 estimates as the
           starting point with ±20% jitter (±50% on k). Plus a
           safeguard: if the analytical k estimate is near a bound,
           jitter is biased toward the centre of the box to avoid
           starting in a region the optimiser can't search out of.
        2. **LHS-seeded** (``lhs_D0`` / ``lhs_k`` / ``lhs_P0``
           supplied): use the caller's LHS sample as the starting
           point. Background uses a uniform sample.
        3. **Pure random** (other seeds, no LHS): all five parameters
           drawn from uniform within their bounds. The escape hatch
           when both above strategies have failed.

        Args:
            cycles: Truncated cycle array.
            fluorescence: Truncated fluorescence array.
            bounds: 5-key parameter box.
            seed: Per-attempt seed (or ``None`` for fresh entropy).
                When 1, triggers the analytical-seeded mode.
                Higher integers trigger random-init mode.
            lhs_D0, lhs_k, lhs_P0: When provided, override the
                random init for those parameters. Used by Tier 2's
                LHS multi-start.
            use_uniform_weighting: When True, residuals are weighted
                uniformly. When False (default), the plateau region
                gets up to ``plateau_weight_multiplier``× more weight,
                helping the optimiser pin P0 down via the plateau
                level rather than letting plateau noise pull k/D0.
            plateau_weight_multiplier: Cap on the plateau weight.
                Default 1.0 (i.e. uniform — the optimiser was tuned
                this way; higher values consistently degraded D0
                accuracy in benchmarks).

        Returns:
            ``(params, r2)`` 5-key parameter dict + R² value.
        """
        np.random.seed(seed)
        
        # Use analytical estimates for first attempt (seed=1), random for others
        if seed == 1 and hasattr(self, 'analytical_estimates') and self.analytical_estimates is not None:
            # Smart initial guess from analytical estimation with added noise
            # Check if k estimate is near bounds - if so, use wider sampling
            k_estimate = self.analytical_estimates['k']
            k_range = bounds['k'][1] - bounds['k'][0]
            k_lower_dist = k_estimate - bounds['k'][0]
            k_upper_dist = bounds['k'][1] - k_estimate

            # If k is in bottom 30% of range, sample from middle-upper range
            if k_lower_dist < 0.3 * k_range:
                print(f"  ⚠️  k estimate ({k_estimate:.4f}) near lower bound - sampling from middle-upper range")
                k_init = np.random.uniform(
                    bounds['k'][0] + 0.4 * k_range,  # Start from 40% into range
                    bounds['k'][1]  # Up to upper bound
                )
            # If k is in top 30% of range, sample from lower-middle range
            elif k_upper_dist < 0.3 * k_range:
                print(f"  ⚠️  k estimate ({k_estimate:.4f}) near upper bound - sampling from lower-middle range")
                k_init = np.random.uniform(
                    bounds['k'][0],
                    bounds['k'][1] - 0.4 * k_range  # Up to 60% into range
                )
            else:
                # k is in middle - use normal ±50% noise
                k_init = k_estimate * np.random.uniform(0.5, 1.5)
                k_init = np.clip(k_init, bounds['k'][0], bounds['k'][1])

            # For other parameters, use ±20% noise
            D0_init = self.analytical_estimates['D0'] * np.random.uniform(0.8, 1.2)
            P0_init = self.analytical_estimates['P0'] * np.random.uniform(0.8, 1.2)
            F_bg_int_init = self.analytical_estimates['F_bg_intercept'] * np.random.uniform(0.9, 1.1)
            F_bg_slope_init = self.analytical_estimates['F_bg_slope']

            # Ensure within bounds
            P0_init = np.clip(P0_init, bounds['P0'][0], bounds['P0'][1])
            D0_init = np.clip(D0_init, bounds['D0'][0], bounds['D0'][1])
            F_bg_int_init = np.clip(F_bg_int_init, bounds['F_bg_intercept'][0], bounds['F_bg_intercept'][1])
            F_bg_slope_init = np.clip(F_bg_slope_init, bounds['F_bg_slope'][0], bounds['F_bg_slope'][1])
        else:
            # Random initial guess within bounds
            # Sample D0 uniformly in log space across the full range.
            # Early-Ct wells need D0 ≈ 1e-5 while late-Ct wells need
            # D0 ≈ 1e-12; biasing toward either end misses the other.
            # The multi-start approach (multiple seeds) and LHS tiers
            # ensure sufficient exploration across the full range.
            D0_init = 10**(np.random.uniform(
                np.log10(bounds['D0'][0]),
                np.log10(bounds['D0'][1])
            ))
            
            # Use LHS samples if provided, otherwise use the D0_init from above
            D0_init = lhs_D0 if lhs_D0 is not None else D0_init
            k_init = lhs_k if lhs_k is not None else np.random.uniform(bounds['k'][0], bounds['k'][1])
            P0_init = lhs_P0 if lhs_P0 is not None else np.random.uniform(bounds['P0'][0], bounds['P0'][1])
            F_bg_int_init = np.random.uniform(
                bounds['F_bg_intercept'][0], 
                bounds['F_bg_intercept'][1]
            )
            F_bg_slope_init = np.random.uniform(
                bounds['F_bg_slope'][0], 
                bounds['F_bg_slope'][1]
            )

        p0 = [D0_init, k_init, P0_init, F_bg_int_init, F_bg_slope_init]

        bounds_to_use = bounds.copy()

        # Debug: print initial guesses
        print(f"    Initial guesses: D0={D0_init:.2e}, k={k_init:.4f}, P0={P0_init:.4f}")

        # Check if we're using fixed background
        use_fixed_bg = hasattr(self, 'fixed_background') and self.fixed_background is not None

        # Prepare bounds for curve_fit
        # Reparameterize D0 → log10(D0) for better numerical conditioning.
        # D0 can span many orders of magnitude (e.g. 1e-11 to 1e-2), and TRF
        # struggles with Jacobians that vary by 9+ orders of magnitude in
        # linear space.  In log space the parameter landscape is smooth and
        # TRF converges reliably.
        log10_D0_init = np.log10(max(p0[0], 1e-30))  # protect against 0
        log10_D0_lo   = np.log10(max(bounds_to_use['D0'][0], 1e-30))
        log10_D0_hi   = np.log10(max(bounds_to_use['D0'][1], 1e-30))

        if use_fixed_bg:
            # Fit only log10(D0), k, P0 (background is fixed)
            lower_bounds = [
                log10_D0_lo,
                bounds_to_use['k'][0],
                bounds_to_use['P0'][0]
            ]
            upper_bounds = [
                log10_D0_hi,
                bounds_to_use['k'][1],
                bounds_to_use['P0'][1]
            ]
            # Initial guess for 3 parameters
            p0_reduced = [log10_D0_init, p0[1], p0[2]]  # log10(D0), k, P0
            fixed_bg_int = self.fixed_background['F_bg_intercept']
            fixed_bg_slope = self.fixed_background['F_bg_slope']
        else:
            # Fit all 5 parameters
            lower_bounds = [
                log10_D0_lo,
                bounds_to_use['k'][0],
                bounds_to_use['P0'][0],
                bounds_to_use['F_bg_intercept'][0],
                bounds_to_use['F_bg_slope'][0]
            ]
            upper_bounds = [
                log10_D0_hi,
                bounds_to_use['k'][1],
                bounds_to_use['P0'][1],
                bounds_to_use['F_bg_intercept'][1],
                bounds_to_use['F_bg_slope'][1]
            ]
            p0_reduced = [log10_D0_init, p0[1], p0[2], p0[3], p0[4]]

        # Validate and fix bounds before fitting
        # Note: position 0 is log10(D0) internally, but we label it 'log10_D0' for clarity
        param_names = ['log10_D0', 'k', 'P0', 'F_bg_intercept', 'F_bg_slope'] if not use_fixed_bg else ['log10_D0', 'k', 'P0']
        for i, name in enumerate(param_names):
            if lower_bounds[i] >= upper_bounds[i]:
                # Fix collapsed bounds by adding minimum margin
                if name == 'F_bg_intercept':
                    center = lower_bounds[i]
                    margin = max(0.05, 0.1 * abs(center))
                    lower_bounds[i] = center - margin
                    upper_bounds[i] = center + margin
                    print(f"  Warning: Fixed collapsed {name} bounds to [{lower_bounds[i]:.6f}, {upper_bounds[i]:.6f}]")
                elif name == 'F_bg_slope':
                    center = lower_bounds[i]
                    margin = 0.001
                    lower_bounds[i] = center - margin
                    upper_bounds[i] = center + margin
                    print(f"  Warning: Fixed collapsed {name} bounds to [{lower_bounds[i]:.6f}, {upper_bounds[i]:.6f}]")
                else:
                    raise ValueError(f"Invalid bounds for {name}: [{lower_bounds[i]}, {upper_bounds[i]}]. Lower must be < upper.")

        # Validate initial guesses are within bounds
        for i, name in enumerate(param_names):
            if p0_reduced[i] < lower_bounds[i] or p0_reduced[i] > upper_bounds[i]:
                print(f"  Warning: {name} initial guess {p0_reduced[i]:.2e} outside bounds [{lower_bounds[i]:.2e}, {upper_bounds[i]:.2e}]")
                p0_reduced[i] = np.clip(p0_reduced[i], lower_bounds[i], upper_bounds[i])
                print(f"           Clipped to {p0_reduced[i]:.2e}")

        # Create adaptive weights to emphasize exponential growth and plateau regions
        # Identify where signal rises above baseline (>10% of max fluorescence)
        F_max = np.max(fluorescence)
        F_min = np.min(fluorescence)
        F_range = F_max - F_min
        threshold = F_min + 0.1 * F_range

        # Find first point above threshold (start of exponential growth)
        signal_start_idx = np.where(fluorescence > threshold)[0]
        if len(signal_start_idx) > 0:
            signal_start_idx = signal_start_idx[0]
        else:
            # Fallback: use 60% point
            signal_start_idx = int(len(cycles) * 0.6)

        n_points = len(cycles)
        weights = np.ones(n_points)

        # Use moderate fixed weighting that works well for most samples
        # Increase from 1.0 to max_weight over the exponential + plateau region
        # Use plateau_weight_multiplier parameter to control emphasis
        if use_uniform_weighting:
            max_weight = 1.0  # No weighting, uniform across all points
        else:
            max_weight = plateau_weight_multiplier

        # Weight exponential + plateau region (from signal start to end)
        for i in range(signal_start_idx, n_points):
            progress = (i - signal_start_idx) / max(1, n_points - signal_start_idx)
            weights[i] = 1.0 + (max_weight - 1.0) * progress  # Linear increase from 1 to max_weight

        # Apply weights through sigma parameter (smaller sigma = higher weight)
        # sigma = 1/sqrt(weight)
        sigma = 1.0 / np.sqrt(weights)

        # Fit using Trust Region Reflective (supports bounds, similar performance to LM)
        # Note: first parameter is log10(D0), converted to D0 inside the lambda
        if use_fixed_bg:
            # Fit only log10(D0), k, P0 with fixed background
            popt_3param, _ = curve_fit(
                lambda n, log_D0, k, P0: self.model.simulate_to_cycle(
                    10**log_D0, k, P0, n, fixed_bg_int, fixed_bg_slope
                ),
                cycles,
                fluorescence,
                p0=p0_reduced,
                sigma=sigma,  # Weights via inverse uncertainty
                bounds=(lower_bounds, upper_bounds),
                method='trf',  # Trust Region Reflective (supports bounds)
                maxfev=10000
            )
            # Convert log10(D0) back to D0 and expand to 5 parameters
            popt = [10**popt_3param[0], popt_3param[1], popt_3param[2], fixed_bg_int, fixed_bg_slope]
        else:
            # Fit all 5 parameters (log10(D0), k, P0, bg_int, bg_slope)
            popt_raw, _ = curve_fit(
                lambda n, log_D0, k, P0, bg_int, bg_slope: self.model.simulate_to_cycle(
                    10**log_D0, k, P0, n, bg_int, bg_slope
                ),
                cycles,
                fluorescence,
                p0=p0_reduced,
                sigma=sigma,  # Weights via inverse uncertainty
                bounds=(lower_bounds, upper_bounds),
                method='trf',  # Trust Region Reflective (supports bounds)
                maxfev=10000
            )
            # Convert log10(D0) back to D0
            popt = [10**popt_raw[0], popt_raw[1], popt_raw[2], popt_raw[3], popt_raw[4]]

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
            'ssr': ssr,  # Add SSR to params
            # Store initial guess for stuck detection
            'k_init': k_init,
            'P0_init': P0_init,
            'D0_init': D0_init
        }

        return params, r2
    
    def calculate_fit_metrics(self) -> Dict[str, float]:
        """Compute the standard suite of fit-quality metrics on the fit window.

        Reads the per-instance ``cycles_fit`` / ``fluorescence_fit`` /
        ``optimal_params`` set by ``fit()``, predicts fluorescence,
        and returns R²/RMSE/MAE/MAPE/NRMSE/SSR/AIC/BIC/χ²_reduced.

        The reason this is a separate method (rather than always
        rolled into ``fit()``): callers sometimes mutate
        ``cycles_fit`` / ``fluorescence_fit`` between fit and metrics
        (e.g. ``run_batch.compute_ct`` swaps them temporarily to a
        ROX-normalised version). Recomputing the metrics on the
        post-mutation arrays would be wrong; instead the caller
        snapshots the original metrics by calling this method
        before the swap.

        Note: AIC and BIC use ``k=5`` (the model's parameter count);
        this is correct only when all 5 parameters were free during
        the fit. With ``fix_background=True`` (production default)
        the *effective* parameter count is 3, but we report k=5 for
        consistency across modes — the relative AIC/BIC numbers are
        what's interpretable in practice anyway.

        Returns:
            Dict with keys ``r_squared``, ``rmse``, ``mae``,
            ``mape``, ``nrmse``, ``ssr``, ``aic``, ``bic``,
            ``reduced_chi_sq``, ``n_points``, ``n_params``, ``dof``.

        Raises:
            ValueError: If ``fit()`` hasn't been run yet.
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
        """Predict fluorescence at arbitrary cycle numbers using fitted parameters.

        Thin wrapper over ``MAK2Model.simulate_to_cycle`` that pulls
        the parameters from the optimizer's stored ``optimal_params``
        when the caller doesn't override them.

        Args:
            cycles: Cycle numbers to evaluate. Need not match the
                fit window — common patterns include predicting on
                the *full* curve to overlay against raw data, or on
                an extended grid for a smooth plot line.
            params: Optional parameter override. When ``None``
                (typical), uses the optimizer's fitted
                ``optimal_params``.

        Returns:
            Predicted fluorescence array (same length as ``cycles``).

        Raises:
            ValueError: If ``params`` is None and ``fit()`` hasn't
                been run yet.
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

    def check_plateau_overshoot(
        self,
        overshoot_threshold: float = 0.05,
        verbose: bool = False
    ) -> Tuple[bool, float]:
        """Detect whether the fit's plateau height exceeds the observed plateau.

        Why this gate exists: a fit can have excellent R² and still be
        wrong if it claims a higher plateau than the data actually
        shows. Plateau height is set in MAK2 by ``k * P0``, and the
        optimiser can trade those off against each other in ways that
        leave the early-cycle exponential portion well-fit while
        wildly inflating P0. The downstream consequence is too-low
        D0 (because ``D0`` and ``P0`` are inversely correlated for a
        fixed exponential portion), which propagates into the
        absolute-quantification answer.

        The check is on background-*subtracted* signal magnitudes,
        not raw fluorescence — otherwise the comparison is dominated
        by background offset and tells us nothing about plateau.

        Args:
            overshoot_threshold: Maximum tolerated overshoot ratio.
                Default 0.05 (5%). The Tier 4 caller passes 0.0
                (any overshoot triggers a refit); other callers can
                relax it.
            verbose: Print diagnostic info.

        Returns:
            ``(overshoots, ratio)``. ``ratio`` is
            ``(max_predicted - max_observed) / max_observed`` after
            baseline subtraction; positive = overshoot.

        Raises:
            ValueError: If ``fit()`` hasn't been run yet.
        """
        if self.cycles_fit is None or self.fluorescence_fit is None:
            raise ValueError("No fitted data available. Run fit() first.")

        if self.optimal_params is None:
            raise ValueError("No fitted parameters available. Run fit() first.")

        fluorescence = self.fluorescence_fit
        cycles = self.cycles_fit

        # Get model predictions
        F_pred = self.predict(cycles)

        # Detect and subtract baseline from both observed and predicted
        baseline_end = max(3, int(len(cycles) * 0.25))
        early_fluor_obs = fluorescence[0:baseline_end]
        early_fluor_pred = F_pred[0:baseline_end]

        # Check if data is already baseline-corrected
        already_corrected = (np.mean(early_fluor_obs) < 0.1) or (np.any(early_fluor_obs < 0))

        if already_corrected:
            # Data already baseline-corrected, use directly
            signal_obs = fluorescence
            signal_pred = F_pred
        else:
            # Subtract baseline to get actual amplification signal
            baseline_obs = np.mean(early_fluor_obs)
            baseline_pred = np.mean(early_fluor_pred)
            signal_obs = fluorescence - baseline_obs
            signal_pred = F_pred - baseline_pred

        # Compare maximum signals
        max_signal_obs = np.max(signal_obs)
        max_signal_pred = np.max(signal_pred)

        # Calculate overshoot ratio
        if max_signal_obs > 0:
            overshoot_ratio = (max_signal_pred / max_signal_obs) - 1.0
        else:
            overshoot_ratio = 0.0  # Can't determine if no signal

        overshoots = overshoot_ratio >= overshoot_threshold

        if verbose:
            print(f"Plateau overshoot check:")
            print(f"  Max observed signal: {max_signal_obs:.6f}")
            print(f"  Max predicted signal: {max_signal_pred:.6f}")
            print(f"  Overshoot ratio: {overshoot_ratio:.1%}")
            print(f"  Threshold: {overshoot_threshold:.1%}")
            print(f"  Result: {'❌ OVERSHOOT' if overshoots else '✓ OK'}")

        return overshoots, overshoot_ratio

    def calculate_ct(
        self,
        method: str = 'threshold',
        threshold: Optional[float] = None,
        baseline_cycles: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """Compute Ct (threshold cycle) from the stored fit, instrument-equivalent.

        Why this is in the optimizer (not a separate utility): the
        Ct calculation operates on the same per-well state the
        optimizer holds (``cycles_fit``, ``fluorescence_fit``) plus
        a separate baseline regression. Keeping it here lets
        ``run_batch.compute_ct`` swap a ROX-normalised
        ``fluorescence_fit`` in temporarily and call this method to
        get a ROX-aware Ct without re-running the kinetic fit.

        Three methods (``method`` arg):

        - ``'threshold'`` (default, instrument-standard): find the
          first cycle where ΔRn (background-subtracted fluorescence)
          crosses ``threshold``. Linearly interpolates between
          cycles for sub-cycle precision.
        - ``'second_derivative'``: find the cycle of maximum second
          derivative (the inflection point). Method-of-record on
          some instruments; less common in industry today.
        - ``'regression'``: log-linear fit on the exponential phase,
          back-extrapolated to the threshold. Mostly historical.

        The baseline regression matters: a constant-mean
        subtraction leaves residual slope in ΔRn which makes the
        threshold cross early in sloping-baseline wells. We fit a
        line through the baseline window and subtract it (the
        same approach instrument firmware uses).

        Auto-detection: if the input fluorescence already looks
        baseline-subtracted (mean < 0.1 in early cycles, or any
        negative values), the baseline subtraction step is skipped.
        Important for ABI Multicomponent data that's been
        pre-processed.

        Args:
            method: One of ``'threshold'``, ``'second_derivative'``,
                ``'regression'``.
            threshold: Fluorescence threshold for the threshold
                method. ``None`` triggers auto-calculation as
                ``baseline_mean + 10 * baseline_SD``.
            baseline_cycles: ``(start, end)`` index pair for the
                baseline window. ``None`` uses the first 15% of
                cycles. ``run_batch.compute_ct`` passes the
                instrument metadata's ``Baseline Start``/``End``
                here so the Ct matches the instrument's calculation.

        Returns:
            Dict with ``ct`` (the Ct value), ``method`` (the method
            used), ``threshold`` (the threshold actually used),
            ``baseline_mean`` / ``baseline_sd`` /
            ``baseline_slope`` / ``baseline_intercept`` (the fitted
            baseline regression — passed back to the caller for
            recording in the result table).

        Raises:
            ValueError: If ``fit()`` hasn't been run yet.
        """
        if self.cycles_fit is None or self.fluorescence_fit is None:
            raise ValueError("No fitted data available. Run fit() first.")

        cycles = self.cycles_fit
        fluorescence = self.fluorescence_fit

        # Auto-detect if data is already baseline-subtracted
        # Use conservative baseline window (first 15%) to avoid including
        # early amplification from high-copy samples
        baseline_end = max(3, int(len(cycles) * 0.15))
        early_fluor = fluorescence[0:baseline_end]

        # Data is likely pre-baseline-subtracted if:
        # 1. Mean of early cycles is very close to zero (< 0.1)
        # 2. OR there are negative values in early cycles
        already_baseline_subtracted = (np.mean(early_fluor) < 0.1) or (np.any(early_fluor < 0))

        if already_baseline_subtracted:
            # Data already baseline-corrected, use directly
            delta_rn = fluorescence
            baseline_mean = 0.0
            baseline_sd = np.std(early_fluor) if np.std(early_fluor) > 0 else 0.01
            baseline_slope = 0.0
            baseline_intercept = 0.0
        else:
            # Determine baseline region
            if baseline_cycles is None:
                baseline_cycles = (0, baseline_end)

            baseline_fluor = fluorescence[baseline_cycles[0]:baseline_cycles[1]]
            baseline_cycles_arr = cycles[baseline_cycles[0]:baseline_cycles[1]]
            baseline_sd = np.std(baseline_fluor)

            # Fit a linear regression to the baseline region and subtract the
            # extrapolated line from the full curve (ΔRn = Rn - baseline_line).
            # This matches the instrument's per-well baseline correction, which
            # removes both the DC offset AND any fluorescence drift (slope) before
            # applying the threshold.  A constant-mean subtraction leaves residual
            # slope in ΔRn, causing the threshold to be crossed early in noisy or
            # sloping-baseline wells.
            if len(baseline_cycles_arr) >= 2:
                coeffs = np.polyfit(baseline_cycles_arr, baseline_fluor, 1)
            else:
                coeffs = np.array([0.0, np.mean(baseline_fluor)])
            baseline_slope     = coeffs[0]
            baseline_intercept = coeffs[1]
            linear_baseline    = np.polyval(coeffs, cycles)
            baseline_mean      = float(np.mean(linear_baseline))
            delta_rn           = fluorescence - linear_baseline

        results = {
            'baseline_mean':      baseline_mean,
            'baseline_sd':        baseline_sd,
            'baseline_slope':     baseline_slope,
            'baseline_intercept': baseline_intercept,
            'method':             method
        }

        if method == 'threshold':
            # Auto-calculate threshold if not provided
            if threshold is None:
                # Use the larger of 10×SD or 5% of the dynamic range.
                # 10×SD can be too low when baseline noise is very small,
                # whether data is baseline-subtracted or not. The 5% floor
                # ensures the threshold sits in the early exponential phase.
                sd_threshold = 10 * baseline_sd
                dynamic_range = np.max(fluorescence) - baseline_mean
                range_threshold = 0.05 * dynamic_range
                threshold = max(sd_threshold, range_threshold)

            results['threshold'] = threshold

            # Find where delta_rn crosses threshold
            above_threshold = delta_rn >= threshold

            if not np.any(above_threshold):
                results['ct'] = np.nan
                return results

            crossing_idx = np.where(above_threshold)[0][0]

            # If crossing at first cycle, the threshold is likely too low or baseline incorrect
            # Return NaN rather than an unrealistic Ct value
            if crossing_idx == 0:
                results['ct'] = np.nan
                return results

            # Linear interpolation
            c1, c2 = cycles[crossing_idx - 1], cycles[crossing_idx]
            f1, f2 = delta_rn[crossing_idx - 1], delta_rn[crossing_idx]

            if f2 == f1:
                ct = c1
            else:
                ct = c1 + (threshold - f1) * (c2 - c1) / (f2 - f1)

            results['ct'] = ct

        elif method == 'second_derivative':
            # Maximum second derivative (inflection point)
            first_deriv = np.gradient(delta_rn)
            second_deriv = np.gradient(first_deriv)
            max_idx = np.argmax(second_deriv)
            results['ct'] = cycles[max_idx]

        elif method == 'regression':
            # Linear regression on exponential phase
            # Use log-transformed delta_rn (remove non-positive values)
            valid_mask = delta_rn > 0
            if np.sum(valid_mask) < 5:
                results['ct'] = np.nan
                return results

            cycles_valid = cycles[valid_mask]
            log_fluor = np.log(delta_rn[valid_mask])

            # Find most linear window (sliding window of 5 points)
            window_size = 5
            best_r2 = -np.inf
            best_coeffs = None

            for i in range(len(cycles_valid) - window_size):
                x = cycles_valid[i:i+window_size]
                y = log_fluor[i:i+window_size]

                coeffs = np.polyfit(x, y, deg=1)
                y_pred = np.polyval(coeffs, x)
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

                if r2 > best_r2:
                    best_r2 = r2
                    best_coeffs = coeffs

            if best_coeffs is None:
                results['ct'] = np.nan
                return results

            slope, intercept = best_coeffs

            # Ct where regression line crosses baseline threshold (10× SD)
            threshold_log = np.log(10 * baseline_sd)
            ct = (threshold_log - intercept) / slope

            results['ct'] = ct
            results['efficiency'] = np.exp(slope) - 1
            results['r2'] = best_r2

        else:
            raise ValueError(f"Unknown method: {method}")

        return results

