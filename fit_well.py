"""Per-well MAK2+ fitting wrapper — encapsulates the preprocessing pipeline.

The MAK2+ optimizer (``optimizer.MAK2Optimizer``) needs the right
*fit window* and *background pre-estimates* to produce good fits.
Specifically:

  - **Left-trim**: the fit window must start ~10 cycles before the
    inflection (max-slope cycle), not at cycle 1. If the window
    includes 25-30 baseline cycles ahead of the take-off, those
    baseline residuals dominate the SSR — the optimizer minimises
    them by tilting ``F_bg_slope`` to fit baseline drift, which
    pulls the growth-region fit off the data. Cutting the long
    baseline out leaves the optimizer to do what it's good at.

  - **Right-trim**: handled inside ``optimizer.fit`` via
    ``auto_truncate`` (cuts off cycles past the inflection + a small
    offset, where post-plateau drift can confuse the fit).

  - **Background pre-estimation**: a quick polyfit on the few
    cycles just before the fit window gives an estimate of
    ``F_bg_slope`` / ``F_bg_intercept``. Passed to the optimizer as
    both initial guess and tight bounds — without it the optimizer
    has to discover background and growth jointly, which is
    under-determined for clean fits and biases noisy ones.

This wrapper lived inside ``run_batch.run_pass1`` until extraction.
Production callers (``run_batch.py``, ``app.py`` batch mode) already
go through this pipeline; the PCRedux scoring driver
(``tuning/score_gates.py``) used to call the bare optimizer and skip
the preprocessing, which produced visibly worse fits on late
amplifiers (long baseline + steep take-off). Extracting fixes that.

The Phase-1 unification task in ``CLAUDE.md`` (per-well preprocessing
duplicated across run_batch / app.py batch / app.py single-sample)
points at this function as the unification target. For now, only the
PCRedux scoring path is rewired; ``run_batch`` continues to inline
its own copy to preserve bit-equality on the Boggy regression
fixture. Subsequent work can flip run_batch over to call ``fit_well``
once a new fixture is captured.
"""
from __future__ import annotations

import numpy as np

from mak2_model import MAK2Model
from optimizer import MAK2Optimizer
# smart_start and adaptive_window_extension currently live in run_batch
# alongside the run_pass1 driver. Importing them keeps this extraction
# bit-equivalent to the inlined run_pass1 path; a later commit can move
# them here and have run_batch import from this module instead.
from run_batch import smart_start, adaptive_window_extension


def fit_well(
    cycles,
    fluor_data,
    *,
    first_fit_cycle: float = 3.0,
    cycles_before_max: int = 15,
    cycles_after_max: int = 4,
    auto_truncate: bool = True,
    truncate_cycle: float | None = None,
    fit_bounds: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Fit one well end-to-end: window selection + bg pre-estimation + MAK2 fit.

    Mirrors the inline pipeline in ``run_batch.run_pass1`` (lines
    ~530-625 as of writing). Returns a per-well result dict shaped
    like the one ``score_gates.fit_curve`` and ``run_pass1`` produce,
    so existing gate-application and scoring code can consume it
    unchanged.

    Args:
        cycles: 1-D array of cycle numbers (typically 1..45).
        fluor_data: 1-D array of fluorescence values, same length as
            ``cycles``.
        first_fit_cycle: Smallest cycle that's eligible to be the
            fit-window start. Defaults to 3 (matches run_batch's
            ``FIRST_FIT_CYCLE``).
        cycles_before_max: Default span from the inflection back to
            the fit-window start. ``adaptive_window_extension``
            extends this if needed to include ≥3 baseline cycles.
        cycles_after_max: Cycles past the inflection retained in the
            fit window (the optimizer's right-truncation knob).
        auto_truncate: Forwarded to ``optimizer.fit``. Defaults True.
        truncate_cycle: Optional manual right-truncation cycle.
        fit_bounds: User-supplied bounds dict merged with the
            algorithmically-derived ``F_bg_slope`` / ``F_bg_intercept``
            / ``D0`` bounds (the merge prefers algorithmic for the
            background pair).
        verbose: Forwarded to ``optimizer.fit``.

    Returns:
        Per-well result dict matching the shape used downstream by
        ``run_quality_gates`` and the PCRedux scoring driver:
        ``R2, D0, k, P0, F_bg_intercept, F_bg_slope,
        fit_start_cycle, fit_end_cycle, fluor_data, error, Success``.
        On failure, parameter fields are ``None`` and ``error``
        carries the exception message.
    """
    cycles = np.asarray(cycles, dtype=float)
    fluor_data = np.asarray(fluor_data, dtype=float)

    floor_idx = int(np.searchsorted(cycles, first_fit_cycle))

    # Inflection search + initial left-trim.
    fit_start_idx, max_slope_idx = smart_start(
        fluor_data, cycles, floor_idx, cycles_before_max
    )

    # Background pre-estimation on the ≤8 cycles just before the window.
    bg_pre_start = max(floor_idx, fit_start_idx - 8)
    bg_c = cycles[bg_pre_start:fit_start_idx]
    bg_f = fluor_data[bg_pre_start:fit_start_idx]
    if len(bg_c) >= 2:
        coeffs = np.polyfit(bg_c, bg_f, 1)
        bg_slope_est = float(coeffs[0])
        bg_int_est = float(coeffs[1])
    else:
        bg_slope_est = 0.0
        bg_int_est = (
            float(fluor_data[fit_start_idx])
            if fit_start_idx < len(fluor_data) else 0.0
        )

    # Extend window backward until ≥3 baseline cycles are inside.
    fit_start_idx, bg_slope_est, bg_int_est = adaptive_window_extension(
        fluor_data, cycles, fit_start_idx, max_slope_idx,
        floor_idx, cycles_before_max, bg_slope_est, bg_int_est,
    )

    cycles_fit = cycles[fit_start_idx:]
    fluor_fit = fluor_data[fit_start_idx:]
    F_range = float(np.max(fluor_fit) - np.min(fluor_fit))

    # Safety net: if the chosen window contains <70% of the total
    # fluorescence range, smart-start missed the sigmoid. Fall back to
    # using the whole curve from floor_idx and re-estimate background
    # on the leading 8 cycles.
    total_range = float(np.max(fluor_data) - np.min(fluor_data))
    if total_range > 0 and F_range < 0.7 * total_range:
        fit_start_idx = floor_idx
        cycles_fit = cycles[fit_start_idx:]
        fluor_fit = fluor_data[fit_start_idx:]
        F_range = float(np.max(fluor_fit) - np.min(fluor_fit))
        bg_pre_start = floor_idx
        bg_c = cycles[bg_pre_start:bg_pre_start + 8]
        bg_f = fluor_data[bg_pre_start:bg_pre_start + 8]
        if len(bg_c) >= 2:
            coeffs = np.polyfit(bg_c, bg_f, 1)
            bg_slope_est = float(coeffs[0])
            bg_int_est = float(coeffs[1])

    # Algorithmic bounds for D0 + tight bounds around pre-estimated background.
    slope_delta = max(abs(bg_slope_est) * 0.40, F_range * 0.002)
    int_delta = max(abs(bg_int_est) * 0.005, F_range * 0.03)
    bg_bounds = {
        'D0':             (1e-15, max(F_range, 1.0)),
        'F_bg_slope':     (bg_slope_est - slope_delta, bg_slope_est + slope_delta),
        'F_bg_intercept': (bg_int_est - int_delta,    bg_int_est + int_delta),
    }
    non_bg_bounds = {
        k: v for k, v in (fit_bounds or {}).items()
        if k not in ('F_bg_slope', 'F_bg_intercept')
    }
    merged_bounds = {**bg_bounds, **non_bg_bounds}

    try:
        opt = MAK2Optimizer(MAK2Model())
        params = opt.fit(
            cycles_fit, fluor_fit,
            cycles_after_max=cycles_after_max,
            auto_truncate=auto_truncate,
            truncate_cycle=truncate_cycle,
            bounds=merged_bounds,
            fixed_background_values={
                'F_bg_slope':     bg_slope_est,
                'F_bg_intercept': bg_int_est,
            },
            verbose=verbose,
        )
        metrics = opt.calculate_fit_metrics()
        fit_end_cycle = (
            float(opt.cycles_fit[-1])
            if opt.cycles_fit is not None and len(opt.cycles_fit) > 0
            else float(cycles[-1])
        )
        return {
            'Sample': 'x',
            'R2': metrics['r_squared'],
            'D0': params['D0'],
            'k':  params['k'],
            'P0': params['P0'],
            'F_bg_intercept': params['F_bg_intercept'],
            'F_bg_slope':     params['F_bg_slope'],
            'fit_start_cycle': float(cycles[fit_start_idx]),
            'fit_end_cycle':   fit_end_cycle,
            'fluor_data':      fluor_data,
            'error':           None,
            'Success':         '✓',
        }
    except Exception as e:
        return {
            'Sample': 'x',
            'R2': None, 'D0': None, 'k': None, 'P0': None,
            'F_bg_intercept': None, 'F_bg_slope': None,
            'fit_start_cycle': float('nan'),
            'fit_end_cycle':   float('nan'),
            'fluor_data':      fluor_data,
            'error':           str(e),
            'Success':         '',
        }
