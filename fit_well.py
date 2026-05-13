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
from data_processing import estimate_baseline_end
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

    # ── Background pre-estimation in a near-elbow window ──────────────
    # We use ``estimate_baseline_end`` (iterative 5σ signal-vs-noise
    # detection) to locate where the baseline ends, then fit bg on a
    # window of up to ``bg_window_size`` cycles ending at that index.
    # Anchoring at the genuine baseline-end is what matters; the window
    # size is a tradeoff:
    #   - Too narrow (e.g., 8 cycles): polyfit slope is dominated by
    #     local noise and may be ~0 even when a real bleach trend exists.
    #   - Too wide (entire baseline from cycle 3): early cycles get
    #     disproportionate weight. Cycles 1-5 are often unreliable
    #     (dye equilibration) and dye photobleaching is typically
    #     exponential — a long-window linear fit overestimates the
    #     decay rate near the elbow.
    #   - 12 cycles: enough for a statistically stable polyfit, short
    #     enough that exp ≈ linear holds, and skips most of the dye-
    #     equilibration noise on a 40-cycle assay (baseline_end ~25,
    #     window = cycles 14-24).
    # Independent of where smart_start / adaptive_window_extension
    # place the fit window, which matters because for some curves
    # adaptive extension can collapse ``fit_start_idx`` all the way
    # back to ``floor_idx``, leaving only the noisiest early cycles.
    baseline_end_idx = estimate_baseline_end(
        cycles, fluor_data, first_cycle_idx=floor_idx,
    )
    bg_window_size = 12
    bg_pre_start = max(floor_idx, baseline_end_idx - bg_window_size)
    bg_c = cycles[bg_pre_start:baseline_end_idx]
    bg_f = fluor_data[bg_pre_start:baseline_end_idx]
    if len(bg_c) >= 2:
        coeffs = np.polyfit(bg_c, bg_f, 1)
        bg_slope_est = float(coeffs[0])
        bg_int_est = float(coeffs[1])
    else:
        # Fallback: not enough baseline cycles. Use the median of whatever
        # baseline we have and assume flat slope.
        bg_slope_est = 0.0
        bg_int_est = (
            float(np.median(fluor_data[floor_idx:baseline_end_idx]))
            if baseline_end_idx > floor_idx else 0.0
        )

    # ── Fit window placement (smart_start finds inflection) ───────────
    fit_start_idx, max_slope_idx = smart_start(
        fluor_data, cycles, floor_idx, cycles_before_max
    )

    # Extend window backward until ≥3 baseline cycles are inside. We
    # pass our stable bg estimate so the "baseline-like" check uses the
    # right reference; we then *discard* the recomputed bg values, since
    # they'd revert to the old "tiny pre-window polyfit" behaviour we're
    # trying to avoid.
    fit_start_idx, _, _ = adaptive_window_extension(
        fluor_data, cycles, fit_start_idx, max_slope_idx,
        floor_idx, cycles_before_max, bg_slope_est, bg_int_est,
    )

    cycles_fit = cycles[fit_start_idx:]
    fluor_fit = fluor_data[fit_start_idx:]
    F_range = float(np.max(fluor_fit) - np.min(fluor_fit))

    # Safety net: if the chosen window contains <70% of the total
    # fluorescence range, smart-start missed the sigmoid. Fall back to
    # using the whole curve from floor_idx. We keep the bg estimate
    # anchored to the algorithmic baseline-end (above), not re-fit on
    # the leading cycles.
    total_range = float(np.max(fluor_data) - np.min(fluor_data))
    if total_range > 0 and F_range < 0.7 * total_range:
        fit_start_idx = floor_idx
        cycles_fit = cycles[fit_start_idx:]
        fluor_fit = fluor_data[fit_start_idx:]
        F_range = float(np.max(fluor_fit) - np.min(fluor_fit))

    # Tight bounds around the window-based background pre-estimate. We
    # intentionally do *not* set a D0 bound here: the optimizer treats
    # 'D0' in bounds as a signal to skip its own analytical bounds
    # estimation (see optimizer.py:383 `use_analytical_init = 'D0' not
    # in bounds`), and that estimation is where the sophisticated
    # background-subtracted-data handling lives (statistical σ-based
    # bg bounds via `slope_uncertainty`, baseline threshold detection
    # for near-zero-fluorescence data, negative-intercept allowance).
    # By omitting D0, we get the analytical D0/k/P0 bounds *plus* our
    # tighter window-based F_bg_slope/F_bg_intercept (the merge loop in
    # optimizer.py preserves bounds already present in `bounds`).
    slope_delta = max(abs(bg_slope_est) * 0.40, F_range * 0.002)
    int_delta = max(abs(bg_int_est) * 0.005, F_range * 0.03)
    bg_bounds = {
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
