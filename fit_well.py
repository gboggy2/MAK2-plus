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

The per-well preprocessing (take-off detection, bg pre-estimation,
window selection, bound construction) lives in ``prepare_fit_inputs``.
Both ``fit_well`` (used by the PCRedux scoring driver and any
single-well API caller) and ``run_batch.run_pass1`` (the production
batch driver) route through it, so improvements land everywhere at
once. The plate regression test verified bit-equality of optimizer
outputs across this refactor; ``fit_well`` and the unified
``run_pass1`` path produce identical fits on the Boggy fixture.
"""
from __future__ import annotations

import os

import numpy as np

from mak2_model import MAK2Model
from optimizer import MAK2Optimizer
from data_processing import estimate_baseline_end
# smart_start and adaptive_window_extension currently live in run_batch
# alongside the run_pass1 driver. Importing them keeps this extraction
# bit-equivalent to the inlined run_pass1 path; a later commit can move
# them here and have run_batch import from this module instead.
from run_batch import smart_start, adaptive_window_extension
from toe_prefit import (
    stage0_ct_from_inflection,
    stage1_toe_fit,
    stage3_toe_gate,
    baseline_std_from_prefit,
    TOE_FIT_R2_MIN,
    TOE_D0_BOUND_FACTOR,
)


def find_take_off_idx(cycles, fluor_data, floor_idx):
    """Return the index where amplification begins (= baseline end).

    Conceptually: the "baseline" is the last stable region before the
    curve starts rising into the sigmoid. We find it by walking
    *backward* from the inflection cycle (max d1), looking for the
    nearest 5-cycle window whose polyfit residual is small (≤ 0.5% of
    F_range). The cycle immediately after that window is the take-off.

    Why backward-from-inflection: it handles three regimes that
    ``estimate_baseline_end`` (forward signal-vs-noise) gets wrong —

    1. **Sharp early amplifiers** (rutledge X1, boggy F1, maro1.42):
       forward signal-vs-noise overshoots into the rise because the
       noise floor is tiny. Backward scan from inflection lands inside
       the truly flat pre-elbow cycles.

    2. **Biphasic baselines** (maro3.116: early hump then sag then
       amplification). Forward scan locks onto whichever pseudo-stable
       region appears first (the early hump) and reports a positive bg
       slope that doesn't reflect the local drift near the elbow.
       Backward scan finds the *last* stable region before the real
       take-off, even if there's a hump earlier in the curve.

    3. **Long drifting baselines** (media-1 maro2.*): both methods land
       in similar places; this approach gives a tighter answer because
       it walks until residual stays low.

    Returns inflection_idx if no clean stable region exists (failure
    fallback — caller can use ``estimate_baseline_end`` as a backup).
    """
    cycles = np.asarray(cycles, dtype=float)
    fluor_data = np.asarray(fluor_data, dtype=float)
    n = len(cycles)
    if n - floor_idx < 8:
        return n - 1

    # Locate the inflection cycle as the max-slope point. Use a small
    # smoothing to reduce noise sensitivity.
    d1 = np.gradient(fluor_data, cycles)
    # Restrict argmax to cycles past floor_idx so leading noise can't win.
    inflection_idx = int(np.argmax(d1[floor_idx:])) + floor_idx
    if inflection_idx - floor_idx < 5:
        return inflection_idx

    F_range = float(np.max(fluor_data) - np.min(fluor_data))
    if F_range <= 0:
        return inflection_idx

    # Walk backward from inflection_idx - 1. At each candidate ``k``,
    # check whether cycles [k-4 : k+1] (5 cycles ending at k) form a
    # stable region — small residual_std on a linear fit AND a slope
    # much shallower than the inflection slope. The two-part test
    # rejects the *linear middle of the sigmoid* — five consecutive
    # cycles around the inflection cycle are approximately linear
    # (slope ≈ d1_max/2), so they pass a residual-only test, but
    # they're clearly not baseline. Real baselines have slope ≈ 0 or
    # just photobleach drift, both well below 20% of d1_max.
    threshold = 0.005  # 0.5% of F_range
    d1_max = float(d1[inflection_idx])
    slope_cap = 0.20 * abs(d1_max) if d1_max != 0 else None
    take_off = None
    for k in range(inflection_idx - 1, floor_idx + 3, -1):
        bg_c = cycles[k - 4:k + 1]
        bg_f = fluor_data[k - 4:k + 1]
        coeffs = np.polyfit(bg_c, bg_f, 1)
        resid = bg_f - np.polyval(coeffs, bg_c)
        rn = float(np.std(resid)) / F_range
        if rn > threshold:
            continue
        if slope_cap is not None and abs(coeffs[0]) > slope_cap:
            continue
        take_off = k + 1
        break

    return take_off if take_off is not None else inflection_idx


def prepare_fit_inputs(
    cycles,
    fluor_data,
    *,
    first_fit_cycle: float = 3.0,
    cycles_before_max: int = 15,
    fit_bounds: dict | None = None,
):
    """Run the per-well preprocessing pipeline; return inputs for the optimizer.

    Shared between ``fit_well`` (PCRedux scoring + future single-well API
    callers) and ``run_batch.run_pass1`` (production batch driver). Both
    need the same window selection + bg pre-estimation + bounds logic;
    by routing them through this helper we ensure every fitting path
    benefits from session-level improvements without behavior drift.

    Returns a dict with everything ``MAK2Optimizer.fit`` needs plus the
    intermediate state callers report in their result objects
    (``fit_start_cycle``, ``baseline_end_cycle``).
    """
    cycles = np.asarray(cycles, dtype=float)
    fluor_data = np.asarray(fluor_data, dtype=float)
    floor_idx = int(np.searchsorted(cycles, first_fit_cycle))

    # Baseline-end via take-off detector with estimate_baseline_end fallback.
    take_off_idx = find_take_off_idx(cycles, fluor_data, floor_idx)
    if take_off_idx is None or take_off_idx - floor_idx < 4:
        baseline_end_idx = estimate_baseline_end(
            cycles, fluor_data, first_cycle_idx=floor_idx,
        )
    else:
        baseline_end_idx = take_off_idx

    # Background pre-estimation: 12-cycle window ending at baseline_end.
    # Linearity gate falls back to early-baseline window with iterative
    # shrink when contaminated (sharp early amps, drift-then-bend curves).
    bg_window_size = 12
    bg_pre_start = max(floor_idx, baseline_end_idx - bg_window_size)
    bg_c = cycles[bg_pre_start:baseline_end_idx]
    bg_f = fluor_data[bg_pre_start:baseline_end_idx]
    F_range_full = float(np.max(fluor_data) - np.min(fluor_data))
    if len(bg_c) >= 5 and F_range_full > 0:
        coeffs_tmp = np.polyfit(bg_c, bg_f, 1)
        resid = bg_f - np.polyval(coeffs_tmp, bg_c)
        resid_norm = float(np.std(resid)) / F_range_full
        if resid_norm > 0.01:
            early_end = min(baseline_end_idx, floor_idx + 1 + bg_window_size)
            bg_pre_start = floor_idx + 1
            bg_c = cycles[bg_pre_start:early_end]
            bg_f = fluor_data[bg_pre_start:early_end]
            for _shrink in range(bg_window_size - 4):
                if len(bg_c) < 5:
                    break
                coeffs_chk = np.polyfit(bg_c, bg_f, 1)
                rd_chk = bg_f - np.polyval(coeffs_chk, bg_c)
                if float(np.std(rd_chk)) / F_range_full <= 0.01:
                    break
                bg_c = bg_c[:-1]
                bg_f = bg_f[:-1]
            baseline_end_idx = bg_pre_start + len(bg_c)

    if len(bg_c) >= 2:
        coeffs = np.polyfit(bg_c, bg_f, 1)
        bg_slope_est = float(coeffs[0])
        bg_int_est = float(coeffs[1])
    else:
        bg_slope_est = 0.0
        bg_int_est = (
            float(np.median(fluor_data[floor_idx:baseline_end_idx]))
            if baseline_end_idx > floor_idx else 0.0
        )

    # Fit-window placement.
    fit_start_idx, max_slope_idx = smart_start(
        fluor_data, cycles, floor_idx, cycles_before_max
    )
    fit_start_idx, _, _ = adaptive_window_extension(
        fluor_data, cycles, fit_start_idx, max_slope_idx,
        floor_idx, cycles_before_max, bg_slope_est, bg_int_est,
    )
    cycles_fit = cycles[fit_start_idx:]
    fluor_fit = fluor_data[fit_start_idx:]
    F_range = float(np.max(fluor_fit) - np.min(fluor_fit))

    # Safety net: <70% range means smart-start missed the sigmoid.
    total_range = float(np.max(fluor_data) - np.min(fluor_data))
    if total_range > 0 and F_range < 0.7 * total_range:
        fit_start_idx = floor_idx
        cycles_fit = cycles[fit_start_idx:]
        fluor_fit = fluor_data[fit_start_idx:]
        F_range = float(np.max(fluor_fit) - np.min(fluor_fit))

    # Tight bounds around the window-based background pre-estimate.
    # Intentionally no D0 bound — see comment in fit_well for why.
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

    return {
        'cycles_fit': cycles_fit,
        'fluor_fit': fluor_fit,
        'merged_bounds': merged_bounds,
        'fixed_background_values': {
            'F_bg_slope':     bg_slope_est,
            'F_bg_intercept': bg_int_est,
        },
        'bg_slope_est':   bg_slope_est,
        'bg_int_est':     bg_int_est,
        'fit_start_idx':  fit_start_idx,
        'max_slope_idx':  max_slope_idx,
        'floor_idx':      floor_idx,
        'baseline_end_idx': baseline_end_idx,
        'fit_start_cycle': float(cycles[fit_start_idx]),
        'baseline_end_cycle': float(cycles[min(baseline_end_idx, len(cycles) - 1)]),
        'F_range': F_range,
    }


def fit_well(
    cycles,
    fluor_data,
    *,
    first_fit_cycle: float = 3.0,
    cycles_before_max: int = 15,
    cycles_after_max: int = 8,
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

    prep = prepare_fit_inputs(
        cycles, fluor_data,
        first_fit_cycle=first_fit_cycle,
        cycles_before_max=cycles_before_max,
        fit_bounds=fit_bounds,
    )

    # Bypass for A/B comparison and pre-Stage-0-1-2-3 regression diagnosis.
    # Setting MAK2_DISABLE_TOE=1 reverts fit_well to its previous behaviour:
    # no D0 anchor from the toe, no toe gate, no retry. The result dict
    # still carries the new field names with None values so callers don't
    # need conditional shape handling.
    _toe_disabled = os.environ.get("MAK2_DISABLE_TOE") == "1"

    # Stage 0 + Stage 1 pre-fitting. Cheap; always runs (Stage 1 self-skips
    # on low SNR). Outputs attached to the result dict regardless of whether
    # Stage 2 uses D0_toe to bound the optimizer.
    if _toe_disabled:
        from toe_prefit import Stage0Result  # local import — keep bypass cheap
        s0 = Stage0Result(success=False, reason="disabled via MAK2_DISABLE_TOE")
    else:
        s0 = stage0_ct_from_inflection(
            cycles, inflection_idx=prep['max_slope_idx'],
        )
    if s0.success:
        bl_std = baseline_std_from_prefit(
            cycles, fluor_data,
            prep['bg_int_est'], prep['bg_slope_est'],
            prep['floor_idx'], prep['baseline_end_idx'],
        )
        s1 = stage1_toe_fit(
            cycles, fluor_data,
            bg_int=prep['bg_int_est'],
            bg_slope=prep['bg_slope_est'],
            Ct_stage0=s0.Ct_stage0,
            baseline_std=bl_std,
        )
    else:
        s1 = None

    # Stage 2: if Stage 1 succeeded with sufficient toe-fit quality, tighten
    # the optimizer's D0 search to a narrow window around D0_toe. The MAK2
    # toe-region doubling limit makes D0_toe a physics-justified prior, so
    # bounding ±factor (default 4x) around it eliminates the vast-empty
    # parameter regions that drive most toe-misfit failures while still
    # giving the optimizer room to correct for noise in the toe estimate.
    bounds_for_fit = prep['merged_bounds']
    if s1 is not None and s1.success and s1.toe_fit_r2 is not None \
            and s1.toe_fit_r2 >= TOE_FIT_R2_MIN \
            and 'D0' not in bounds_for_fit:
        D0_lo = s1.D0_toe / TOE_D0_BOUND_FACTOR
        D0_hi = s1.D0_toe * TOE_D0_BOUND_FACTOR
        bounds_for_fit = {**bounds_for_fit, 'D0': (D0_lo, D0_hi)}

    def _run_fit(bounds):
        opt = MAK2Optimizer(MAK2Model())
        params = opt.fit(
            prep['cycles_fit'], prep['fluor_fit'],
            cycles_after_max=cycles_after_max,
            auto_truncate=auto_truncate,
            truncate_cycle=truncate_cycle,
            bounds=bounds,
            fixed_background_values=prep['fixed_background_values'],
            verbose=verbose,
        )
        metrics = opt.calculate_fit_metrics()
        end_cycle = (
            float(opt.cycles_fit[-1])
            if opt.cycles_fit is not None and len(opt.cycles_fit) > 0
            else float(cycles[-1])
        )
        return params, metrics, end_cycle

    try:
        params, metrics, fit_end_cycle = _run_fit(bounds_for_fit)

        # Stage 3: toe residual gate. Only meaningful when we have a Ct_stage0
        # to define the toe window. One retry with widened D0 bounds on the
        # side suggested by the residual sign; if it still fails the gate is
        # advisory (TOE_MISFIT flag), not a hard reject.
        s3 = None
        toe_misfit = False
        if s0 is not None and s0.success:
            s3 = stage3_toe_gate(
                cycles, fluor_data, params, Ct_stage0=s0.Ct_stage0,
            )
            if (s3.evaluated and not s3.passed
                    and s1 is not None and s1.success
                    and s1.toe_fit_r2 is not None
                    and s1.toe_fit_r2 >= TOE_FIT_R2_MIN):
                # Widen D0 on the side that the residual sign implicates.
                factor_sq = TOE_D0_BOUND_FACTOR ** 2
                if s3.sign > 0:
                    # Model below data => D0 likely too low. Drop the floor.
                    D0_lo = s1.D0_toe / factor_sq
                    D0_hi = s1.D0_toe * TOE_D0_BOUND_FACTOR
                else:
                    # Model above data => D0 likely too high. Raise the cap.
                    D0_lo = s1.D0_toe / TOE_D0_BOUND_FACTOR
                    D0_hi = s1.D0_toe * factor_sq
                retry_bounds = {**bounds_for_fit, 'D0': (D0_lo, D0_hi)}
                try:
                    p2, m2, fe2 = _run_fit(retry_bounds)
                    s3_retry = stage3_toe_gate(
                        cycles, fluor_data, p2, Ct_stage0=s0.Ct_stage0,
                    )
                    # Accept the retry if it passed the toe gate OR (gate
                    # still fails but the global fit didn't regress).
                    if s3_retry.evaluated and (
                        s3_retry.passed or m2['r_squared'] >= metrics['r_squared']
                    ):
                        params, metrics, fit_end_cycle = p2, m2, fe2
                        s3 = s3_retry
                except Exception:
                    pass  # Keep the first-pass result if retry blows up.
            if s3.evaluated and not s3.passed:
                toe_misfit = True

        return {
            'Sample': 'x',
            'R2': metrics['r_squared'],
            'D0': params['D0'],
            'k':  params['k'],
            'P0': params['P0'],
            'F_bg_intercept': params['F_bg_intercept'],
            'F_bg_slope':     params['F_bg_slope'],
            'fit_start_cycle': prep['fit_start_cycle'],
            'fit_end_cycle':   fit_end_cycle,
            'fluor_data':      fluor_data,
            'error':           None,
            'Success':         '✓',
            **_toe_fields(s0, s1, s3, toe_misfit),
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
            **_toe_fields(s0, s1, None, False),
        }


def _toe_fields(s0, s1, s3, toe_misfit: bool) -> dict:
    """Pack Stage 0/1/3 diagnostics into the per-well result dict."""
    out = {
        'Ct_stage0':   s0.Ct_stage0 if s0 and s0.success else None,
        'D0_toe':      None,
        'toe_fit_r2':  None,
        'toe_snr':     None,
        'toe_skipped_reason': None,
        'toe_mean_residual':  None,
        'toe_rel_residual':   None,
        'TOE_MISFIT':         bool(toe_misfit),
    }
    if s1 is None:
        out['toe_skipped_reason'] = 'stage0 failed'
    elif s1.success:
        out['D0_toe']     = s1.D0_toe
        out['toe_fit_r2'] = s1.toe_fit_r2
        out['toe_snr']    = s1.snr
    else:
        out['toe_skipped_reason'] = s1.reason
        out['toe_snr']    = s1.snr
    if s3 is not None and s3.evaluated:
        out['toe_mean_residual'] = s3.mean_residual
        out['toe_rel_residual']  = s3.rel_residual
    return out
