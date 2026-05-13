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
    # stable region — small residual_std on a linear fit. The first k
    # (largest, closest to inflection) satisfying that is the last
    # baseline cycle; take-off = k + 1.
    threshold = 0.005  # 0.5% of F_range
    take_off = None
    for k in range(inflection_idx - 1, floor_idx + 3, -1):
        bg_c = cycles[k - 4:k + 1]
        bg_f = fluor_data[k - 4:k + 1]
        coeffs = np.polyfit(bg_c, bg_f, 1)
        resid = bg_f - np.polyval(coeffs, bg_c)
        rn = float(np.std(resid)) / F_range
        if rn <= threshold:
            take_off = k + 1
            break

    return take_off if take_off is not None else inflection_idx


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
    # Locate the take-off cycle (= baseline_end) by backward-from-
    # inflection scan. See ``find_take_off_idx`` docstring for why this
    # is more reliable than ``estimate_baseline_end``'s forward
    # signal-vs-noise rule on sharp early amplifiers, biphasic
    # baselines, and drifting baselines. Fall back to the legacy
    # detector if the take-off search can't find a stable region.
    take_off_idx = find_take_off_idx(cycles, fluor_data, floor_idx)
    if take_off_idx is None or take_off_idx - floor_idx < 4:
        baseline_end_idx = estimate_baseline_end(
            cycles, fluor_data, first_cycle_idx=floor_idx,
        )
    else:
        baseline_end_idx = take_off_idx
    # Curvature-based safety-cap: ``estimate_baseline_end`` can overshoot
    # in two distinct regimes — on sharp early amplifiers (rutledge X1,
    # tiny noise + steep take-off) it lands deep inside the rise; on
    # gradually-drifting baselines (media-1 maro2.*, photobleaching) it
    # may overshoot by a few cycles but most of the window is genuine
    # drift we want to capture.
    #
    # A pure magnitude cap (e.g. min + 5% F_range) can't tell these
    # apart: it collapses the media-1 window to 2–3 cycles because the
    # drift itself crosses 5% of range. So instead: take the proposed
    # 12-cycle window, fit a line, identify cycles whose residual is
    # >3σ from the line. Those are the growth-contaminated cycles —
    # walk the upper endpoint back past them. A clean linear drift
    # leaves no outliers, so the window stays wide.
    bg_window_size = 12
    bg_pre_start = max(floor_idx, baseline_end_idx - bg_window_size)
    bg_c = cycles[bg_pre_start:baseline_end_idx]
    bg_f = fluor_data[bg_pre_start:baseline_end_idx]
    F_range_full = float(np.max(fluor_data) - np.min(fluor_data))
    if len(bg_c) >= 5 and F_range_full > 0:
        # Detect growth contamination by linearity. Genuine baseline
        # drift (photobleaching) is straight — polyfit residuals are
        # small. Growth contamination shows up as a parabolic curve —
        # polyfit residual_std is large relative to F_range.
        coeffs_tmp = np.polyfit(bg_c, bg_f, 1)
        resid = bg_f - np.polyval(coeffs_tmp, bg_c)
        resid_norm = float(np.std(resid)) / F_range_full
        # Threshold: 1% of F_range. Linear drift on media-1 maro2.*
        # comes in at ~0.1%; rutledge X1 / media-1 maro2.66
        # contamination at ~3%.
        if resid_norm > 0.01:
            # Window is contaminated. ``estimate_baseline_end`` has
            # overshot into the take-off. Fall back to the *early*
            # portion of the baseline (cycles floor_idx+1 to
            # floor_idx+13). This skips most of the dye-equilibration
            # noise of cycles 1-2 while staying well inside the truly
            # flat region for both sharp early amplifiers (rutledge X1
            # take-off ~cyc 14) and gradually-drifting late amplifiers
            # (media-1 maro2.66 bend starts ~cyc 30). Wider window than
            # the previous magnitude-cap fallback gave (2-5 cycles),
            # which kept the polyfit noisy.
            early_end = min(baseline_end_idx, floor_idx + 1 + bg_window_size)
            bg_pre_start = floor_idx + 1
            bg_c = cycles[bg_pre_start:early_end]
            bg_f = fluor_data[bg_pre_start:early_end]
            # The early window can ITSELF be contaminated for very
            # early/sharp amplifiers (boggy F1.1, rutledge X1) where
            # take-off begins before cycle 13. Iteratively trim the
            # upper end while the polyfit residual_norm stays elevated.
            for _shrink in range(bg_window_size - 4):
                if len(bg_c) < 5:
                    break
                coeffs_chk = np.polyfit(bg_c, bg_f, 1)
                rd_chk = bg_f - np.polyval(coeffs_chk, bg_c)
                if float(np.std(rd_chk)) / F_range_full <= 0.01:
                    break
                bg_c = bg_c[:-1]
                bg_f = bg_f[:-1]
            # Update baseline_end_idx so downstream code sees the right
            # boundary (used by adaptive_window_extension for the
            # baseline-cycles check).
            baseline_end_idx = bg_pre_start + len(bg_c)
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
