"""Shared helpers for the channel-aware Pass 2 retry.

Includes ``retry_one_well`` — the canonical per-well retry — used by
both ``run_batch.run_pass2`` and the app.py batch loop. Caller is
responsible for iteration, progress UI, and checkpoint state; this
module only handles the actual fit logic.


Pass 1 fits each well in isolation. After Pass 1 finishes, the
distribution of ``k`` / ``P0`` / ``F_bg_*`` across the plate is
informative — wells that landed in bad local minima or had pathological
fits can be rescued by re-fitting with the channel-typical values as
priors. This module owns the two pieces of that logic that are
mathematically identical across every Pass 2 caller (app.py batch and
``run_batch.run_pass2``):

  - ``compute_channel_priors``: per-channel and plate-wide medians of
    the kinetic + background parameters, computed over the subset of
    Pass 1 fits that look reliable.

  - ``identify_retry_candidates``: predicates flag a Pass 1 result for
    a retry. Includes the "hopeless wells" skip step that exempts late
    amplifiers from the R² < 0.85 cutoff.

The per-well retry *fitting* itself is not extracted here — the retry
loop has UI integration (app.py) vs CLI integration (run_batch)
differences that don't simplify cleanly. Both call sites still do the
fit themselves, but using the prior dicts and the index list that
these two functions produce.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# Threshold below which we won't retry — pure noise/drift fits don't
# benefit from channel priors and just consume time. Late amplifiers
# are exempted because their priors-based retries do recover them.
HOPELESS_R2 = 0.85


def channel_of(name: str) -> str:
    """Extract the channel prefix from a composite sample name.

    Sample names from multi-channel plates are ``"{channel}::{well}"`` or
    ``"{channel}_{well}"``. Plain well names get ``'default'``. Canonical
    implementation imported by both app.py and run_batch.
    """
    if '::' in name:
        return name.split('::')[0]
    if '_' in name:
        return name.split('_')[0]
    return 'default'


def well_pos_of(name: str) -> str:
    """Extract the bare well position (``A1``, ``H12``…) from a composite name.

    Inverse of ``channel_of``: returns the well part of
    ``"{channel}::{well}"`` or ``"{channel}_{well}"``. Falls back to
    the input itself for plain names.
    """
    if '::' in name:
        parts = name.split('::')
        return parts[1] if len(parts) > 1 else name
    if '_' in name:
        parts = name.split('_')
        return '_'.join(parts[1:]) if len(parts) > 1 else name
    return name


_channel_of = channel_of  # internal alias used by compute_channel_priors below


# Acceptance thresholds for the retry's "is this fit good enough" decision.
PASS2_R2_TARGET = 0.999
PASS2_R2_LATE_AMP = 0.995
PASS2_R2_NOISY = 0.995
PASS2_TIMEOUT_LATE_S = 10.0
PASS2_TIMEOUT_NORMAL_S = 30.0


def compute_channel_priors(results_list: list[dict]) -> tuple[dict, dict]:
    """Return ``(channel_medians, plate_medians)`` from Pass 1 results.

    "Reliable" Pass 1 fits (R² > 0.95, k < 0.5, Success ✓) contribute
    to the per-channel and plate-wide medians of k / P0 / F_bg_intercept
    / F_bg_slope. Channels need ≥ 2 reliable fits to enter
    ``channel_medians``; everything else falls back to ``plate_medians``
    when a retry needs a prior.

    The plate-wide fallback uses sensible defaults (k=0.15, P0=1e5,
    F_bg_intercept=1e5, F_bg_slope=0) when no reliable fits exist at
    all (e.g. early in a debug run).
    """
    ch_k: dict[str, list[float]] = {}
    ch_P0: dict[str, list[float]] = {}
    ch_Fbg: dict[str, list[float]] = {}
    ch_slope: dict[str, list[float]] = {}

    for r in results_list:
        if (r.get('k') is not None and r.get('R2') is not None
                and r['R2'] > 0.95 and r['k'] < 0.5
                and str(r.get('Success', '')).startswith('✓')):
            ch = _channel_of(r['Sample'])
            ch_k.setdefault(ch, []).append(r['k'])
            ch_P0.setdefault(ch, []).append(r['P0'])
            ch_Fbg.setdefault(ch, []).append(r['F_bg_intercept'])
            if r.get('F_bg_slope') is not None:
                ch_slope.setdefault(ch, []).append(r['F_bg_slope'])

    channel_medians: dict[str, dict] = {}
    for ch, ks in ch_k.items():
        if len(ks) >= 2:
            channel_medians[ch] = {
                'k':              float(np.median(ks)),
                'P0':             float(np.median(ch_P0[ch])),
                'F_bg_intercept': float(np.median(ch_Fbg[ch])),
                'F_bg_slope':     float(np.median(ch_slope.get(ch, [0.0]))),
                'n':              len(ks),
            }

    all_k    = [v for vs in ch_k.values()    for v in vs]
    all_P0   = [v for vs in ch_P0.values()   for v in vs]
    all_Fbg  = [v for vs in ch_Fbg.values()  for v in vs]
    all_Sl   = [v for vs in ch_slope.values() for v in vs]
    plate_medians = {
        'k':              float(np.median(all_k))   if all_k   else 0.15,
        'P0':             float(np.median(all_P0))  if all_P0  else 1e5,
        'F_bg_intercept': float(np.median(all_Fbg)) if all_Fbg else 1e5,
        'F_bg_slope':     float(np.median(all_Sl))  if all_Sl  else 0.0,
    }
    return channel_medians, plate_medians


def identify_retry_candidates(
    results_list: list[dict],
    cycles: np.ndarray,
    cycles_after_max: int,
) -> tuple[list[int], int]:
    """Return sorted indices of Pass 1 wells worth re-fitting in Pass 2.

    Five inclusion predicates (any one triggers a retry):

      (a) High SSR relative to fluorescence range, combined with
          R² < 0.999.
      (b) Optimisation failed entirely (k is None).
      (c) Degenerate k > 0.5 (unphysical for qPCR), combined with
          R² < 0.999.
      (d) R² below the 0.999 target.
      (e) "Tail overshoot": the model is consistently above the data
          in the last 3 cycles of the fit window, which happens when
          smart truncation cuts the window before the plateau and the
          optimizer ends up unconstrained on the late-cycle k.

    Wells with R² < 0.85 are exempted from retry (pure noise/drift
    can't be rescued) *unless* they are late amplifiers, defined as
    fits whose end cycle is within ``cycles_after_max`` of the last
    cycle — those benefit from extended-baseline retries.
    """
    # Local import avoids a circular import: pass2_helpers used by app.py
    # and run_batch.py, both of which sit above mak2_model in dep order.
    from mak2_model import MAK2Model

    last_cyc = float(cycles[-1]) if len(cycles) else 0.0
    late_margin = min(max(1, cycles_after_max), 5)
    retry: set[int] = set()

    for i, r in enumerate(results_list):
        fd = r.get('fluor_data')
        r2 = r.get('R2')

        # (a) high SSR with sub-target R²
        if (r.get('SSR') is not None and fd is not None
                and (r2 is None or r2 < 0.999)):
            F_rng = float(np.max(fd) - np.min(fd))
            if r['SSR'] > 0.01 * F_rng ** 2:
                retry.add(i)

        # (b) total optimisation failure
        if r.get('k') is None:
            retry.add(i)

        # (c) degenerate k
        if (r.get('k') is not None and r['k'] > 0.5
                and (r2 is None or r2 < 0.999)):
            retry.add(i)

        # (d) below target R²
        if r2 is not None and r2 < 0.999:
            retry.add(i)

        # (e) tail overshoot — model below data at end of fit window.
        if (r2 is not None and r2 < 0.999
                and fd is not None and r.get('error') is None):
            try:
                fe = r.get('fit_end_cycle')
                fs = r.get('fit_start_cycle')
                if (fe is not None and fs is not None
                        and not (isinstance(fe, float) and np.isnan(fe))):
                    c_arr = (cycles[:len(fd)] if len(cycles) >= len(fd)
                             else np.arange(1, len(fd) + 1, dtype=float))
                    win_mask = (c_arr >= fs) & (c_arr <= fe)
                    fd_win = fd[win_mask]
                    if r.get('D0') is not None and not np.isnan(r['D0']):
                        f_pred = MAK2Model().simulate_to_cycle(
                            D0=r['D0'], k=r['k'], P0=r['P0'],
                            cycles=c_arr[win_mask],
                            F_bg_intercept=r['F_bg_intercept'],
                            F_bg_slope=r['F_bg_slope'],
                        )
                        resid = fd_win - f_pred
                        last3 = float(np.mean(resid[-3:])) if len(resid) >= 3 else 0.0
                        F_rng = float(np.max(fd) - np.min(fd))
                        if F_rng > 0 and last3 < -0.03 * F_rng:
                            retry.add(i)
            except Exception:
                pass

    # Hopeless-well exemption: keep late amplifiers, drop everything
    # else below R² 0.85. Skip count returned so callers can surface
    # "skipped N wells" feedback in their UI.
    skipped = 0
    for i in list(retry):
        r2_i = results_list[i].get('R2')
        if r2_i is not None and r2_i < HOPELESS_R2:
            fe_i = results_list[i].get('fit_end_cycle')
            is_late = (fe_i is not None and fe_i >= last_cyc - late_margin)
            if not is_late:
                retry.discard(i)
                skipped += 1

    return sorted(retry), skipped


def retry_one_well(
    pass1_result: dict,
    priors: dict,
    cycles: np.ndarray,
    *,
    metadata: dict | None = None,
    rox: np.ndarray | None = None,
    channel_threshold: float | None = None,
    first_fit_cycle: float = 3.0,
    cycles_before_max: int = 15,
    cycles_after_max: int = 8,
    auto_truncate: bool = True,
    truncate_cycle: float | None = None,
) -> dict:
    """Run the channel-aware Pass 2 retry on one well.

    Tries several variants in order of cheapness, keeping the highest-R²
    fit across all attempts (including the original Pass 1):

      1. **Informed-priors retry**: re-fit with bounds anchored at the
         channel-median k / P0 / F_bg and the per-well background
         pre-estimate (or channel-median if absent).
      2. **Late-amp analytical pre-estimation** (only when Pass 1 was a
         late amplifier that failed): seed D0 and k from a log-linear
         exponential fit to the rising portion.
      3. **Window variations** (when target R² still not met): 8 different
         (cycles_before_max, cycles_after_max) combinations to bracket
         the inflection differently.
      4. **k relaxation** (last-resort): drop the lower k bound to allow
         very slow growth, retry once.

    Returns the result dict for the well — either an updated dict if a
    retry beat Pass 1's R², or the original dict with an updated
    ``Success`` flag if nothing improved. The caller does
    ``results_list[idx] = retry_one_well(results_list[idx], ...)``.
    """
    # Local imports avoid eager pull-in for callers that never reach Pass 2.
    from mak2_model import MAK2Model
    from optimizer import MAK2Optimizer
    from data_processing import estimate_baseline_end
    from optimizer import estimate_MAK2_params_from_exponential
    from fit_well import _compute_ct  # canonical Ct helper
    from run_batch import smart_start, adaptive_window_extension

    import time

    result = pass1_result
    sample_name = result['Sample']
    fluor_data = result.get('fluor_data')
    bg_slope_est_r = result.get('bg_slope_est')
    bg_intercept_est_r = result.get('bg_intercept_est')

    if fluor_data is None:
        # Nothing to retry; just mark the Success and return as-is.
        out = dict(result)
        out['Success'] = '⚠️ no data for retry'
        return out

    last_cyc = float(cycles[-1])
    late_margin = min(max(1, cycles_after_max), 5)
    pass1_late = (
        result.get('fit_end_cycle') is not None
        and result['fit_end_cycle'] >= last_cyc - late_margin
    )
    retry_timeout = PASS2_TIMEOUT_LATE_S if pass1_late else PASS2_TIMEOUT_NORMAL_S
    retry_t0 = time.perf_counter()

    pk = priors['k']
    pP0 = priors['P0']
    pFbg = priors['F_bg_intercept']
    pSlope = priors['F_bg_slope']

    try:
        F_max = float(np.max(fluor_data))
        F_range_r = float(np.max(fluor_data) - np.min(fluor_data))

        # Background bounds: prefer per-well estimate from Pass 1; fall back
        # to channel-median priors with looser tolerances.
        if bg_slope_est_r is not None and bg_intercept_est_r is not None:
            slope_margin = max(abs(bg_slope_est_r) * 0.40, F_range_r * 0.01)
            int_margin = max(abs(bg_intercept_est_r) * 0.15, F_max * 0.02)
            bg_slope_bounds = (bg_slope_est_r - slope_margin,
                               bg_slope_est_r + slope_margin)
            bg_int_bounds = (max(0.0, bg_intercept_est_r - int_margin),
                             bg_intercept_est_r + int_margin)
        else:
            slope_delta = max(abs(pSlope) * 3.0, F_range_r * 0.05)
            bg_slope_bounds = (pSlope - slope_delta, pSlope + slope_delta)
            fbg_lo = max(0.0, pFbg * 0.30)
            fbg_hi = pFbg * 3.0 if pFbg > 0 else F_max
            bg_int_bounds = (fbg_lo, fbg_hi)

        informed_bounds = {
            'k':              (max(0.01, pk * 0.20),
                               min(1.0, max(0.5, pk * 5.0))),
            'P0':             (max(pP0 * 0.05, F_range_r * 0.01),
                               max(pP0 * 7.0, F_range_r * 2.0)),
            'D0':             (1e-15, F_range_r * 10),
            'F_bg_intercept': bg_int_bounds,
            'F_bg_slope':     bg_slope_bounds,
        }

        # Window placement: same smart_start + adaptive extension as Pass 1.
        r_floor = int(np.searchsorted(cycles, first_fit_cycle))

        r_meta_bl_end_cycle = None
        if metadata:
            v = metadata.get('Baseline End')
            if v is not None:
                try:
                    r_meta_bl_end_cycle = float(v)
                except (ValueError, TypeError):
                    pass

        # Algorithmic baseline-end on raw (or ROX-normalized) fluorescence.
        if rox is not None and len(rox) == len(fluor_data):
            fluor_for_bl = fluor_data / np.maximum(rox, 1e-10)
        else:
            fluor_for_bl = fluor_data
        r_est_bl_end_idx = estimate_baseline_end(
            cycles, fluor_for_bl, first_cycle_idx=r_floor,
        )
        r_est_bl_end_cycle = float(cycles[min(r_est_bl_end_idx, len(cycles) - 1)])

        retry_start_idx, r_max_slope_idx = smart_start(
            fluor_data, cycles, r_floor, cycles_before_max,
        )

        # Background for retry window
        r_bg_pre_start = max(r_floor, retry_start_idx - 8)
        r_bg_c = cycles[r_bg_pre_start:retry_start_idx]
        r_bg_f = fluor_data[r_bg_pre_start:retry_start_idx]
        if len(r_bg_c) >= 2:
            r_bg_coeffs = np.polyfit(r_bg_c, r_bg_f, 1)
            r_bg_slope_win = float(r_bg_coeffs[0])
            r_bg_int_win = float(r_bg_coeffs[1])
        else:
            r_bg_slope_win = 0.0
            r_bg_int_win = (float(fluor_data[retry_start_idx])
                            if retry_start_idx < len(fluor_data) else 0.0)

        retry_start_idx, r_bg_slope_win, r_bg_int_win = adaptive_window_extension(
            fluor_data, cycles, retry_start_idx, r_max_slope_idx,
            r_floor, cycles_before_max, r_bg_slope_win, r_bg_int_win,
        )

        cycles_retry = cycles[retry_start_idx:]
        fluor_retry = fluor_data[retry_start_idx:]

        r_sm = max(abs(r_bg_slope_win) * 0.40, F_range_r * 0.002)
        r_s_min, r_s_max = r_bg_slope_win - r_sm, r_bg_slope_win + r_sm
        r_int_delta = max(abs(r_bg_int_win) * 0.005, F_range_r * 0.03)
        r_int_lo, r_int_hi = r_bg_int_win - r_int_delta, r_bg_int_win + r_int_delta
        informed_bounds['D0'] = (1e-8, max(F_range_r, 1.0))
        informed_bounds['F_bg_slope'] = (r_s_min, r_s_max)
        informed_bounds['F_bg_intercept'] = (r_int_lo, r_int_hi)

        # Late-amp enhancement: when Pass 1 was a late amplifier that
        # didn't fit, derive D0/k seeds analytically from a log-linear
        # fit to the exponential rise.
        pass1_r2 = result.get('R2')
        pass1_failed = (
            pass1_r2 is None
            or (isinstance(pass1_r2, float)
                and (np.isnan(pass1_r2) or pass1_r2 < 0.90))
        )
        if pass1_late and pass1_failed:
            try:
                _, la_bounds = estimate_MAK2_params_from_exponential(
                    cycles_retry, fluor_retry,
                    P0_assumed=pP0 if pP0 > 0 else 1.0,
                    verbose=False,
                )
                if 'D0' in la_bounds:
                    informed_bounds['D0'] = la_bounds['D0']
                if 'k' in la_bounds:
                    la_k_lo = max(0.01, la_bounds['k'][0] * 0.5)
                    la_k_hi = min(1.2, la_bounds['k'][1] * 2.0)
                    informed_bounds['k'] = (la_k_lo, la_k_hi)
            except Exception:
                pass

        # Initial retry fit
        retry_cam = cycles_after_max + 3
        opt_retry = MAK2Optimizer(MAK2Model())
        params_retry = opt_retry.fit(
            cycles_retry, fluor_retry,
            cycles_after_max=retry_cam,
            auto_truncate=auto_truncate,
            truncate_cycle=truncate_cycle,
            bounds=informed_bounds,
            fixed_background_values={
                'F_bg_slope': r_bg_slope_win,
                'F_bg_intercept': r_bg_int_win,
            },
            verbose=False,
        )
        fit_end_cycle_r = (
            float(opt_retry.cycles_fit[-1])
            if opt_retry.cycles_fit is not None and len(opt_retry.cycles_fit) > 0
            else last_cyc
        )
        metrics_retry = opt_retry.calculate_fit_metrics()
        retry_is_late = (fit_end_cycle_r >= last_cyc - late_margin)

        # Compute Ct on the initial retry result. Subsequent variations
        # share these values (they're cheap to recompute but consistent
        # with the existing behavior).
        ct_retry, ct_bl_mean, ct_bl_slope, ct_bl_int, ct_rox_mean_r = _compute_ct(
            cycles, fluor_data, params_retry, metadata, rox, channel_threshold,
            cycles_after_max=cycles_after_max,
        )

        # Track best across attempts. Initialize with Pass 1 baseline.
        original_r2 = result.get('R2')
        retry_r2 = metrics_retry['r_squared']
        best_r2 = original_r2 if original_r2 is not None else -999.0
        best_result = None
        retry_better = (
            original_r2 is None
            or (retry_r2 is not None and retry_r2 > original_r2)
        )
        if retry_better:
            best_r2 = retry_r2 if retry_r2 is not None else best_r2
            best_result = {
                'params': params_retry, 'metrics': metrics_retry,
                'start_idx': retry_start_idx, 'fit_end': fit_end_cycle_r,
                'ct': ct_retry, 'ct_bl_mean': ct_bl_mean,
                'ct_bl_slope': ct_bl_slope, 'ct_bl_int': ct_bl_int,
                'ct_rox_mean': ct_rox_mean_r,
            }

        retry_stage = 'initial-retry'
        retry_attempts = 1
        retry_timed_out = False

        # Window variations: 8 (before, after) combinations.
        if best_r2 < PASS2_R2_TARGET and time.perf_counter() - retry_t0 < retry_timeout:
            retry_stage = 'window-variations'
            win_variations = [
                (cycles_before_max,     cycles_after_max),
                (cycles_before_max,     max(3, cycles_after_max - 1)),
                (cycles_before_max + 4, cycles_after_max),
                (cycles_before_max + 8, cycles_after_max),
                (cycles_before_max,     cycles_after_max + 3),
                (cycles_before_max - 2, cycles_after_max + 3),
                (cycles_before_max + 4, cycles_after_max + 3),
                (cycles_before_max - 4, cycles_after_max),
            ]
            for wv_before, wv_cam in win_variations:
                wv_before = max(3, wv_before)
                wv_start = max(r_floor, r_max_slope_idx - wv_before)
                wv_c = cycles[wv_start:]
                wv_f = fluor_data[wv_start:]

                wv_bg_pre = max(r_floor, wv_start - 6)
                wv_bg_post = min(len(cycles), wv_start + 2)
                wv_bg_c = cycles[wv_bg_pre:wv_bg_post]
                wv_bg_f = fluor_data[wv_bg_pre:wv_bg_post]
                if len(wv_bg_c) >= 2:
                    wv_coeffs = np.polyfit(wv_bg_c, wv_bg_f, 1)
                    wv_slope = float(wv_coeffs[0])
                    wv_int = float(wv_coeffs[1])
                else:
                    wv_slope = 0.0
                    wv_int = float(wv_f[0]) if len(wv_f) else 0.0

                wv_sm = max(abs(wv_slope) * 0.40, F_range_r * 0.002)
                wv_id = max(abs(wv_int) * 0.005, F_range_r * 0.03)
                wv_bounds = dict(informed_bounds)
                wv_bounds['D0'] = (1e-8, max(F_range_r, 1.0))
                wv_bounds['F_bg_slope'] = (wv_slope - wv_sm, wv_slope + wv_sm)
                wv_bounds['F_bg_intercept'] = (wv_int - wv_id, wv_int + wv_id)

                try:
                    wv_opt = MAK2Optimizer(MAK2Model())
                    wv_params = wv_opt.fit(
                        wv_c, wv_f,
                        cycles_after_max=wv_cam,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=wv_bounds,
                        fixed_background_values={
                            'F_bg_slope': wv_slope, 'F_bg_intercept': wv_int,
                        },
                        verbose=False,
                    )
                    wv_metrics = wv_opt.calculate_fit_metrics()
                    wv_r2 = wv_metrics['r_squared']
                    if wv_r2 is not None and wv_r2 > best_r2:
                        best_r2 = wv_r2
                        wv_fe = (
                            float(wv_opt.cycles_fit[-1])
                            if wv_opt.cycles_fit is not None
                            and len(wv_opt.cycles_fit) > 0
                            else last_cyc
                        )
                        best_result = {
                            'params': wv_params, 'metrics': wv_metrics,
                            'start_idx': wv_start, 'fit_end': wv_fe,
                            'ct': ct_retry, 'ct_bl_mean': ct_bl_mean,
                            'ct_bl_slope': ct_bl_slope, 'ct_bl_int': ct_bl_int,
                            'ct_rox_mean': ct_rox_mean_r,
                        }
                        if best_r2 >= PASS2_R2_TARGET:
                            break
                except Exception:
                    pass
                retry_attempts += 1
                if time.perf_counter() - retry_t0 >= retry_timeout:
                    retry_timed_out = True
                    break

        # k relaxation: drop the lower-bound floor in case Pass 1's
        # converged k is right at the prior-derived lower edge.
        if (best_r2 < PASS2_R2_TARGET and not retry_timed_out
                and time.perf_counter() - retry_t0 < retry_timeout):
            retry_stage = 'k-relaxation'
            retry_attempts += 1
            cur_k = (best_result['params']['k'] if best_result
                     else params_retry.get('k'))
            k_lo = informed_bounds['k'][0]
            if cur_k is not None and cur_k < k_lo * 1.5:
                relax_bounds = dict(informed_bounds)
                relax_bounds['k'] = (0.001, relax_bounds['k'][1])
                relax_bounds['D0'] = (1e-15, max(F_range_r * 10, 1.0))
                relax_bounds['F_bg_slope'] = (r_s_min, r_s_max)
                relax_bounds['F_bg_intercept'] = (r_int_lo, r_int_hi)
                try:
                    rk_opt = MAK2Optimizer(MAK2Model())
                    rk_params = rk_opt.fit(
                        cycles_retry, fluor_retry,
                        cycles_after_max=retry_cam,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=relax_bounds,
                        fixed_background_values={
                            'F_bg_slope': r_bg_slope_win,
                            'F_bg_intercept': r_bg_int_win,
                        },
                        verbose=False,
                    )
                    rk_metrics = rk_opt.calculate_fit_metrics()
                    rk_r2 = rk_metrics['r_squared']
                    if rk_r2 is not None and rk_r2 > best_r2:
                        best_r2 = rk_r2
                        rk_fe = (
                            float(rk_opt.cycles_fit[-1])
                            if rk_opt.cycles_fit is not None
                            and len(rk_opt.cycles_fit) > 0
                            else last_cyc
                        )
                        best_result = {
                            'params': rk_params, 'metrics': rk_metrics,
                            'start_idx': retry_start_idx, 'fit_end': rk_fe,
                            'ct': ct_retry, 'ct_bl_mean': ct_bl_mean,
                            'ct_bl_slope': ct_bl_slope, 'ct_bl_int': ct_bl_int,
                            'ct_rox_mean': ct_rox_mean_r,
                        }
                except Exception:
                    pass

        # Decide acceptance.
        if best_result is not None and (original_r2 is None or best_r2 > original_r2):
            # Retry won. Build the new result dict.
            br = best_result
            success = _retry_success_label(
                best_r2, retry_is_late, retry_timed_out, retry_stage,
            )
            return {
                'Sample':                sample_name,
                'Ct':                    br['ct'],
                'Ct_baseline_mean':      br['ct_bl_mean'],
                'Ct_baseline_slope':     br['ct_bl_slope'],
                'Ct_baseline_intercept': br['ct_bl_int'],
                'D0':                    br['params']['D0'],
                'k':                     br['params']['k'],
                'P0':                    br['params']['P0'],
                'F_bg_intercept':        br['params']['F_bg_intercept'],
                'F_bg_slope':            br['params']['F_bg_slope'],
                'R2':                    br['metrics']['r_squared'],
                'RMSE':                  br['metrics']['rmse'],
                'NRMSE':                 br['metrics']['nrmse'] * 100,
                'SSR':                   br['metrics']['ssr'],
                'Tier':                  result.get('Tier'),
                'Instrument':            result.get('Instrument', ''),
                'Success':               success,
                'retry_stage':           retry_stage,
                'retry_attempts':        retry_attempts,
                'retry_elapsed_s':       round(time.perf_counter() - retry_t0, 1),
                'retry_timed_out':       retry_timed_out,
                'FixedBG':    '', 'Fallback': '', 'FallbackOK': '',
                'bg_slope_est':     bg_slope_est_r,
                'bg_intercept_est': bg_intercept_est_r,
                'bl_end_meta':      r_meta_bl_end_cycle,
                'bl_end_est':       r_est_bl_end_cycle,
                'fit_start_cycle':  float(cycles[br['start_idx']]),
                'fit_end_cycle':    br['fit_end'],
                'ct_rox_mean':      br['ct_rox_mean'],
                'fluor_data':       fluor_data,
            }

        # Retry didn't beat Pass 1. Keep the original but update Success.
        out = dict(result)
        orig_r2 = result.get('R2')
        orig_fe = result.get('fit_end_cycle')
        orig_late = (
            orig_fe is not None
            and not (isinstance(orig_fe, float) and np.isnan(orig_fe))
            and float(orig_fe) >= last_cyc - late_margin
        )
        orig_r2_thr = PASS2_R2_LATE_AMP if orig_late else PASS2_R2_TARGET
        if orig_r2 is not None and orig_r2 >= orig_r2_thr:
            out['Success'] = '✓'
        else:
            orig_k = result.get('k')
            if orig_k is not None and orig_k > 0.5:
                out['Success'] = '⚠️ Degenerate k'
            elif orig_r2 is not None and orig_r2 >= PASS2_R2_NOISY:
                out['Success'] = '✓ (noisy data)'
            else:
                out['Success'] = '⚠️ R² below target'
        return out

    except Exception as e:
        out = dict(result)
        if result.get('k') is None:
            out['Success'] = f'✗ Error: {str(e)[:30]}'
        else:
            out['Success'] = '⚠️ Retry failed'
        return out


def _retry_success_label(
    best_r2: float, retry_is_late: bool, retry_timed_out: bool, retry_stage: str,
) -> str:
    """Pick the Success flag string used in run_batch.run_pass2 today."""
    if best_r2 >= PASS2_R2_TARGET:
        return '✓ (window-retry)'
    if retry_is_late and best_r2 >= PASS2_R2_LATE_AMP:
        return '✓ (late-amp)'
    if retry_timed_out:
        return '⚠️ timeout@' + retry_stage
    if best_r2 >= PASS2_R2_NOISY:
        return '✓ (noisy data)'
    return '⚠️ R² below target'
