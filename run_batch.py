#!/Users/boggy/anaconda3/envs/qpcr_mak2/bin/python
"""
MAK2+ Offline Batch Fitting Script
===================================
Processes multiple qPCR plates overnight and produces Excel result files
that can be uploaded to the MAK2+ app via "Load Previous Results" for
visualization.

Usage:
    caffeinate -s python run_batch.py

Output:
    Results/<PlateX>_MAK2Plus_Results.xlsx  (one per plate)
"""

import sys
import os
import time
import io
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

# ── Ensure the MAK2+ modules are importable ���─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mak2_model import MAK2Model, estimate_MAK2_params_from_exponential
from optimizer import MAK2Optimizer
from data_processing import detect_no_signal_samples, estimate_baseline_end
from qpcr_data_converter import QPCRDataConverter, load_abi_results_csv
from replicate_analysis import calculate_replicate_stats, parse_sample_groups, compare_precision
from calibration import build_standard_curve, build_ct_standard_curve, apply_calibration, apply_ct_calibration

# ── Configuration ───────��────────────────────────────��────────────────────────
DATA_DIR = Path("/Users/boggy/Desktop/Desktop031424/Personal")
OUTPUT_DIR = DATA_DIR / "Results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Plates to process: (multicomponent_csv, metadata_csv, output_name)
PLATES = []
for letter in "ABCDEFGHI":
    mc_file = DATA_DIR / f"Plate_{letter}_Multicomponent.csv"
    meta_file = DATA_DIR / f"Plate_{letter}_data.csv"
    if mc_file.exists() and meta_file.exists():
        PLATES.append((mc_file, meta_file, f"Plate{letter}"))

# Default settings (matching app sidebar defaults)
FIRST_FIT_CYCLE = 3
CYCLES_BEFORE_MAX = 10
CYCLES_AFTER_MAX = 4
AUTO_TRUNCATE = True
TRUNCATE_CYCLE = None
CUSTOM_BOUNDS_DICT = None

# Replicate grouping: "sample_name" uses Sample Name from metadata
REPLICATE_GROUPING = "sample_name"


# ── Helper functions ───────────���──────────────────────────────────────────────

def _ch(name):
    """Return the channel prefix of a sample name."""
    if '::' in name: return name.split('::')[0]
    if '_'  in name: return name.split('_')[0]
    return 'default'

def _get_well_pos(name):
    """Extract bare well position from a sample name."""
    if '::' in name: return name.split('::')[1] if len(name.split('::')) > 1 else name
    if '_'  in name: return '_'.join(name.split('_')[1:]) if len(name.split('_')) > 1 else name
    return name

def pre_estimate_background(cycles, fluor, bl_start_idx, bl_end_idx):
    """Linear regression on baseline region for background estimation."""
    bg_c = cycles[bl_start_idx:bl_end_idx + 1]
    bg_f = fluor[bl_start_idx:bl_end_idx + 1]
    if len(bg_c) >= 2:
        coeffs = np.polyfit(bg_c, bg_f, 1)
        return float(coeffs[0]), float(coeffs[1])
    return 0.0, float(fluor[bl_start_idx]) if bl_start_idx < len(fluor) else 0.0


def smart_start(fluor_data, cycles, floor_idx, cycles_before_max):
    """Right-to-left inflection search — identical to app.py logic."""
    full_seg = fluor_data[floor_idx:]
    if len(full_seg) >= 5:
        raw_grad = np.gradient(full_seg)
        kern = np.ones(5) / 5.0
        smooth_g = np.convolve(raw_grad, kern, mode='same')
        grad_range = float(np.max(smooth_g) - np.min(smooth_g))
        grad_floor = grad_range * 0.05
        best_val = smooth_g[-1]
        best_idx = len(smooth_g) - 1
        found_peak = False
        for j in range(len(smooth_g) - 2, -1, -1):
            if smooth_g[j] > best_val:
                best_val = smooth_g[j]
                best_idx = j
            elif best_val > grad_floor and smooth_g[j] < best_val * 0.5:
                found_peak = True
                break
        max_slope_offset = best_idx if found_peak else int(np.argmax(smooth_g))
    elif len(full_seg) >= 2:
        max_slope_offset = int(np.argmax(np.gradient(full_seg)))
    else:
        max_slope_offset = 0
    max_slope_idx = floor_idx + max_slope_offset
    fit_start_idx = max(floor_idx, max_slope_idx - cycles_before_max)
    return fit_start_idx, max_slope_idx


def adaptive_window_extension(fluor_data, cycles, fit_start_idx, max_slope_idx,
                              floor_idx, cycles_before_max, bg_slope, bg_int):
    """Extend fit window backward until ≥3 baseline cycles included."""
    bg_pre_start = max(floor_idx, fit_start_idx - 8)
    bg_f = fluor_data[bg_pre_start:fit_start_idx]
    bl_noise = float(np.std(bg_f)) if len(bg_f) >= 2 else 0.0
    bg_mean_abs = float(np.mean(np.abs(bg_f))) if len(bg_f) >= 2 else 0.0
    skip_ext = (bl_noise < 1e-6 and bg_mean_abs < 1e-6)

    n_baseline = 0
    if not skip_ext:
        for bi in range(fit_start_idx, min(fit_start_idx + 6, len(cycles))):
            bg_level = bg_slope * cycles[bi] + bg_int
            if fluor_data[bi] <= bg_level + 3.0 * bl_noise:
                n_baseline += 1
    else:
        n_baseline = 3

    if n_baseline < 3:
        try_before = cycles_before_max
        while n_baseline < 3 and try_before < max_slope_idx - floor_idx:
            try_before += 2
            try_start = max(floor_idx, max_slope_idx - try_before)
            if try_start == fit_start_idx:
                break
            n_baseline = 0
            for bi in range(try_start, min(try_start + 6, len(cycles))):
                bg_level = bg_slope * cycles[bi] + bg_int
                if fluor_data[bi] <= bg_level + 3.0 * bl_noise:
                    n_baseline += 1
            fit_start_idx = try_start

        # Recompute background with wider window
        bg_pre_start = max(floor_idx, fit_start_idx - 8)
        bg_c = cycles[bg_pre_start:fit_start_idx]
        bg_f = fluor_data[bg_pre_start:fit_start_idx]
        if len(bg_c) >= 2:
            coeffs = np.polyfit(bg_c, bg_f, 1)
            bg_slope = float(coeffs[0])
            bg_int = float(coeffs[1])

    return fit_start_idx, bg_slope, bg_int


def compute_ct(optimizer_obj, cycles, fluor_data, sample_name, sample_metadata,
               rox_by_well, channel_thresholds, global_threshold,
               channel_baseline_means, global_baseline_mean):
    """Compute Ct value — same logic as app.py Pass 1."""
    ct_value = np.nan
    ct_baseline_val = 0.0
    ct_bl_slope = 0.0
    ct_bl_intercept = 0.0
    ct_rox_mean = None

    try:
        ch = _ch(sample_name)
        sample_ch_thresh = channel_thresholds.get(ch, global_threshold)
        baseline_cycles_param = None

        if sample_metadata:
            wm_ct = sample_metadata.get(sample_name, {})
            bl_start = wm_ct.get('Baseline Start')
            bl_end = wm_ct.get('Baseline End')
            try:
                bl_start_i = int(np.searchsorted(cycles, float(bl_start)))
                bl_end_i = int(np.searchsorted(cycles, float(bl_end)))
                if bl_end_i > bl_start_i + 1:
                    baseline_cycles_param = (bl_start_i, bl_end_i)
            except (TypeError, ValueError):
                pass

        well_pos = _get_well_pos(sample_name)
        rox_arr = rox_by_well.get(well_pos, None)
        if rox_arr is not None and len(rox_arr) == len(fluor_data):
            fluor_for_ct = fluor_data / np.maximum(rox_arr, 1e-10)
            ct_rox_mean = float(np.mean(rox_arr))
            ct_threshold = sample_ch_thresh
        else:
            fluor_for_ct = fluor_data
            ct_rox_mean = None
            ct_threshold = None

        orig_cycles = optimizer_obj.cycles_fit
        orig_fluor = optimizer_obj.fluorescence_fit
        optimizer_obj.cycles_fit = cycles
        optimizer_obj.fluorescence_fit = fluor_for_ct
        ct_results = optimizer_obj.calculate_ct(
            method='threshold',
            threshold=ct_threshold,
            baseline_cycles=baseline_cycles_param,
        )
        optimizer_obj.cycles_fit = orig_cycles
        optimizer_obj.fluorescence_fit = orig_fluor
        ct_value = ct_results['ct']
        ct_baseline_val = ct_results.get('baseline_mean', 0.0)
        ct_bl_slope = ct_results.get('baseline_slope', 0.0)
        ct_bl_intercept = ct_results.get('baseline_intercept', 0.0)

        # Respect instrument Undetermined
        if sample_metadata:
            wm = sample_metadata.get(sample_name, {})
            if 'Ct_instrument' in wm:
                inst_ct = wm['Ct_instrument']
                inst_undetermined = (
                    inst_ct is None
                    or (isinstance(inst_ct, float) and np.isnan(inst_ct))
                )
                if inst_undetermined:
                    ct_value = np.nan
    except Exception:
        ct_value = np.nan

    return ct_value, ct_baseline_val, ct_bl_slope, ct_bl_intercept, ct_rox_mean


def load_plate_data(mc_file, meta_file):
    """Load multicomponent CSV and metadata CSV for a plate.
    Returns: (cycles, all_samples, channels, rox_by_well, sample_metadata, abi_results_meta)
    """
    converter = QPCRDataConverter()

    # Load multicomponent data
    cycles, samples, metadata = converter.load_from_file(str(mc_file))
    extra_info = metadata.get('extra_info', {})

    channels = extra_info.get('channels', [])
    passive_ref = extra_info.get('passive_reference', None)
    samples_by_channel = extra_info.get('samples_by_channel', {})

    # Extract fluorescence per channel (same as app.py channel loading)
    all_samples = {}
    rox_by_well = {}

    for channel in channels:
        c_cycles, c_samples, _ = converter.filter_by_channel(
            extra_info, channel, normalize_by_reference=False
        )
        for well_pos, fluor in c_samples.items():
            key = f"{channel}_{well_pos}"
            all_samples[key] = fluor

    # Extract ROX data
    if passive_ref and passive_ref in samples_by_channel:
        for well_pos, rox_fluor in samples_by_channel[passive_ref].items():
            rox_by_well[well_pos] = np.asarray(rox_fluor)

    # Load metadata
    abi_results_meta = load_abi_results_csv(str(meta_file))
    sample_metadata = abi_results_meta.get('sample_metadata', {})

    return cycles, all_samples, channels, rox_by_well, sample_metadata, abi_results_meta


def run_pass1(all_samples_to_fit, cycles, sample_metadata, rox_by_well,
              channel_thresholds, global_threshold, channel_baseline_means,
              global_baseline_mean):
    """Pass 1: Fit all samples — identical to app.py logic."""
    results_list = []
    total = len(all_samples_to_fit)

    for i, (sample_name, fluor_data) in enumerate(all_samples_to_fit.items()):
        print(f"  Pass 1: [{i+1}/{total}] {sample_name}", end="")
        t0 = time.perf_counter()

        try:
            # No-amplification pre-check
            na_sd_window = min(12, len(fluor_data) // 4)
            na_baseline_sd = float(np.std(fluor_data[:na_sd_window])) if na_sd_window >= 3 else 1.0
            na_range = float(np.max(fluor_data) - np.min(fluor_data))
            if na_baseline_sd > 0 and na_range < 5.0 * na_baseline_sd:
                na_baseline_mean = float(np.mean(fluor_data[:na_sd_window])) if na_sd_window >= 3 else 0.0
                na_tail_mean = float(np.mean(fluor_data[-5:])) if len(fluor_data) >= 5 else 0.0
                if not (na_tail_mean > na_baseline_mean + 2.0 * na_baseline_sd):
                    results_list.append({
                        'Sample': sample_name,
                        'D0': np.nan, 'k': np.nan, 'P0': np.nan,
                        'F_bg_intercept': np.nan, 'F_bg_slope': np.nan,
                        'R2': np.nan, 'SSR': np.nan, 'RMSE': np.nan, 'NRMSE': np.nan,
                        'Tier': None, 'Instrument': '',
                        'Ct': np.nan, 'Ct_baseline_mean': np.nan,
                        'Ct_baseline_slope': np.nan, 'Ct_baseline_intercept': np.nan,
                        'fit_start_cycle': np.nan, 'fit_end_cycle': np.nan,
                        'bl_end_meta': np.nan, 'bl_end_est': np.nan,
                        'ct_rox_mean': np.nan,
                        'Success': '', 'FixedBG': '', 'Fallback': '', 'FallbackOK': '',
                        'bg_slope_est': None, 'bg_intercept_est': None,
                        'error': 'No amplification detected',
                        'fluor_data': fluor_data,
                    })
                    print(f"  → no amplification ({time.perf_counter()-t0:.1f}s)")
                    continue

            model_batch = MAK2Model()
            optimizer_batch = MAK2Optimizer(model_batch)

            # Background pre-estimation from metadata
            bg_slope_est = None
            bg_intercept_est = None
            fit_bounds = dict(CUSTOM_BOUNDS_DICT) if CUSTOM_BOUNDS_DICT else {}

            if sample_metadata:
                wm_bg = sample_metadata.get(sample_name, {})
                bl_start = wm_bg.get('Baseline Start')
                bl_end = wm_bg.get('Baseline End')
                try:
                    bl_si = int(np.searchsorted(cycles, float(bl_start)))
                    bl_ei = int(np.searchsorted(cycles, float(bl_end)))
                    if bl_ei > bl_si + 2:
                        bg_slope_est, bg_intercept_est = pre_estimate_background(
                            cycles, fluor_data, bl_si, bl_ei
                        )
                except (TypeError, ValueError):
                    pass

            # Smart start
            floor_idx = int(np.searchsorted(cycles, FIRST_FIT_CYCLE))

            # Baseline end from metadata
            meta_bl_end = None
            meta_bl_end_cycle = None
            if sample_metadata:
                wm = sample_metadata.get(sample_name, {})
                meta_bl_end_val = wm.get('Baseline End', None)
                if meta_bl_end_val is not None:
                    try:
                        meta_bl_end = int(np.searchsorted(cycles, float(meta_bl_end_val)))
                        meta_bl_end_cycle = float(meta_bl_end_val)
                    except (ValueError, TypeError):
                        pass

            # Algorithmic baseline end
            well_pos = _get_well_pos(sample_name)
            rox_arr = rox_by_well.get(well_pos, None)
            if rox_arr is not None and len(rox_arr) == len(fluor_data):
                fluor_for_bl = fluor_data / np.maximum(rox_arr, 1e-10)
            else:
                fluor_for_bl = fluor_data
            est_bl_end_idx = estimate_baseline_end(cycles, fluor_for_bl, first_cycle_idx=floor_idx)
            est_bl_end_cycle = float(cycles[min(est_bl_end_idx, len(cycles) - 1)])

            # Baseline end anchor
            LATE_CEIL = int(len(cycles) * 0.85)
            if meta_bl_end is not None:
                if est_bl_end_idx < LATE_CEIL:
                    baseline_end_idx = max(meta_bl_end, est_bl_end_idx)
                else:
                    baseline_end_idx = meta_bl_end
            else:
                baseline_end_idx = est_bl_end_idx

            fit_start_idx, max_slope_idx = smart_start(
                fluor_data, cycles, floor_idx, CYCLES_BEFORE_MAX
            )

            # Background pre-estimation (window-based)
            bg_pre_start = max(floor_idx, fit_start_idx - 8)
            bg_c = cycles[bg_pre_start:fit_start_idx]
            bg_f = fluor_data[bg_pre_start:fit_start_idx]
            if len(bg_c) >= 2:
                bg_coeffs = np.polyfit(bg_c, bg_f, 1)
                _bg_slope_est = float(bg_coeffs[0])
                _bg_int_est = float(bg_coeffs[1])
            else:
                _bg_slope_est = 0.0
                _bg_int_est = float(fluor_data[fit_start_idx]) if fit_start_idx < len(fluor_data) else 0.0

            # Adaptive window extension
            fit_start_idx, _bg_slope_est, _bg_int_est = adaptive_window_extension(
                fluor_data, cycles, fit_start_idx, max_slope_idx,
                floor_idx, CYCLES_BEFORE_MAX, _bg_slope_est, _bg_int_est
            )

            cycles_fit = cycles[fit_start_idx:]
            fluor_fit = fluor_data[fit_start_idx:]
            F_range = float(np.max(fluor_fit) - np.min(fluor_fit))

            # Safety net: detect if smart-start missed the sigmoid
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
                    bg_coeffs = np.polyfit(bg_c, bg_f, 1)
                    _bg_slope_est = float(bg_coeffs[0])
                    _bg_int_est = float(bg_coeffs[1])

            # Build bounds
            slope_delta = max(abs(_bg_slope_est) * 0.40, F_range * 0.002)
            s_min = _bg_slope_est - slope_delta
            s_max = _bg_slope_est + slope_delta
            int_delta = max(abs(_bg_int_est) * 0.005, F_range * 0.03)
            int_lo = _bg_int_est - int_delta
            int_hi = _bg_int_est + int_delta

            bg_bounds = {
                'D0': (1e-15, max(F_range, 1.0)),
                'F_bg_slope': (s_min, s_max),
                'F_bg_intercept': (int_lo, int_hi),
            }
            non_bg_bounds = {k: v for k, v in (fit_bounds or {}).items()
                            if k not in ('F_bg_slope', 'F_bg_intercept')}
            merged_bounds = {**bg_bounds, **non_bg_bounds}

            # Fit
            params_batch = optimizer_batch.fit(
                cycles_fit, fluor_fit,
                cycles_after_max=CYCLES_AFTER_MAX,
                auto_truncate=AUTO_TRUNCATE,
                truncate_cycle=TRUNCATE_CYCLE,
                bounds=merged_bounds,
                fixed_background_values={
                    'F_bg_slope': _bg_slope_est,
                    'F_bg_intercept': _bg_int_est,
                },
                verbose=False,
            )

            fit_end_cycle = float(optimizer_batch.cycles_fit[-1]) \
                if optimizer_batch.cycles_fit is not None and len(optimizer_batch.cycles_fit) > 0 \
                else float(cycles[-1])

            metrics_batch = optimizer_batch.calculate_fit_metrics()

            # Ct computation
            ct_value, ct_baseline_val, ct_bl_slope, ct_bl_intercept, ct_rox_mean = compute_ct(
                optimizer_batch, cycles, fluor_data, sample_name, sample_metadata,
                rox_by_well, channel_thresholds, global_threshold,
                channel_baseline_means, global_baseline_mean
            )

            # Tier classification
            if params_batch.get('de_used', False):
                tier = 'T3-DE'
            elif params_batch.get('fallback_succeeded', False):
                tier = 'T2-LHS'
            elif params_batch.get('used_fixed_background', False):
                tier = 'T1-Fixed'
            else:
                tier = 'T1-Full'

            # Instrument status
            inst_status = ''
            if sample_metadata:
                wm = sample_metadata.get(sample_name, {})
                if 'Ct_instrument' in wm:
                    flags = []
                    if wm.get('NOAMP'):   flags.append('NOAMP')
                    if wm.get('EXPFAIL'): flags.append('EXPFAIL')
                    if wm.get('HIGHSD'):  flags.append('HIGHSD')
                    inst_ct = wm.get('Ct_instrument')
                    inst_undetermined = (
                        inst_ct is None
                        or (isinstance(inst_ct, float) and np.isnan(inst_ct))
                    )
                    inst_status = ('Undetermined' if inst_undetermined else 'Determined')
                    if flags:
                        inst_status += ' (' + ','.join(flags) + ')'

            r2 = metrics_batch['r_squared']
            results_list.append({
                'Sample': sample_name,
                'Ct': ct_value,
                'Ct_baseline_mean': ct_baseline_val,
                'Ct_baseline_slope': ct_bl_slope,
                'Ct_baseline_intercept': ct_bl_intercept,
                'D0': params_batch['D0'],
                'k': params_batch['k'],
                'P0': params_batch['P0'],
                'F_bg_intercept': params_batch['F_bg_intercept'],
                'F_bg_slope': params_batch['F_bg_slope'],
                'R2': r2,
                'RMSE': metrics_batch['rmse'],
                'NRMSE': metrics_batch['nrmse'] * 100,
                'SSR': metrics_batch['ssr'],
                'Tier': tier,
                'Instrument': inst_status,
                'Success': '✓',
                'FixedBG': '✓' if params_batch.get('used_fixed_background', False) else '',
                'Fallback': '✓' if params_batch.get('fallback_attempted', False) else '',
                'FallbackOK': '✓' if params_batch.get('fallback_succeeded', False) else '',
                'bg_slope_est': bg_slope_est,
                'bg_intercept_est': bg_intercept_est,
                'bl_end_meta': meta_bl_end_cycle,
                'bl_end_est': est_bl_end_cycle,
                'fit_start_cycle': float(cycles[fit_start_idx]),
                'fit_end_cycle': fit_end_cycle,
                'ct_rox_mean': ct_rox_mean,
                'fluor_data': fluor_data,
            })
            elapsed = time.perf_counter() - t0
            print(f"  → R²={r2:.4f} k={params_batch['k']:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            results_list.append({
                'Sample': sample_name,
                'Ct': np.nan,
                'Ct_baseline_mean': 0.0,
                'Ct_baseline_slope': 0.0,
                'Ct_baseline_intercept': 0.0,
                'D0': None, 'k': None, 'P0': None,
                'F_bg_intercept': None, 'F_bg_slope': None,
                'R2': None, 'SSR': None, 'RMSE': None, 'NRMSE': None,
                'Tier': None, 'Instrument': '',
                'Success': f'✗ Error: {str(e)[:30]}',
                'FixedBG': '', 'Fallback': '', 'FallbackOK': '',
                'bg_slope_est': None, 'bg_intercept_est': None,
                'bl_end_meta': None, 'bl_end_est': None,
                'fit_start_cycle': np.nan, 'fit_end_cycle': np.nan,
                'ct_rox_mean': np.nan,
                'error': str(e),
                'fluor_data': fluor_data,
            })
            print(f"  → ERROR: {e}")

    return results_list


def run_pass2(results_list, cycles, sample_metadata, rox_by_well,
              channel_thresholds, global_threshold, channel_baseline_means,
              global_baseline_mean):
    """Pass 2: Channel-aware retry — identical to app.py logic."""
    last_cyc = float(cycles[-1])

    # Step 1: collect per-channel stats
    ch_k = {}; ch_P0 = {}; ch_Fbg = {}; ch_slope = {}
    for r in results_list:
        if (r['k'] is not None and r['R2'] is not None
                and r['R2'] > 0.95 and r['k'] < 0.5
                and str(r['Success']).startswith('✓')):
            ch = _ch(r['Sample'])
            ch_k.setdefault(ch, []).append(r['k'])
            ch_P0.setdefault(ch, []).append(r['P0'])
            ch_Fbg.setdefault(ch, []).append(r['F_bg_intercept'])
            if r['F_bg_slope'] is not None:
                ch_slope.setdefault(ch, []).append(r['F_bg_slope'])

    channel_medians = {}
    for ch in ch_k:
        if len(ch_k[ch]) >= 2:
            channel_medians[ch] = {
                'k': np.median(ch_k[ch]),
                'P0': np.median(ch_P0[ch]),
                'F_bg_intercept': np.median(ch_Fbg[ch]),
                'F_bg_slope': np.median(ch_slope.get(ch, [0.0])),
                'n': len(ch_k[ch]),
            }

    all_k_vals = [v for lst in ch_k.values() for v in lst]
    all_P0_vals = [v for lst in ch_P0.values() for v in lst]
    all_fbg_vals = [v for lst in ch_Fbg.values() for v in lst]
    all_sl_vals = [v for lst in ch_slope.values() for v in lst]
    plate_medians = {
        'k': np.median(all_k_vals) if all_k_vals else 0.15,
        'P0': np.median(all_P0_vals) if all_P0_vals else 1e5,
        'F_bg_intercept': np.median(all_fbg_vals) if all_fbg_vals else 1e5,
        'F_bg_slope': np.median(all_sl_vals) if all_sl_vals else 0.0,
    }

    # Step 2: identify retry candidates
    retry_indices = set()
    for i, r in enumerate(results_list):
        fd = r.get('fluor_data')
        if (r['SSR'] is not None and fd is not None
                and (r['R2'] is None or r['R2'] < 0.999)):
            F_rng = np.max(fd) - np.min(fd)
            if r['SSR'] > 0.01 * F_rng ** 2:
                retry_indices.add(i)
        if r['k'] is None:
            retry_indices.add(i)
        if (r['k'] is not None and r['k'] > 0.5
                and (r['R2'] is None or r['R2'] < 0.999)):
            retry_indices.add(i)
        if r['R2'] is not None and r['R2'] < 0.999:
            retry_indices.add(i)
        # Tail overshoot check
        if (r.get('R2') is not None and r['R2'] < 0.999
                and fd is not None and r.get('error') is None):
            try:
                fe = r.get('fit_end_cycle')
                fs = r.get('fit_start_cycle')
                if fe is not None and fs is not None and not (isinstance(fe, float) and np.isnan(fe)):
                    c_arr = cycles[:len(fd)] if len(cycles) >= len(fd) else np.arange(1, len(fd)+1, dtype=float)
                    win_mask = (c_arr >= fs) & (c_arr <= fe)
                    fd_win = fd[win_mask]
                    if r.get('D0') is not None and not np.isnan(r['D0']):
                        m_tmp = MAK2Model()
                        f_pred_win = m_tmp.simulate_to_cycle(
                            D0=r['D0'], k=r['k'], P0=r['P0'],
                            cycles=c_arr[win_mask],
                            F_bg_intercept=r['F_bg_intercept'],
                            F_bg_slope=r['F_bg_slope'],
                        )
                        resid_win = fd_win - f_pred_win
                        last3_mean = float(np.mean(resid_win[-3:])) if len(resid_win) >= 3 else 0.0
                        F_rng = float(np.max(fd) - np.min(fd))
                        if F_rng > 0 and last3_mean < -0.03 * F_rng:
                            retry_indices.add(i)
            except Exception:
                pass

    # Skip hopeless wells
    for i in list(retry_indices):
        r2_i = results_list[i].get('R2')
        if r2_i is not None and r2_i < 0.85:
            fe_i = results_list[i].get('fit_end_cycle')
            is_late = (fe_i is not None and fe_i >= last_cyc - min(max(1, CYCLES_AFTER_MAX), 5))
            if not is_late:
                retry_indices.discard(i)

    retry_indices = sorted(retry_indices)
    if not retry_indices:
        print("  Pass 2: No samples need retry")
        return results_list

    print(f"  Pass 2: Retrying {len(retry_indices)} samples "
          f"({len(channel_medians)} channel(s) learned)")

    for idx in retry_indices:
        retry_t0 = time.perf_counter()
        result = results_list[idx]
        sample_name = result['Sample']
        fluor_data = result['fluor_data']
        if fluor_data is None:
            continue

        pass1_late = (
            result.get('fit_end_cycle') is not None
            and result['fit_end_cycle'] >= last_cyc - min(max(1, CYCLES_AFTER_MAX), 5)
        )
        RETRY_TIMEOUT = 10.0 if pass1_late else 30.0

        ch = _ch(sample_name)
        priors = channel_medians.get(ch, plate_medians)
        pk = priors['k']
        pP0 = priors['P0']
        pFbg = priors['F_bg_intercept']
        pSlope = priors['F_bg_slope']

        bg_slope_est_r = result.get('bg_slope_est')
        bg_intercept_est_r = result.get('bg_intercept_est')

        print(f"    [{idx}] {sample_name} (R²={result.get('R2', 'None')})", end="")

        try:
            model_retry = MAK2Model()
            optimizer_retry = MAK2Optimizer(model_retry)

            F_max = float(np.max(fluor_data))
            F_range_r = float(np.max(fluor_data) - np.min(fluor_data))

            # Background bounds
            if bg_slope_est_r is not None and bg_intercept_est_r is not None:
                slope_margin = max(abs(bg_slope_est_r) * 0.40, F_range_r * 0.01)
                int_margin = max(abs(bg_intercept_est_r) * 0.15, F_max * 0.02)
                bg_slope_bounds = (bg_slope_est_r - slope_margin, bg_slope_est_r + slope_margin)
                bg_int_bounds = (max(0.0, bg_intercept_est_r - int_margin), bg_intercept_est_r + int_margin)
            else:
                slope_delta = max(abs(pSlope) * 3.0, F_range_r * 0.05)
                bg_slope_bounds = (pSlope - slope_delta, pSlope + slope_delta)
                fbg_lo = max(0.0, pFbg * 0.30)
                fbg_hi = pFbg * 3.0 if pFbg > 0 else F_max
                bg_int_bounds = (fbg_lo, fbg_hi)

            informed_bounds = {
                'k': (max(0.01, pk * 0.20), min(1.0, max(0.5, pk * 5.0))),
                'P0': (max(pP0 * 0.05, F_range_r * 0.01), max(pP0 * 7.0, F_range_r * 2.0)),
                'D0': (1e-15, F_range_r * 10),
                'F_bg_intercept': bg_int_bounds,
                'F_bg_slope': bg_slope_bounds,
            }

            # Smart start for retry
            r_floor = int(np.searchsorted(cycles, FIRST_FIT_CYCLE))

            # Metadata baseline end
            r_meta_bl_end = None
            r_meta_bl_end_cycle = None
            if sample_metadata:
                r_wm = sample_metadata.get(sample_name, {})
                r_meta_bl_end_val = r_wm.get('Baseline End', None)
                if r_meta_bl_end_val is not None:
                    try:
                        r_meta_bl_end = int(np.searchsorted(cycles, float(r_meta_bl_end_val)))
                        r_meta_bl_end_cycle = float(r_meta_bl_end_val)
                    except (ValueError, TypeError):
                        pass

            # Algorithmic baseline end for retry
            well_pos = _get_well_pos(sample_name)
            rox_arr = rox_by_well.get(well_pos, None)
            if rox_arr is not None and len(rox_arr) == len(fluor_data):
                fluor_for_bl = fluor_data / np.maximum(rox_arr, 1e-10)
            else:
                fluor_for_bl = fluor_data
            r_est_bl_end_idx = estimate_baseline_end(cycles, fluor_for_bl, first_cycle_idx=r_floor)
            r_est_bl_end_cycle = float(cycles[min(r_est_bl_end_idx, len(cycles) - 1)])

            r_LATE_CEIL = int(len(cycles) * 0.85)
            if r_meta_bl_end is not None:
                if r_est_bl_end_idx < r_LATE_CEIL:
                    r_baseline_end_idx = max(r_meta_bl_end, r_est_bl_end_idx)
                else:
                    r_baseline_end_idx = r_meta_bl_end
            else:
                r_baseline_end_idx = r_est_bl_end_idx

            retry_start_idx, r_max_slope_idx = smart_start(
                fluor_data, cycles, r_floor, CYCLES_BEFORE_MAX
            )

            # Background for retry
            r_bg_pre_start = max(r_floor, retry_start_idx - 8)
            r_bg_c = cycles[r_bg_pre_start:retry_start_idx]
            r_bg_f = fluor_data[r_bg_pre_start:retry_start_idx]
            if len(r_bg_c) >= 2:
                r_bg_coeffs = np.polyfit(r_bg_c, r_bg_f, 1)
                r_bg_slope_win = float(r_bg_coeffs[0])
                r_bg_int_win = float(r_bg_coeffs[1])
            else:
                r_bg_slope_win = 0.0
                r_bg_int_win = float(fluor_data[retry_start_idx]) if retry_start_idx < len(fluor_data) else 0.0

            # Adaptive extension for retry
            retry_start_idx, r_bg_slope_win, r_bg_int_win = adaptive_window_extension(
                fluor_data, cycles, retry_start_idx, r_max_slope_idx,
                r_floor, CYCLES_BEFORE_MAX, r_bg_slope_win, r_bg_int_win
            )

            cycles_retry = cycles[retry_start_idx:]
            fluor_retry = fluor_data[retry_start_idx:]

            r_sm = max(abs(r_bg_slope_win) * 0.40, F_range_r * 0.002)
            r_s_min = r_bg_slope_win - r_sm
            r_s_max = r_bg_slope_win + r_sm
            r_int_delta = max(abs(r_bg_int_win) * 0.005, F_range_r * 0.03)
            r_int_lo = r_bg_int_win - r_int_delta
            r_int_hi = r_bg_int_win + r_int_delta
            informed_bounds['D0'] = (1e-8, max(F_range_r, 1.0))
            informed_bounds['F_bg_slope'] = (r_s_min, r_s_max)
            informed_bounds['F_bg_intercept'] = (r_int_lo, r_int_hi)

            # Late-amp enhancement
            pass1_r2 = result.get('R2')
            pass1_failed = (pass1_r2 is None
                            or (isinstance(pass1_r2, float) and (np.isnan(pass1_r2) or pass1_r2 < 0.90)))
            if pass1_late and pass1_failed:
                try:
                    la_est, la_bounds = estimate_MAK2_params_from_exponential(
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
            retry_cam = CYCLES_AFTER_MAX + 3
            params_retry = optimizer_retry.fit(
                cycles_retry, fluor_retry,
                cycles_after_max=retry_cam,
                auto_truncate=AUTO_TRUNCATE,
                truncate_cycle=TRUNCATE_CYCLE,
                bounds=informed_bounds,
                fixed_background_values={
                    'F_bg_slope': r_bg_slope_win,
                    'F_bg_intercept': r_bg_int_win,
                },
                verbose=False,
            )
            fit_end_cycle_r = float(optimizer_retry.cycles_fit[-1]) \
                if optimizer_retry.cycles_fit is not None and len(optimizer_retry.cycles_fit) > 0 \
                else float(cycles[-1])
            metrics_retry = optimizer_retry.calculate_fit_metrics()

            retry_is_late = (fit_end_cycle_r >= last_cyc - min(max(1, CYCLES_AFTER_MAX), 5))
            r2_target = 0.999

            # Ct for retry
            ct_retry, ct_baseline_retry, ct_bl_slope_retry, ct_bl_intercept_retry, ct_rox_mean_r = compute_ct(
                optimizer_retry, cycles, fluor_data, sample_name, sample_metadata,
                rox_by_well, channel_thresholds, global_threshold,
                channel_baseline_means, global_baseline_mean
            )

            # Track best result
            original_r2 = result.get('R2')
            retry_r2 = metrics_retry['r_squared']
            best_r2 = original_r2 if original_r2 is not None else -999.0
            best_result = None
            retry_better = (
                original_r2 is None
                or (retry_r2 is not None and original_r2 is not None and retry_r2 > original_r2)
            )

            if retry_better:
                best_r2 = retry_r2 if retry_r2 is not None else best_r2
                best_result = {
                    'params': params_retry, 'metrics': metrics_retry,
                    'optimizer': optimizer_retry, 'start_idx': retry_start_idx,
                    'fit_end': fit_end_cycle_r, 'ct': ct_retry,
                    'ct_bl_mean': ct_baseline_retry, 'ct_bl_slope': ct_bl_slope_retry,
                    'ct_bl_int': ct_bl_intercept_retry, 'ct_rox_mean': ct_rox_mean_r,
                }

            retry_stage = 'initial-retry'
            retry_attempts = 1
            retry_timed_out = False

            # Window variations
            if best_r2 < r2_target and time.perf_counter() - retry_t0 < RETRY_TIMEOUT:
                retry_stage = 'window-variations'
                win_variations = [
                    (CYCLES_BEFORE_MAX, CYCLES_AFTER_MAX),
                    (CYCLES_BEFORE_MAX, max(3, CYCLES_AFTER_MAX - 1)),
                    (CYCLES_BEFORE_MAX + 4, CYCLES_AFTER_MAX),
                    (CYCLES_BEFORE_MAX + 8, CYCLES_AFTER_MAX),
                    (CYCLES_BEFORE_MAX, CYCLES_AFTER_MAX + 3),
                    (CYCLES_BEFORE_MAX - 2, CYCLES_AFTER_MAX + 3),
                    (CYCLES_BEFORE_MAX + 4, CYCLES_AFTER_MAX + 3),
                    (CYCLES_BEFORE_MAX - 4, CYCLES_AFTER_MAX),
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
                        wv_model = MAK2Model()
                        wv_opt = MAK2Optimizer(wv_model)
                        wv_params = wv_opt.fit(
                            wv_c, wv_f,
                            cycles_after_max=wv_cam,
                            auto_truncate=AUTO_TRUNCATE,
                            truncate_cycle=TRUNCATE_CYCLE,
                            bounds=wv_bounds,
                            fixed_background_values={
                                'F_bg_slope': wv_slope,
                                'F_bg_intercept': wv_int,
                            },
                            verbose=False,
                        )
                        wv_metrics = wv_opt.calculate_fit_metrics()
                        wv_r2 = wv_metrics['r_squared']
                        if wv_r2 is not None and wv_r2 > best_r2:
                            best_r2 = wv_r2
                            wv_fe = float(wv_opt.cycles_fit[-1]) \
                                if wv_opt.cycles_fit is not None and len(wv_opt.cycles_fit) > 0 \
                                else float(cycles[-1])
                            best_result = {
                                'params': wv_params, 'metrics': wv_metrics,
                                'optimizer': wv_opt, 'start_idx': wv_start,
                                'fit_end': wv_fe, 'ct': ct_retry,
                                'ct_bl_mean': ct_baseline_retry,
                                'ct_bl_slope': ct_bl_slope_retry,
                                'ct_bl_int': ct_bl_intercept_retry,
                                'ct_rox_mean': ct_rox_mean_r,
                            }
                            if best_r2 >= r2_target:
                                break
                    except Exception:
                        pass
                    retry_attempts += 1
                    if time.perf_counter() - retry_t0 >= RETRY_TIMEOUT:
                        retry_timed_out = True
                        break

            # k relaxation
            if (best_r2 < r2_target and not retry_timed_out
                    and time.perf_counter() - retry_t0 < RETRY_TIMEOUT):
                retry_stage = 'k-relaxation'
                retry_attempts += 1
                cur_k = (best_result['params']['k'] if best_result else params_retry.get('k'))
                k_lo = informed_bounds['k'][0]
                if cur_k is not None and cur_k < k_lo * 1.5:
                    relax_bounds = dict(informed_bounds)
                    relax_bounds['k'] = (0.001, relax_bounds['k'][1])
                    relax_bounds['D0'] = (1e-15, max(F_range_r * 10, 1.0))
                    relax_bounds['F_bg_slope'] = (r_s_min, r_s_max)
                    relax_bounds['F_bg_intercept'] = (r_int_lo, r_int_hi)
                    try:
                        rk_model = MAK2Model()
                        rk_opt = MAK2Optimizer(rk_model)
                        rk_params = rk_opt.fit(
                            cycles_retry, fluor_retry,
                            cycles_after_max=retry_cam,
                            auto_truncate=AUTO_TRUNCATE,
                            truncate_cycle=TRUNCATE_CYCLE,
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
                            rk_fe = float(rk_opt.cycles_fit[-1]) \
                                if rk_opt.cycles_fit is not None and len(rk_opt.cycles_fit) > 0 \
                                else float(cycles[-1])
                            best_result = {
                                'params': rk_params, 'metrics': rk_metrics,
                                'optimizer': rk_opt, 'start_idx': retry_start_idx,
                                'fit_end': rk_fe, 'ct': ct_retry,
                                'ct_bl_mean': ct_baseline_retry,
                                'ct_bl_slope': ct_bl_slope_retry,
                                'ct_bl_int': ct_bl_intercept_retry,
                                'ct_rox_mean': ct_rox_mean_r,
                            }
                    except Exception:
                        pass

            # Accept best result
            if best_result is not None and (original_r2 is None or best_r2 > original_r2):
                br = best_result
                results_list[idx] = {
                    'Sample': sample_name,
                    'Ct': br['ct'],
                    'Ct_baseline_mean': br['ct_bl_mean'],
                    'Ct_baseline_slope': br['ct_bl_slope'],
                    'Ct_baseline_intercept': br['ct_bl_int'],
                    'D0': br['params']['D0'],
                    'k': br['params']['k'],
                    'P0': br['params']['P0'],
                    'F_bg_intercept': br['params']['F_bg_intercept'],
                    'F_bg_slope': br['params']['F_bg_slope'],
                    'R2': br['metrics']['r_squared'],
                    'RMSE': br['metrics']['rmse'],
                    'NRMSE': br['metrics']['nrmse'] * 100,
                    'SSR': br['metrics']['ssr'],
                    'Tier': result.get('Tier'),
                    'Instrument': result.get('Instrument', ''),
                    'Success': ('✓ (window-retry)' if best_r2 >= 0.999
                                else ('✓ (late-amp)' if retry_is_late and best_r2 >= 0.995
                                      else ('⚠️ timeout@' + retry_stage if retry_timed_out
                                            else ('✓ (noisy data)' if best_r2 >= 0.995
                                                  else '⚠️ R² below target')))),
                    'retry_stage': retry_stage,
                    'retry_attempts': retry_attempts,
                    'FixedBG': '', 'Fallback': '', 'FallbackOK': '',
                    'bg_slope_est': bg_slope_est_r,
                    'bg_intercept_est': bg_intercept_est_r,
                    'bl_end_meta': r_meta_bl_end_cycle,
                    'bl_end_est': r_est_bl_end_cycle,
                    'fit_start_cycle': float(cycles[br['start_idx']]),
                    'fit_end_cycle': br['fit_end'],
                    'ct_rox_mean': br['ct_rox_mean'],
                    'fluor_data': fluor_data,
                }
                elapsed = time.perf_counter() - retry_t0
                print(f"  → R²={best_r2:.4f} ({elapsed:.1f}s)")
            else:
                # Keep original
                orig_r2 = result.get('R2')
                orig_fe = result.get('fit_end_cycle')
                orig_late = (orig_fe is not None
                             and not (isinstance(orig_fe, float) and np.isnan(orig_fe))
                             and float(orig_fe) >= last_cyc - min(max(1, CYCLES_AFTER_MAX), 5))
                orig_r2_thr = 0.995 if orig_late else 0.999
                if orig_r2 is not None and orig_r2 >= orig_r2_thr:
                    results_list[idx]['Success'] = '✓'
                else:
                    orig_k = result.get('k')
                    if orig_k is not None and orig_k > 0.5:
                        results_list[idx]['Success'] = '⚠️ Degenerate k'
                    elif orig_r2 is not None and orig_r2 >= 0.995:
                        results_list[idx]['Success'] = '✓ (noisy data)'
                    else:
                        results_list[idx]['Success'] = '⚠️ R² below target'
                elapsed = time.perf_counter() - retry_t0
                print(f"  → kept original ({elapsed:.1f}s)")

        except Exception as e:
            if result['k'] is None:
                results_list[idx]['Success'] = f'✗ Error: {str(e)[:30]}'
            else:
                results_list[idx]['Success'] = '⚠️ Retry failed'
            print(f"  → retry error: {e}")

    return results_list


def run_quality_gates(results_list, cycles):
    """Pass 3: Quality gates — identical to app.py logic."""
    for pf_idx, pf_r in enumerate(results_list):
        if pf_r.get('error') is not None:
            continue
        pf_r2 = pf_r.get('R2')
        pf_reject = False
        pf_reason = ''

        # Late amplifier detection
        pf_fe_g0 = pf_r.get('fit_end_cycle')
        pf_is_late = (
            pf_fe_g0 is not None
            and not (isinstance(pf_fe_g0, float) and np.isnan(pf_fe_g0))
            and pf_fe_g0 >= float(cycles[-1]) - min(max(1, CYCLES_AFTER_MAX), 5)
        )

        # Gate 0: R² threshold
        pf_r2_thresh = 0.85 if pf_is_late else 0.99
        if pf_r2 is not None and pf_r2 < pf_r2_thresh:
            pf_reject = True
            pf_reason = f'R² = {pf_r2:.4f} < {pf_r2_thresh}'

        # Gate 2: fit window width
        if not pf_reject:
            pf_fs2 = pf_r.get('fit_start_cycle')
            pf_fe2 = pf_r.get('fit_end_cycle')
            if (pf_fs2 is not None and pf_fe2 is not None
                    and pf_fe2 - pf_fs2 < 10):
                pf_reject = True
                pf_reason = f'Fit window {pf_fe2 - pf_fs2:.0f} cycles < 10'

        # Gate 2b: linear vs MAK2
        pf_late_bypass_2b = pf_is_late and pf_r2 is not None and pf_r2 >= 0.995
        pf_fit_width_2b = (
            (pf_r.get('fit_end_cycle') or 0)
            - (pf_r.get('fit_start_cycle') or 0)
        )
        pf_high_r2_2b = (
            pf_r2 is not None
            and pf_r2 >= 0.999
            and pf_fit_width_2b >= 10
        )
        if (not pf_reject and not pf_late_bypass_2b
                and not pf_high_r2_2b
                and pf_r.get('D0') is not None
                and not (isinstance(pf_r['D0'], float) and np.isnan(pf_r['D0']))
                and pf_r.get('fluor_data') is not None):
            try:
                pf_fs2b = pf_r.get('fit_start_cycle')
                pf_fe2b = pf_r.get('fit_end_cycle')
                if pf_fs2b is not None and pf_fe2b is not None:
                    pf_m2b = MAK2Model()
                    pf_c_full2b = cycles[:len(pf_r['fluor_data'])]
                    pf_pred2b = pf_m2b.simulate_to_cycle(
                        D0=pf_r['D0'], k=pf_r['k'], P0=pf_r['P0'],
                        cycles=pf_c_full2b,
                        F_bg_intercept=pf_r['F_bg_intercept'],
                        F_bg_slope=pf_r['F_bg_slope'],
                    )
                    pf_win2b = (pf_c_full2b >= pf_fs2b) & (pf_c_full2b <= pf_fe2b)
                    pf_cycles_win = pf_c_full2b[pf_win2b]
                    pf_pred_win2b = pf_pred2b[pf_win2b]
                    pf_fluor_full2b = np.asarray(pf_r['fluor_data'])
                    pf_fluor_win2b = pf_fluor_full2b[pf_win2b]

                    pf_d1_2b = np.gradient(pf_pred_win2b, pf_cycles_win)
                    pf_max_slope_idx = int(np.argmax(pf_d1_2b))
                    pf_max_slope_cycle = pf_cycles_win[pf_max_slope_idx]

                    pf_pre_mask = pf_cycles_win <= pf_max_slope_cycle
                    pf_fluor_pre = pf_fluor_win2b[pf_pre_mask]
                    pf_cycles_pre = pf_cycles_win[pf_pre_mask]

                    if len(pf_fluor_pre) >= 4:
                        pf_coeffs = np.polyfit(pf_cycles_pre, pf_fluor_pre, 1)
                        pf_lin_pred = np.polyval(pf_coeffs, pf_cycles_pre)
                        pf_ss_tot = float(np.sum((pf_fluor_pre - np.mean(pf_fluor_pre))**2))
                        if pf_ss_tot > 0:
                            pf_ss_res_lin = float(np.sum((pf_fluor_pre - pf_lin_pred)**2))
                            pf_r2_lin = 1.0 - pf_ss_res_lin / pf_ss_tot
                            pf_mak2_pre = pf_pred_win2b[pf_pre_mask]
                            pf_ss_res_mak = float(np.sum((pf_fluor_pre - pf_mak2_pre)**2))
                            pf_r2_mak = 1.0 - pf_ss_res_mak / pf_ss_tot
                            if pf_r2_mak - pf_r2_lin < 0.10:
                                pf_reject = True
                                pf_reason = (
                                    f'MAK2 not better than linear in growth region '
                                    f'(R²_MAK2={pf_r2_mak:.4f}, R²_lin={pf_r2_lin:.4f})'
                                )
            except Exception:
                pass

        # Gate 3: sigmoid shape
        pf_fit_width = (
            (pf_r.get('fit_end_cycle') or 0)
            - (pf_r.get('fit_start_cycle') or 0)
        )
        pf_high_r2 = pf_r2 is not None and pf_r2 >= 0.999 and pf_fit_width >= 10
        pf_late_bypass_3 = pf_is_late and pf_r2 is not None and pf_r2 >= 0.995
        if (not pf_reject and not pf_late_bypass_3 and not pf_high_r2
                and pf_r.get('D0') is not None
                and not (isinstance(pf_r['D0'], float) and np.isnan(pf_r['D0']))
                and pf_r.get('fluor_data') is not None):
            try:
                pf_fs3 = pf_r.get('fit_start_cycle')
                pf_fe3 = pf_r.get('fit_end_cycle')
                if pf_fs3 is not None and pf_fe3 is not None:
                    pf_m = MAK2Model()
                    pf_c_full = cycles[:len(pf_r['fluor_data'])]
                    pf_pred = pf_m.simulate_to_cycle(
                        D0=pf_r['D0'], k=pf_r['k'], P0=pf_r['P0'],
                        cycles=pf_c_full,
                        F_bg_intercept=pf_r['F_bg_intercept'],
                        F_bg_slope=pf_r['F_bg_slope'],
                    )
                    pf_win_mask = (pf_c_full >= pf_fs3) & (pf_c_full <= pf_fe3)
                    pf_pred_win = pf_pred[pf_win_mask]
                    if len(pf_pred_win) >= 5:
                        pf_d1 = np.gradient(pf_pred_win)
                        pf_d2 = np.gradient(pf_d1)
                        pf_pred_range = float(np.max(pf_pred_win) - np.min(pf_pred_win))
                        pf_d2_thresh = pf_pred_range * 0.01
                        pf_has_inflection = (
                            np.any(pf_d2 > pf_d2_thresh)
                            and np.any(pf_d2 < -pf_d2_thresh)
                        )
                        if not pf_has_inflection:
                            pf_reject = True
                            pf_reason = 'No inflection (monotone curve)'
            except Exception:
                pass

        if pf_reject:
            results_list[pf_idx]['Success'] = ''
            results_list[pf_idx]['error'] = f'No amplification detected ({pf_reason})'
            results_list[pf_idx]['D0'] = None
            results_list[pf_idx]['k'] = None
            results_list[pf_idx]['P0'] = None
            results_list[pf_idx]['Ct'] = None

    return results_list


def build_replicate_groups(results_list, sample_metadata, channels):
    """Build replicate groups from Sample Name in metadata."""
    # Map sample keys to their sample names
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'fluor_data'} for r in results_list])
    if results_df.empty:
        return None

    # Add Group column based on sample name from metadata
    groups = {}
    for _, row in results_df.iterrows():
        sample_key = row['Sample']
        meta = sample_metadata.get(sample_key, {})
        sample_name = meta.get('Sample Name', sample_key)
        target = meta.get('Target Name', '')

        # For multi-target, prefix with target name
        if target and len(channels) > 1:
            group_key = f"{target} -- {sample_name}"
        else:
            group_key = sample_name

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(sample_key)

    # Only keep groups with >1 member
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not groups:
        return None

    # Build group mapping: sample_key -> group_name
    sample_to_group = {}
    for group_name, members in groups.items():
        for m in members:
            sample_to_group[m] = group_name

    results_df['Group'] = results_df['Sample'].map(sample_to_group)

    # Filter to only grouped samples with valid results
    grouped_df = results_df[results_df['Group'].notna()].copy()
    if grouped_df.empty:
        return None

    return calculate_replicate_stats(grouped_df)


def build_excel(results_list, cycles, all_samples, no_signal_samples,
                no_signal_fluor, sample_metadata, channels, batch_settings,
                replicate_stats_df=None, precision_comparison_df=None,
                std_curve_sheets=None):
    """Build the Excel file matching the app's format exactly."""
    from openpyxl.chart import ScatterChart, Reference, Series as XlSeries

    if std_curve_sheets is None:
        std_curve_sheets = {}

    # Build results DataFrame (excluding internal-only keys)
    # KEEP 'error' column — the app needs it to show FAIL status on reload
    hidden = {'fluor_data', 'bg_slope_est', 'bg_intercept_est', 'retry_stage',
              'retry_attempts', 'retry_timed_out', 'retry_elapsed_s'}
    display_results = [{k: v for k, v in r.items() if k not in hidden} for r in results_list]
    results_df = pd.DataFrame(display_results)

    # Ensure empty strings in Success don't become NaN on Excel roundtrip
    results_df['Success'] = results_df['Success'].fillna('')

    # Add Channel and Well columns for multi-channel data
    if channels and len(channels) > 1 and results_df['Sample'].str.contains('_').any():
        results_df.insert(0, 'Channel', results_df['Sample'].apply(_ch))
        results_df.insert(1, 'Well', results_df['Sample'].apply(_get_well_pos))

    extra_sheets = {}

    # No Signal Samples sheet
    if no_signal_samples:
        no_signal_df = pd.DataFrame([
            {
                'Sample': name,
                'Reason': info['reason'],
                'Fluorescence Range': f"{info['F_range']:.4f}",
                '% of Max on Plate': f"{info['F_range_pct']:.1f}%"
            }
            for name, info in no_signal_samples.items()
        ])
        extra_sheets['No Signal Samples'] = no_signal_df

    # Replicate Statistics sheet
    if replicate_stats_df is not None and len(replicate_stats_df) > 0:
        extra_sheets['Replicate Statistics'] = replicate_stats_df

    # Precision Comparison sheet
    if precision_comparison_df is not None and len(precision_comparison_df) > 0:
        extra_sheets['Precision Comparison'] = precision_comparison_df

    # Standard curve sheets (variance, D0, Ct — per channel)
    for sheet_name, df in std_curve_sheets.items():
        if df is not None and len(df) > 0:
            extra_sheets[sheet_name] = df

    # Input Data sheet: raw fluorescence
    input_data = {'Cycle': cycles}
    for wn, wd in all_samples.items():
        arr = np.asarray(wd)
        if arr.ndim >= 1:
            input_data[wn] = arr[:len(cycles)]
    for ns_name in no_signal_samples:
        if ns_name not in input_data and ns_name in no_signal_fluor:
            arr = np.asarray(no_signal_fluor[ns_name])
            if arr.ndim >= 1:
                input_data[ns_name] = arr[:len(cycles)]
    for rl in results_list:
        rn = rl.get('Sample', '')
        if rn and rn not in input_data and rl.get('fluor_data') is not None:
            arr = np.asarray(rl['fluor_data'])
            if arr.ndim >= 1:
                input_data[rn] = arr[:len(cycles)]
    extra_sheets['Input Data'] = pd.DataFrame(input_data)

    # Metadata sheet
    if sample_metadata:
        meta_rows = []
        for mk, mv in sample_metadata.items():
            row = {'Well_Key': mk}
            if isinstance(mv, dict):
                row.update(mv)
            meta_rows.append(row)
        if meta_rows:
            extra_sheets['Metadata'] = pd.DataFrame(meta_rows)

    # Settings sheet
    if batch_settings:
        settings_rows = []
        for sk, sv in batch_settings.items():
            if isinstance(sv, dict):
                for sk2, sv2 in sv.items():
                    settings_rows.append({
                        'Setting': f'{sk}.{sk2}',
                        'Value': str(sv2) if sv2 is not None else '',
                    })
            else:
                settings_rows.append({
                    'Setting': sk,
                    'Value': str(sv) if sv is not None else '',
                })
        if settings_rows:
            extra_sheets['Settings'] = pd.DataFrame(settings_rows)

    # Build Excel file
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Batch Results', index=False)
        for sheet_name, df in extra_sheets.items():
            if df is not None and len(df) > 0:
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
    buf.seek(0)
    return buf.getvalue()


def process_plate(mc_file, meta_file, plate_name):
    """Process a single plate end-to-end."""
    print(f"\n{'='*70}")
    print(f"  PLATE: {plate_name}")
    print(f"  Data:  {mc_file.name}")
    print(f"  Meta:  {meta_file.name}")
    print(f"{'='*70}")

    plate_t0 = time.perf_counter()

    # Load data
    print("\nLoading data...")
    cycles, all_samples, channels, rox_by_well, sample_metadata, abi_results_meta = \
        load_plate_data(mc_file, meta_file)
    print(f"  Loaded {len(all_samples)} wells, {len(cycles)} cycles, "
          f"channels: {channels}")

    # Phase 0: Signal detection (per-channel)
    print("\nPhase 0: Signal detection...")
    if len(channels) > 1:
        valid_samples = {}
        no_signal_samples = {}
        plate_stats = {'max_range': 0}
        for det_ch in channels:
            det_ch_samples = {
                name: fluor for name, fluor in all_samples.items()
                if _ch(name) == det_ch
            }
            if not det_ch_samples:
                continue
            v, ns, ps = detect_no_signal_samples(
                cycles, det_ch_samples,
                min_range_pct=2.0, min_r2=0.80, verbose=False
            )
            valid_samples.update(v)
            no_signal_samples.update(ns)
            plate_stats['max_range'] = max(plate_stats['max_range'], ps.get('max_range', 0))
    else:
        valid_samples, no_signal_samples, plate_stats = detect_no_signal_samples(
            cycles, all_samples, min_range_pct=2.0, min_r2=0.80, verbose=False
        )

    # Metadata rescue
    if sample_metadata:
        rescued = []
        for sname, info in list(no_signal_samples.items()):
            wm = sample_metadata.get(sname, {})
            if 'Ct_instrument' not in wm:
                continue
            inst_ct = wm.get('Ct_instrument')
            inst_noamp = wm.get('NOAMP', False)
            inst_undetermined = (
                inst_ct is None or (isinstance(inst_ct, float) and np.isnan(inst_ct))
            )
            if not inst_undetermined and not inst_noamp:
                valid_samples[sname] = all_samples[sname]
                del no_signal_samples[sname]
                rescued.append(sname)
        if rescued:
            print(f"  Rescued {len(rescued)} wells via instrument metadata")

    no_signal_fluor = {name: all_samples[name] for name in no_signal_samples if name in all_samples}
    all_samples_to_fit = valid_samples
    print(f"  Valid: {len(all_samples_to_fit)}, No signal: {len(no_signal_samples)}")

    # Phase 0b: Threshold computation
    print("\nComputing thresholds...")
    global_threshold = None
    global_baseline_mean = 0.0
    channel_thresholds = {}
    channel_baseline_means = {}

    all_fluorescence = list(all_samples_to_fit.values())
    if all_fluorescence:
        baseline_end = max(3, int(len(cycles) * 0.15))
        ch_arrays = {}
        for sname, fd in all_samples_to_fit.items():
            ch_arrays.setdefault(_ch(sname), []).append(fd)

        for ch, arrays in ch_arrays.items():
            ch_sds = []
            ch_means = []
            ch_early_all = []
            for fd in arrays:
                early = fd[:baseline_end]
                ch_early_all.extend(early)
                ch_sds.append(np.std(early))
                ch_means.append(np.mean(early))

            ch_sub = (np.mean(ch_early_all) < 0.1) or np.any(np.array(ch_early_all) < 0)
            ch_baseline_mean = 0.0 if ch_sub else np.median(ch_means)
            ch_median_sd = np.median(ch_sds)
            ch_max = np.max([np.max(fd) for fd in arrays])
            ch_dyn_range = ch_max - ch_baseline_mean
            ch_thresh = max(
                10 * ch_median_sd if ch_median_sd > 0 else 0.01,
                0.05 * ch_dyn_range
            )
            channel_thresholds[ch] = ch_thresh
            channel_baseline_means[ch] = ch_baseline_mean

        if len(channel_thresholds) == 1:
            global_threshold = next(iter(channel_thresholds.values()))
            global_baseline_mean = next(iter(channel_baseline_means.values()))
        else:
            all_sds = [s for arrays in ch_arrays.values()
                       for fd in arrays for s in [np.std(fd[:baseline_end])]]
            all_means = [np.mean(fd[:baseline_end])
                         for arrays in ch_arrays.values() for fd in arrays]
            already_sub = (np.mean([v for fd in all_fluorescence for v in fd[:baseline_end]]) < 0.1)
            global_baseline_mean = 0.0 if already_sub else np.median(all_means)
            plate_max = np.max([np.max(f) for f in all_fluorescence])
            global_threshold = max(
                10 * np.median(all_sds) if all_sds else 0.01,
                0.05 * (plate_max - global_baseline_mean)
            )

    # Override with instrument thresholds when ROX is available
    rox_norm_active = bool(rox_by_well)
    if abi_results_meta and rox_norm_active:
        inst_thresholds = abi_results_meta.get('channel_thresholds', {})
        if inst_thresholds:
            for ch, val in inst_thresholds.items():
                if ch in channel_thresholds:
                    channel_thresholds[ch] = val
            if len(inst_thresholds) == 1:
                global_threshold = next(iter(inst_thresholds.values()))

    # Pass 1
    print(f"\nPass 1: Fitting {len(all_samples_to_fit)} samples...")
    results_list = run_pass1(
        all_samples_to_fit, cycles, sample_metadata, rox_by_well,
        channel_thresholds, global_threshold, channel_baseline_means,
        global_baseline_mean
    )

    # Pass 2
    print(f"\nPass 2: Channel-aware retry...")
    results_list = run_pass2(
        results_list, cycles, sample_metadata, rox_by_well,
        channel_thresholds, global_threshold, channel_baseline_means,
        global_baseline_mean
    )

    # Pass 3: Quality gates
    print(f"\nPass 3: Quality gates...")
    results_list = run_quality_gates(results_list, cycles)

    # Count results
    n_success = sum(1 for r in results_list if str(r.get('Success', '')).startswith('✓'))
    n_failed = sum(1 for r in results_list if r.get('error') is not None)
    n_warn = len(results_list) - n_success - n_failed
    print(f"\n  Results: {n_success} ✓  {n_warn} ⚠️  {n_failed} ✗")

    # Replicate statistics
    print("\nComputing replicate statistics...")
    replicate_stats_df = build_replicate_groups(results_list, sample_metadata, channels)
    precision_comparison_df = None
    if replicate_stats_df is not None:
        print(f"  {len(replicate_stats_df)} replicate groups")
        # Precision comparison (Ct vs D0 CV%)
        try:
            precision_comparison_df = compare_precision(replicate_stats_df, efficiency=0.95)
            if precision_comparison_df is not None and len(precision_comparison_df) > 0:
                print(f"  Precision comparison: {len(precision_comparison_df)} groups")
        except Exception as e:
            print(f"  Precision comparison failed: {e}")
    else:
        print("  No replicate groups found")

    # Standard curve calibration
    print("\nComputing standard curves...")
    std_curve_sheets = {}
    has_standards = (
        sample_metadata is not None
        and any(m.get('Task') == 'STANDARD' for m in sample_metadata.values())
    )

    # Build results_df for calibration (same as what goes into Excel)
    hidden_cal = {'fluor_data', 'bg_slope_est', 'bg_intercept_est', 'retry_stage',
                  'retry_attempts', 'retry_timed_out', 'retry_elapsed_s'}
    cal_results = [{k: v for k, v in r.items() if k not in hidden_cal} for r in results_list]
    results_df_cal = pd.DataFrame(cal_results)

    if has_standards:
        # Determine per-channel calibration
        derived_ch = results_df_cal['Sample'].apply(_ch)
        unique_ch = derived_ch[derived_ch != 'default'].unique()
        if len(unique_ch) > 1:
            results_df_cal.insert(0, 'Channel', derived_ch)
            cal_channels = list(unique_ch)
        else:
            cal_channels = [None]

        for cal_ch in cal_channels:
            if cal_ch is not None:
                ch_mask = results_df_cal['Channel'] == cal_ch
                ch_df = results_df_cal[ch_mask].copy()
                ch_meta = {k: v for k, v in sample_metadata.items()
                           if k.startswith(f"{cal_ch}_")}
            else:
                ch_df = results_df_cal
                ch_meta = sample_metadata

            ch_label = f" ({cal_ch})" if cal_ch else ""

            calibration = build_standard_curve(ch_df, ch_meta)
            ct_calibration = build_ct_standard_curve(ch_df, ch_meta)

            if calibration is not None:
                print(f"  D0 standard curve{ch_label}: R²={calibration['r_squared']:.4f}, "
                      f"{calibration['n_standards']} wells, {calibration['n_concentrations']} levels")
                # D0 standard curve data sheet
                std_curve_sheets[f'Std Curve D0{ch_label}'] = calibration['per_point_data'].copy()

                # Variance sheet
                var_data = []
                for copies_val, var_info in sorted(calibration['replicate_variance'].items(), reverse=True):
                    row = {
                        'Known Copies': copies_val,
                        'N Replicates': var_info['n_replicates'],
                        'Mean D0': var_info['mean_D0'],
                        'SD D0': var_info['sd_D0'],
                        'D0 CV%': var_info['cv_D0_pct'],
                    }
                    if ct_calibration is not None:
                        ct_var = ct_calibration['replicate_variance'].get(copies_val)
                        if ct_var:
                            row['Mean Ct'] = ct_var['mean_Ct']
                            row['SD Ct'] = ct_var['sd_Ct']
                            row['Ct CV%'] = ct_var['cv_Ct_pct']
                    var_data.append(row)
                if var_data:
                    std_curve_sheets[f'Std Curve Variance{ch_label}'] = pd.DataFrame(var_data)

                # Apply calibration to results
                if cal_ch is not None:
                    ch_mask_apply = results_df_cal['Channel'] == cal_ch
                    cal_subset = apply_calibration(results_df_cal[ch_mask_apply].copy(), calibration=calibration)
                    results_df_cal.loc[ch_mask_apply, 'Copies_D0'] = cal_subset['Copies_D0']
                else:
                    results_df_cal = apply_calibration(results_df_cal, calibration=calibration)

            if ct_calibration is not None:
                print(f"  Ct standard curve{ch_label}: R²={ct_calibration['r_squared']:.4f}, "
                      f"efficiency={ct_calibration['efficiency']*100:.1f}%")
                std_curve_sheets[f'Std Curve Ct{ch_label}'] = ct_calibration['per_point_data'].copy()

                if cal_ch is not None:
                    ch_mask_apply = results_df_cal['Channel'] == cal_ch
                    ct_subset = apply_ct_calibration(results_df_cal[ch_mask_apply].copy(), ct_calibration)
                    results_df_cal.loc[ch_mask_apply, 'Copies_Ct'] = ct_subset['Copies_Ct']
                else:
                    results_df_cal = apply_ct_calibration(results_df_cal, ct_calibration)

            if calibration is None and ct_calibration is None:
                print(f"  No standard curve{ch_label} (insufficient data)")

        # Update results_list with calibrated copies if computed
        if 'Copies_D0' in results_df_cal.columns or 'Copies_Ct' in results_df_cal.columns:
            for i, r in enumerate(results_list):
                if 'Copies_D0' in results_df_cal.columns:
                    results_list[i]['Copies_D0'] = results_df_cal.iloc[i].get('Copies_D0', np.nan)
                if 'Copies_Ct' in results_df_cal.columns:
                    results_list[i]['Copies_Ct'] = results_df_cal.iloc[i].get('Copies_Ct', np.nan)
    else:
        print("  No STANDARD wells found in metadata — skipping")

    # Build batch settings
    batch_settings = {
        'first_fit_cycle': FIRST_FIT_CYCLE,
        'cycles_before_max': CYCLES_BEFORE_MAX,
        'cycles_after_max': CYCLES_AFTER_MAX,
        'auto_truncate': AUTO_TRUNCATE,
        'truncate_cycle': TRUNCATE_CYCLE,
        'custom_bounds_dict': CUSTOM_BOUNDS_DICT,
        'global_threshold': global_threshold,
        'global_baseline_mean': global_baseline_mean,
        'channel_thresholds': channel_thresholds,
        'channel_baseline_means': channel_baseline_means,
    }

    # Build and save Excel
    print("\nBuilding Excel file...")
    excel_bytes = build_excel(
        results_list, cycles, all_samples, no_signal_samples,
        no_signal_fluor, sample_metadata, channels, batch_settings,
        replicate_stats_df, precision_comparison_df, std_curve_sheets
    )

    output_path = OUTPUT_DIR / f"{plate_name}_MAK2Plus_Results.xlsx"
    with open(output_path, 'wb') as f:
        f.write(excel_bytes)

    elapsed = time.perf_counter() - plate_t0
    print(f"\n✅ {plate_name} complete in {elapsed/60:.1f} minutes")
    print(f"   Saved to: {output_path}")

    return output_path


# ── Main ────────���────────────────────���────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  MAK2+ Offline Batch Fitting")
    print(f"  {len(PLATES)} plates to process")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    if not PLATES:
        print("\n❌ No plate files found! Check DATA_DIR path.")
        sys.exit(1)

    total_t0 = time.perf_counter()
    completed = []
    failed = []

    for mc_file, meta_file, plate_name in PLATES:
        try:
            output_path = process_plate(mc_file, meta_file, plate_name)
            completed.append((plate_name, output_path))
        except Exception as e:
            print(f"\n❌ {plate_name} FAILED: {e}")
            traceback.print_exc()
            failed.append((plate_name, str(e)))

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — {total_elapsed/60:.1f} minutes total")
    print(f"  Completed: {len(completed)}/{len(PLATES)}")
    for name, path in completed:
        print(f"    ✅ {name}: {path}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for name, err in failed:
            print(f"    ❌ {name}: {err}")
    print("=" * 70)
