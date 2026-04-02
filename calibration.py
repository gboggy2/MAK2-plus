"""
Standard curve calibration and replicate visualization for qPCR data.

Provides:
- Standard curve construction from known copy numbers vs D0
- Conversion factor derivation with error estimates
- Calibration application to convert D0 → copy numbers
- Diagnostic log-log calibration plot
- Replicate overlay visualization

Author: Greg Boggy, PhD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import linregress, t as t_dist, poisson
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Standard Curve ────────────────────────────────────────────────────────────


def build_standard_curve(
    results_df: pd.DataFrame,
    sample_metadata: Dict[str, Dict],
    r2_min: float = 0.99
) -> Optional[Dict]:
    """
    Build standard curve calibration using power-law (log-log regression).

    Primary conversion:
        log10(copies) = slope * log10(D0) + intercept

    This corrects for the ~4% D0 compression intrinsic to the MAK2 model.
    Testing showed the power-law outperforms both a single median CF and
    D0-neighbor interpolation in cross-validation, and works well with as
    few as 2-3 standard concentration levels.

    A median CF is also computed for diagnostic/informational purposes.

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch fitting results with 'Sample' and 'D0' columns.
    sample_metadata : dict
        Per-well metadata keyed by well position, e.g.:
        {'D1': {'Task': 'STANDARD', 'Quantity': 3313000.0, ...}, ...}
    r2_min : float
        Minimum R² for warning (default 0.99).

    Returns
    -------
    dict or None
        Calibration dict with keys:
            slope, intercept, r_squared, slope_stderr, intercept_stderr,
            median_cf, cf_spread, per_level_cf,
            replicate_variance, pooled_replicate_cv,
            n_standards, n_concentrations,
            per_point_data, warnings
        Returns None if < 2 concentration levels or no valid standards.
    """
    # 1. Identify standard wells from metadata
    standard_wells = {}
    for well, meta in (sample_metadata or {}).items():
        if meta.get('Task') == 'STANDARD' and meta.get('Quantity') is not None:
            standard_wells[well] = meta['Quantity']

    # Also check results_df columns (Task/Known_Copies from EDS Sample Setup)
    if 'Task' in results_df.columns and 'Known_Copies' in results_df.columns:
        for _, row in results_df.iterrows():
            well = row.get('Sample', '')
            if well and well not in standard_wells and row.get('Task') == 'STANDARD':
                qty = row.get('Known_Copies')
                if pd.notna(qty) and float(qty) > 0:
                    standard_wells[well] = float(qty)

    if len(standard_wells) < 2:
        return None

    # 2. Match to results_df to get D0 values
    points = []
    for well, copies in standard_wells.items():
        match = results_df[results_df['Sample'] == well]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        if pd.isna(row.get('D0')) or row.get('D0', 0) <= 0:
            continue
        success = str(row.get('Success', ''))
        if '\u2717' in success:
            continue
        points.append({
            'Well': well,
            'D0': row['D0'],
            'Known_Copies': copies,
            'R2': row.get('R2', np.nan),
            'log10_D0': np.log10(row['D0']),
            'log10_Copies': np.log10(copies),
            'CF': copies / row['D0'],
        })

    if len(points) < 2:
        return None

    points_df = pd.DataFrame(points)

    unique_copies = points_df['Known_Copies'].unique()
    if len(unique_copies) < 2:
        return None

    # 3. Primary: log-log regression (power-law standard curve)
    log_d0 = points_df['log10_D0'].values
    log_copies = points_df['log10_Copies'].values
    result = linregress(log_d0, log_copies)
    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    slope_stderr = result.stderr
    intercept_stderr = result.intercept_stderr

    # 4. Diagnostic: median CF and per-level CF (informational only)
    cf_values = points_df['CF'].values
    median_cf = float(np.median(cf_values))

    per_level_cf = {}
    for copies_val in sorted(unique_copies, reverse=True):
        mask = points_df['Known_Copies'] == copies_val
        group_cfs = points_df.loc[mask, 'CF'].values
        per_level_cf[copies_val] = {
            'median_cf': float(np.median(group_cfs)),
            'mean_cf': float(np.mean(group_cfs)),
            'n': len(group_cfs),
        }

    # CF spread: max / min of per-level medians
    level_medians = [v['median_cf'] for v in per_level_cf.values()]
    cf_spread = max(level_medians) / min(level_medians) if min(level_medians) > 0 else np.nan

    # 5. Replicate variance
    replicate_variance = {}
    all_cvs = []
    for copies_val in sorted(unique_copies, reverse=True):
        mask = points_df['Known_Copies'] == copies_val
        group = points_df[mask]
        d0_vals = group['D0'].values
        log_d0_vals = group['log10_D0'].values

        if len(d0_vals) > 1:
            mean_d0 = np.mean(d0_vals)
            sd_d0 = np.std(d0_vals, ddof=1)
            cv_d0 = (sd_d0 / mean_d0) * 100
            sd_log = np.std(log_d0_vals, ddof=1)
            all_cvs.append(cv_d0)
        else:
            mean_d0 = d0_vals[0]
            sd_d0 = np.nan
            cv_d0 = np.nan
            sd_log = np.nan

        replicate_variance[copies_val] = {
            'n_replicates': len(d0_vals),
            'mean_D0': mean_d0,
            'sd_D0': sd_d0,
            'cv_D0_pct': cv_d0,
            'sd_log10_D0': sd_log,
        }

    pooled_cv = np.mean([cv for cv in all_cvs if not np.isnan(cv)]) if all_cvs else np.nan

    # 6. Warnings
    warnings = []
    if r_squared < r2_min:
        warnings.append(
            f"Standard curve R\u00b2 ({r_squared:.4f}) is below {r2_min}. "
            f"Standard data may have high variability."
        )
    if pooled_cv and not np.isnan(pooled_cv) and pooled_cv > 20:
        warnings.append(
            f"High replicate variability (pooled CV = {pooled_cv:.1f}%). "
            f"Consider re-running standards."
        )
    if abs(slope - 1.0) > 0.15:
        warnings.append(
            f"Slope ({slope:.3f}) deviates from 1.0 by "
            f"{abs(slope - 1.0):.3f}. This is larger than the typical "
            f"~4% D0 compression. Check standard concentrations."
        )

    return {
        # Primary: power-law regression
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'slope_stderr': slope_stderr,
        'intercept_stderr': intercept_stderr,
        # Diagnostic: median CF info
        'median_cf': median_cf,
        'cf_spread': cf_spread,
        'per_level_cf': per_level_cf,
        # Replicate stats
        'replicate_variance': replicate_variance,
        'pooled_replicate_cv': pooled_cv,
        'n_standards': len(points_df),
        'n_concentrations': len(unique_copies),
        'per_point_data': points_df,
        'warnings': warnings,
    }


# ── Ct-Based Standard Curve ──────────────────────────────────────────────────


def build_ct_standard_curve(
    results_df: pd.DataFrame,
    sample_metadata: Dict[str, Dict],
) -> Optional[Dict]:
    """
    Build traditional Ct-based standard curve: log10(copies) = slope * Ct + intercept.

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch fitting results with 'Sample' and 'Ct' columns.
    sample_metadata : dict
        Per-well metadata with Task='STANDARD' and Quantity.

    Returns
    -------
    dict or None
        Calibration dict with slope, intercept, r_squared, efficiency,
        per_point_data, replicate_variance, warnings, etc.
    """
    if 'Ct' not in results_df.columns:
        return None

    # 1. Identify standard wells from metadata
    standard_wells = {}
    for well, meta in (sample_metadata or {}).items():
        if meta.get('Task') == 'STANDARD' and meta.get('Quantity') is not None:
            standard_wells[well] = meta['Quantity']

    # Also check results_df columns (Task/Known_Copies from EDS Sample Setup)
    if 'Task' in results_df.columns and 'Known_Copies' in results_df.columns:
        for _, row in results_df.iterrows():
            well = row.get('Sample', '')
            if well and well not in standard_wells and row.get('Task') == 'STANDARD':
                qty = row.get('Known_Copies')
                if pd.notna(qty) and float(qty) > 0:
                    standard_wells[well] = float(qty)

    if len(standard_wells) < 2:
        return None

    # 2. Match to results_df — prefer instrument Ct for fair comparison
    _used_instrument_ct = 0
    _used_fitted_ct = 0
    points = []
    for well, copies in standard_wells.items():
        match = results_df[results_df['Sample'] == well]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        # Prefer instrument Ct from metadata; fall back to app-calculated Ct
        _well_meta = (sample_metadata or {}).get(well, {})
        ct_val = _well_meta.get('Ct_instrument', None)
        if ct_val is not None and not (isinstance(ct_val, float) and np.isnan(ct_val)) and ct_val > 0:
            _used_instrument_ct += 1
        else:
            ct_val = row.get('Ct', np.nan)
            _used_fitted_ct += 1
        if pd.isna(ct_val) or ct_val <= 0:
            continue
        success = str(row.get('Success', ''))
        if '\u2717' in success:
            continue
        points.append({
            'Well': well,
            'Ct': ct_val,
            'Known_Copies': copies,
            'log10_Copies': np.log10(copies),
        })

    if len(points) < 3:
        return None

    points_df = pd.DataFrame(points)
    unique_copies = points_df['Known_Copies'].unique()
    if len(unique_copies) < 2:
        return None

    # 3. Linear regression: log10(copies) = slope * Ct + intercept
    ct_vals = points_df['Ct'].values
    log_copies = points_df['log10_Copies'].values

    result = linregress(ct_vals, log_copies)
    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    slope_stderr = result.stderr
    intercept_stderr = result.intercept_stderr

    # 4. Efficiency: E = 10^(-slope) - 1
    # Our regression is log10(copies) = slope * Ct + intercept, so our slope
    # is the reciprocal of the traditional Ct-vs-log10(copies) slope.
    # The standard formula E = 10^(-1/m) - 1 uses the traditional slope m,
    # which equals 1/slope here, giving E = 10^(-1/(1/slope)) - 1 = 10^(-slope) - 1.
    efficiency = (10 ** (-slope) - 1) if slope != 0 else np.nan

    # 5. Replicate variance
    replicate_variance = {}
    all_cvs = []
    for copies_val in sorted(unique_copies, reverse=True):
        mask = points_df['Known_Copies'] == copies_val
        group = points_df[mask]
        ct_group = group['Ct'].values

        if len(ct_group) > 1:
            mean_ct = np.mean(ct_group)
            sd_ct = np.std(ct_group, ddof=1)
            cv_ct = (sd_ct / mean_ct) * 100 if mean_ct > 0 else np.nan
            all_cvs.append(cv_ct)
        else:
            mean_ct = ct_group[0]
            sd_ct = np.nan
            cv_ct = np.nan

        replicate_variance[copies_val] = {
            'n_replicates': len(ct_group),
            'mean_Ct': mean_ct,
            'sd_Ct': sd_ct,
            'cv_Ct_pct': cv_ct,
        }

    pooled_cv = np.mean([cv for cv in all_cvs if not np.isnan(cv)]) if all_cvs else np.nan

    # 6. Warnings
    warnings = []
    if r_squared < 0.98:
        warnings.append(
            f"R\u00b2 ({r_squared:.4f}) is below 0.98. "
            f"Ct standard curve may be unreliable."
        )
    if not np.isnan(efficiency):
        if efficiency < 0.80:
            warnings.append(
                f"Low amplification efficiency ({efficiency*100:.1f}%). "
                f"Expected 80\u2013110%."
            )
        elif efficiency > 1.10:
            warnings.append(
                f"High amplification efficiency ({efficiency*100:.1f}%). "
                f"Expected 80\u2013110%. May indicate inhibition or primer dimers."
            )
    if pooled_cv and not np.isnan(pooled_cv) and pooled_cv > 3.0:
        warnings.append(
            f"High Ct replicate variability (pooled CV = {pooled_cv:.1f}%)."
        )

    # Determine Ct source label
    if _used_instrument_ct > 0 and _used_fitted_ct == 0:
        _ct_source = 'Instrument'
    elif _used_instrument_ct == 0:
        _ct_source = 'MAK2+ (calculated)'
    else:
        _ct_source = f'Mixed ({_used_instrument_ct} instrument, {_used_fitted_ct} calculated)'

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'slope_stderr': slope_stderr,
        'intercept_stderr': intercept_stderr,
        'efficiency': efficiency,
        'replicate_variance': replicate_variance,
        'pooled_replicate_cv': pooled_cv,
        'n_standards': len(points_df),
        'n_concentrations': len(unique_copies),
        'per_point_data': points_df,
        'ct_source': _ct_source,
        'warnings': warnings,
    }


# ── Calibration Application ──────────────────────────────────────────────────


def apply_calibration(
    results_df: pd.DataFrame,
    calibration: Optional[Dict] = None,
    manual_cf: Optional[float] = None
) -> pd.DataFrame:
    """
    Add 'Copies_D0' column to results_df using D0-based calibration.

    When a standard curve calibration dict is provided, uses the power-law:
        copies = 10^(slope * log10(D0) + intercept)

    When a manual_cf is provided, uses the simple linear relationship:
        copies = manual_cf * D0

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch fitting results with 'D0' column.
    calibration : dict, optional
        Output from build_standard_curve(). Uses power-law regression.
    manual_cf : float, optional
        Manual conversion factor: copies = manual_cf * D0.
        Used only if calibration is None.

    Returns
    -------
    pd.DataFrame
        Copy of results_df with 'Copies_D0' column added.
    """
    df = results_df.copy()

    if calibration is not None:
        slope = calibration['slope']
        intercept = calibration['intercept']
        valid_d0 = df['D0'].notna() & (df['D0'] > 0)
        df.loc[valid_d0, 'Copies_D0'] = 10 ** (
            slope * np.log10(df.loc[valid_d0, 'D0']) + intercept
        )
        df.loc[~valid_d0, 'Copies_D0'] = np.nan
    elif manual_cf is not None and manual_cf > 0:
        valid_d0 = df['D0'].notna() & (df['D0'] > 0)
        df.loc[valid_d0, 'Copies_D0'] = manual_cf * df.loc[valid_d0, 'D0']
        df.loc[~valid_d0, 'Copies_D0'] = np.nan
    else:
        return df

    return df


def apply_ct_calibration(
    results_df: pd.DataFrame,
    ct_calibration: Dict,
) -> pd.DataFrame:
    """
    Add 'Copies_Ct' column to results_df using Ct-based standard curve.

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch fitting results with 'Ct' column.
    ct_calibration : dict
        Output from build_ct_standard_curve().

    Returns
    -------
    pd.DataFrame
        Copy of results_df with 'Copies_Ct' column added.
    """
    df = results_df.copy()
    slope = ct_calibration['slope']
    intercept = ct_calibration['intercept']

    valid_ct = df['Ct'].notna() & (df['Ct'] > 0)
    df.loc[valid_ct, 'Copies_Ct'] = 10 ** (
        slope * df.loc[valid_ct, 'Ct'] + intercept
    )
    df.loc[~valid_ct, 'Copies_Ct'] = np.nan

    return df


# ── Limited Dilution Calibration ─────────────────────────────────────────────


def build_limited_dilution_calibration(
    results_df: pd.DataFrame,
    ld_wells: List[str],
    no_signal_wells: Optional[List[str]] = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
) -> Optional[Dict]:
    """
    Build conversion factor from limited dilution (digital-style) assay.

    Dilute sample to single-molecule level so wells follow Poisson distribution.
    Negative wells estimate lambda; positive-well D0 values give single-copy D0.

    CF = (N_total * lambda) / sum(D0_positive)

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch fitting results with 'Sample' and 'D0' columns.
    ld_wells : list of str
        ALL well positions in the limited dilution panel (positive AND negative).
    no_signal_wells : list of str, optional
        Wells flagged as no-signal by detect_no_signal_samples(). If None,
        negative wells are inferred from absence in results_df or D0 <= 0.
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation.
    confidence_level : float
        Confidence level for bootstrap intervals.

    Returns
    -------
    dict or None
        Calibration dict with keys: method, conversion_factor, cf_ci_lower,
        cf_ci_upper, cf_se, lambda_hat, lambda_se, n_total, n_positive,
        n_negative, fraction_negative, mean_d0_positive, d0_single_copy,
        pos_d0_values, poisson_probs, warnings, etc.
        Returns None if calibration is not possible.
    """
    warnings = []

    if not ld_wells or len(ld_wells) < 3:
        return None

    N_total = len(ld_wells)

    # ── Classify wells as positive or negative ───────────────────────────
    if no_signal_wells is None:
        no_signal_wells = []

    no_signal_set = set(no_signal_wells)
    fitted_wells = set(results_df['Sample'].values) if 'Sample' in results_df.columns else set()

    positive_wells = []
    negative_wells = []
    pos_d0_values = []

    for well in ld_wells:
        # Check if this well is negative
        if well in no_signal_set:
            negative_wells.append(well)
            continue

        if well not in fitted_wells:
            negative_wells.append(well)
            continue

        row = results_df[results_df['Sample'] == well].iloc[0]
        d0 = row.get('D0')

        # Failed fit or invalid D0
        if pd.isna(d0) or d0 <= 0:
            negative_wells.append(well)
            continue

        success = str(row.get('Success', ''))
        if '\u2717' in success:  # ✗ failure marker
            negative_wells.append(well)
            continue

        positive_wells.append(well)
        pos_d0_values.append(d0)

    N_neg = len(negative_wells)
    N_pos = len(positive_wells)
    pos_d0_values = np.array(pos_d0_values)

    # ── Edge cases ────────────────────────────────────────────────────────
    if N_neg == 0:
        warnings.append(
            "All wells show amplification — lambda cannot be estimated. "
            "Dilute sample further so 30–70% of wells are negative."
        )
        return {'warnings': warnings, 'method': 'limited_dilution'}

    if N_pos == 0:
        warnings.append(
            "No positive wells detected. Template may be too dilute or absent."
        )
        return {'warnings': warnings, 'method': 'limited_dilution'}

    if N_pos < 2:
        warnings.append(
            f"Only {N_pos} positive well — insufficient for reliable calibration."
        )
        return {'warnings': warnings, 'method': 'limited_dilution'}

    # ── Poisson rate ──────────────────────────────────────────────────────
    frac_neg = N_neg / N_total
    lambda_hat = -np.log(frac_neg)

    # Delta method SE for lambda
    lambda_se = np.sqrt(np.exp(lambda_hat) / N_total)

    # ── Conversion factor ─────────────────────────────────────────────────
    total_copies = N_total * lambda_hat
    total_d0 = np.sum(pos_d0_values)
    cf = total_copies / total_d0

    d0_single_copy = 1.0 / cf
    expected_copies_per_positive = lambda_hat / (1 - np.exp(-lambda_hat))

    # ── Poisson diagnostics ───────────────────────────────────────────────
    p0 = poisson.pmf(0, lambda_hat)
    p1 = poisson.pmf(1, lambda_hat)
    p2 = poisson.pmf(2, lambda_hat)
    p3plus = 1 - p0 - p1 - p2

    poisson_probs = {'P0': p0, 'P1': p1, 'P2': p2, 'P3+': p3plus}

    # Estimate observed copy-number bins from D0 values
    # Wells with D0 < 1.5 * d0_single → 1 copy, 1.5-2.5 → 2, >2.5 → 3+
    n_obs_1 = int(np.sum(pos_d0_values < 1.5 * d0_single_copy))
    n_obs_2 = int(np.sum(
        (pos_d0_values >= 1.5 * d0_single_copy) &
        (pos_d0_values < 2.5 * d0_single_copy)
    ))
    n_obs_3plus = int(np.sum(pos_d0_values >= 2.5 * d0_single_copy))

    observed_bins = {0: N_neg, 1: n_obs_1, 2: n_obs_2, '3+': n_obs_3plus}
    expected_counts = {
        0: N_total * p0,
        1: N_total * p1,
        2: N_total * p2,
        '3+': N_total * p3plus,
    }

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    well_d0_array = np.zeros(N_total)
    # Fill in D0 values for positive wells (first N_pos entries)
    # and leave zeros for negative wells
    pos_idx = 0
    for i, well in enumerate(ld_wells):
        if well in positive_wells:
            well_d0_array[i] = pos_d0_values[pos_idx]
            pos_idx += 1

    boot_cfs = []
    for _ in range(n_bootstrap):
        boot_indices = rng.choice(N_total, size=N_total, replace=True)
        boot_d0 = well_d0_array[boot_indices]
        boot_n_neg = int(np.sum(boot_d0 == 0))
        boot_n_pos = N_total - boot_n_neg

        if boot_n_neg == 0 or boot_n_pos == 0:
            continue  # skip degenerate resamples

        boot_lambda = -np.log(boot_n_neg / N_total)
        boot_total_copies = N_total * boot_lambda
        boot_total_d0 = np.sum(boot_d0[boot_d0 > 0])
        boot_cf = boot_total_copies / boot_total_d0
        boot_cfs.append(boot_cf)

    boot_cfs = np.array(boot_cfs)
    alpha = 1 - confidence_level
    if len(boot_cfs) > 10:
        cf_ci_lower = float(np.percentile(boot_cfs, 100 * alpha / 2))
        cf_ci_upper = float(np.percentile(boot_cfs, 100 * (1 - alpha / 2)))
        cf_se = float(np.std(boot_cfs))
    else:
        cf_ci_lower = np.nan
        cf_ci_upper = np.nan
        cf_se = np.nan
        warnings.append("Bootstrap failed — too many degenerate resamples.")

    # ── D0 statistics ─────────────────────────────────────────────────────
    mean_d0 = float(np.mean(pos_d0_values))
    median_d0 = float(np.median(pos_d0_values))
    sd_d0 = float(np.std(pos_d0_values, ddof=1)) if N_pos > 1 else np.nan
    cv_d0 = (sd_d0 / mean_d0 * 100) if N_pos > 1 else np.nan

    # ── Warnings ──────────────────────────────────────────────────────────
    if frac_neg < 0.10:
        warnings.append(
            f"Only {frac_neg*100:.0f}% of wells are negative "
            f"(\u03bb = {lambda_hat:.2f}). Multi-copy wells are common. "
            f"For better accuracy, dilute further to achieve 30\u201370% negative wells."
        )
    elif frac_neg < 0.20:
        pct_multi = p2 + p3plus
        warnings.append(
            f"Moderate loading (\u03bb = {lambda_hat:.2f}). "
            f"~{pct_multi/(1-p0)*100:.0f}% of positive wells have 2+ copies. "
            f"Results are valid but accuracy improves with more dilution."
        )

    if frac_neg > 0.90:
        warnings.append(
            f"Only {N_pos} positive wells ({(1-frac_neg)*100:.0f}%). "
            f"Large uncertainty on D0 estimate. Consider more wells or less dilution."
        )

    if N_total < 20:
        warnings.append(
            f"Only {N_total} limited dilution wells. "
            f"CIs will be wide. Recommend 48\u201396 wells."
        )

    if not np.isnan(cv_d0) and cv_d0 > 100:
        warnings.append(
            f"Extremely variable D0 values (CV = {cv_d0:.0f}%). "
            f"May indicate contamination or uneven dilution."
        )

    return {
        'method': 'limited_dilution',
        'conversion_factor': float(cf),
        'cf_ci_lower': cf_ci_lower,
        'cf_ci_upper': cf_ci_upper,
        'cf_se': cf_se,
        'lambda_hat': float(lambda_hat),
        'lambda_se': float(lambda_se),
        'n_total': N_total,
        'n_positive': N_pos,
        'n_negative': N_neg,
        'fraction_negative': float(frac_neg),
        'mean_d0_positive': mean_d0,
        'median_d0_positive': median_d0,
        'sd_d0_positive': sd_d0,
        'cv_d0_positive': cv_d0,
        'd0_single_copy': float(d0_single_copy),
        'pos_d0_values': pos_d0_values,
        'positive_wells': positive_wells,
        'negative_wells': negative_wells,
        'expected_copies_per_positive': float(expected_copies_per_positive),
        'poisson_probs': poisson_probs,
        'observed_bins': observed_bins,
        'expected_counts': expected_counts,
        'n_bootstrap': n_bootstrap,
        'n_bootstrap_successful': len(boot_cfs),
        'confidence_level': confidence_level,
        'warnings': warnings,
    }


# ── Diagnostic Calibration Plot ──────────────────────────────────────────────


def plot_calibration(calibration: Dict, channel_label: str = "") -> go.Figure:
    """
    Create diagnostic log-log scatter plot of standard curve.

    Shows individual standard wells, concentration means with error bars,
    the power-law regression line (primary), and the CF=median line
    (diagnostic, slope=1 on log-log).

    Parameters
    ----------
    calibration : dict
        Output from build_standard_curve().
    channel_label : str
        Optional channel/target label to include in the title (e.g. " (FAM)").

    Returns
    -------
    plotly.graph_objects.Figure
    """
    points_df = calibration['per_point_data']
    median_cf = calibration['median_cf']
    slope = calibration['slope']
    intercept = calibration['intercept']
    r_squared = calibration['r_squared']
    replicate_var = calibration['replicate_variance']

    fig = go.Figure()

    # Color palette for concentration levels
    unique_copies = sorted(points_df['Known_Copies'].unique(), reverse=True)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]

    # Plot individual wells by concentration level
    for i, copies_val in enumerate(unique_copies):
        mask = points_df['Known_Copies'] == copies_val
        group = points_df[mask]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=group['log10_D0'],
            y=group['log10_Copies'],
            mode='markers',
            name=f'{copies_val:.2e} copies',
            marker=dict(size=8, color=color, opacity=0.6),
            hovertemplate=(
                'Well: %{customdata[0]}<br>'
                'D0: %{customdata[1]:.4e}<br>'
                'Known Copies: %{customdata[2]:.2e}<br>'
                'CF: %{customdata[4]:.2e}<br>'
                'R\u00b2: %{customdata[3]:.4f}'
                '<extra></extra>'
            ),
            customdata=group[['Well', 'D0', 'Known_Copies', 'R2', 'CF']].values,
        ))

    # Plot concentration means with error bars
    mean_x = []
    mean_y = []
    error_x_vals = []
    for copies_val in unique_copies:
        mask = points_df['Known_Copies'] == copies_val
        group = points_df[mask]
        mean_log_d0 = group['log10_D0'].mean()
        mean_log_copies = group['log10_Copies'].mean()
        mean_x.append(mean_log_d0)
        mean_y.append(mean_log_copies)

        var_info = replicate_var.get(copies_val, {})
        sd_log = var_info.get('sd_log10_D0', 0)
        error_x_vals.append(sd_log if not np.isnan(sd_log) else 0)

    fig.add_trace(go.Scatter(
        x=mean_x,
        y=mean_y,
        mode='markers',
        name='Concentration means',
        marker=dict(size=14, color='black', symbol='diamond',
                    line=dict(width=2, color='white')),
        error_x=dict(type='data', array=error_x_vals, visible=True,
                     color='black', thickness=2, width=6),
        showlegend=True,
    ))

    # Power-law regression line (solid, primary)
    x_range = np.array([
        min(points_df['log10_D0']) - 0.3,
        max(points_df['log10_D0']) + 0.3
    ])
    y_fit = slope * x_range + intercept
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_fit,
        mode='lines',
        name=f'Power law (slope={slope:.3f})',
        line=dict(color='blue', width=2),
    ))

    # Median CF line: copies = CF * D0, i.e. slope=1 on log-log (diagnostic)
    y_cf = x_range + np.log10(median_cf)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_cf,
        mode='lines',
        name=f'Median CF (slope=1)',
        line=dict(color='gray', dash='dash', width=1.5),
    ))

    # Annotation
    cf_spread = calibration.get('cf_spread', np.nan)
    spread_str = f", CF spread={cf_spread:.2f}\u00d7" if not np.isnan(cf_spread) else ""

    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=(
            f"<b>log\u2081\u2080(copies) = {slope:.4f} \u00d7 log\u2081\u2080(D0) + {intercept:.4f}</b><br>"
            f"R\u00b2 = {r_squared:.6f}{spread_str}<br>"
            f"n = {calibration['n_standards']} wells, "
            f"{calibration['n_concentrations']} levels<br>"
            f"<i>Median CF = {median_cf:.2e} (diagnostic)</i>"
        ),
        showarrow=False,
        font=dict(size=11),
        align='left',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title=f'Standard Curve: D0 vs Known Copy Number{channel_label}',
        xaxis_title='log\u2081\u2080(D0) from MAK2 fit',
        yaxis_title='log\u2081\u2080(Known Copy Number)',
        height=500,
        showlegend=True,
        legend=dict(x=0.7, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
    )

    return fig


# ── Ct Standard Curve Plot ───────────────────────────────────────────────────


def plot_ct_calibration(ct_calibration: Dict, channel_label: str = "") -> go.Figure:
    """
    Create Ct standard curve plot: Ct (x) vs log10(Known Copies) (y).

    Shows individual standard wells, concentration means with Ct error bars,
    regression line, and annotation with slope, R², efficiency.

    Parameters
    ----------
    ct_calibration : dict
        Output from build_ct_standard_curve().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    points_df = ct_calibration['per_point_data']
    slope = ct_calibration['slope']
    intercept = ct_calibration['intercept']
    r_squared = ct_calibration['r_squared']
    efficiency = ct_calibration['efficiency']
    replicate_var = ct_calibration['replicate_variance']

    fig = go.Figure()

    unique_copies = sorted(points_df['Known_Copies'].unique(), reverse=True)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]

    # Individual wells
    for i, copies_val in enumerate(unique_copies):
        mask = points_df['Known_Copies'] == copies_val
        group = points_df[mask]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=group['Ct'],
            y=group['log10_Copies'],
            mode='markers',
            name=f'{copies_val:.2e} copies',
            marker=dict(size=8, color=color, opacity=0.6),
            hovertemplate=(
                'Well: %{customdata[0]}<br>'
                'Ct: %{x:.2f}<br>'
                'Known Copies: %{customdata[1]:.2e}'
                '<extra></extra>'
            ),
            customdata=group[['Well', 'Known_Copies']].values,
        ))

    # Concentration means with Ct error bars
    mean_x = []
    mean_y = []
    error_x_vals = []
    for copies_val in unique_copies:
        mask = points_df['Known_Copies'] == copies_val
        group = points_df[mask]
        mean_ct = group['Ct'].mean()
        mean_log_copies = group['log10_Copies'].mean()
        mean_x.append(mean_ct)
        mean_y.append(mean_log_copies)

        var_info = replicate_var.get(copies_val, {})
        sd_ct = var_info.get('sd_Ct', 0)
        error_x_vals.append(sd_ct if not np.isnan(sd_ct) else 0)

    fig.add_trace(go.Scatter(
        x=mean_x,
        y=mean_y,
        mode='markers',
        name='Concentration means',
        marker=dict(size=14, color='black', symbol='diamond',
                    line=dict(width=2, color='white')),
        error_x=dict(type='data', array=error_x_vals, visible=True,
                     color='black', thickness=2, width=6),
        showlegend=True,
    ))

    # Regression line
    ct_min = min(points_df['Ct'])
    ct_max = max(points_df['Ct'])
    ct_margin = (ct_max - ct_min) * 0.1
    x_range = np.array([ct_min - ct_margin, ct_max + ct_margin])
    y_fit = slope * x_range + intercept

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_fit,
        mode='lines',
        name=f'Fit (slope={slope:.4f})',
        line=dict(color='red', dash='dash', width=2),
    ))

    # Annotation
    eff_str = f"{efficiency*100:.1f}%" if not np.isnan(efficiency) else "N/A"
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=(
            f"log\u2081\u2080(copies) = {slope:.4f} \u00d7 Ct + {intercept:.3f}<br>"
            f"R\u00b2 = {r_squared:.4f}<br>"
            f"Efficiency = {eff_str}<br>"
            f"n = {ct_calibration['n_standards']} wells, "
            f"{ct_calibration['n_concentrations']} levels"
        ),
        showarrow=False,
        font=dict(size=11),
        align='left',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title=f'Ct Standard Curve{channel_label}',
        xaxis_title='Ct (threshold cycle)',
        yaxis_title='log\u2081\u2080(Known Copy Number)',
        height=500,
        showlegend=True,
        legend=dict(x=0.7, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
    )

    return fig


# ── Limited Dilution Diagnostic Plot ─────────────────────────────────────────


def plot_limited_dilution_diagnostics(ld_calibration: Dict) -> go.Figure:
    """
    Create diagnostic plots for limited dilution calibration.

    Two subplots side by side:
    - Left: D0 histogram of positive wells with expected copy-number marks
    - Right: Poisson expected vs observed copy-number distribution

    Parameters
    ----------
    ld_calibration : dict
        Output from build_limited_dilution_calibration().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'D0 Distribution of Positive Wells',
            'Poisson: Expected vs Observed'
        ),
        horizontal_spacing=0.12,
    )

    pos_d0 = ld_calibration['pos_d0_values']
    d0_single = ld_calibration['d0_single_copy']
    lambda_hat = ld_calibration['lambda_hat']
    cf = ld_calibration['conversion_factor']
    N_total = ld_calibration['n_total']
    N_pos = ld_calibration['n_positive']
    N_neg = ld_calibration['n_negative']

    # ── Left panel: D0 histogram ──────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=pos_d0,
            nbinsx=max(10, N_pos // 3),
            marker_color='#1f77b4',
            opacity=0.7,
            name='Positive wells',
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Vertical lines for expected copy-number D0 values
    copy_colors = ['#2ca02c', '#ff7f0e', '#d62728']
    copy_labels = ['1 copy', '2 copies', '3 copies']
    for k, (color, label) in enumerate(zip(copy_colors, copy_labels), start=1):
        d0_k = k * d0_single
        # Only draw if within reasonable range of data
        if d0_k <= np.max(pos_d0) * 1.5:
            fig.add_vline(
                x=d0_k, line_dash='dash', line_color=color, line_width=2,
                annotation_text=label,
                annotation_position='top',
                annotation_font_color=color,
                row=1, col=1,
            )

    # Annotation
    fig.add_annotation(
        x=0.02, y=0.95,
        xref='x domain', yref='y domain',
        text=(
            f"\u03bb = {lambda_hat:.3f}<br>"
            f"CF = {cf:.2e}<br>"
            f"D0(1 copy) = {d0_single:.3e}<br>"
            f"{N_pos}/{N_total} wells positive"
        ),
        showarrow=False,
        font=dict(size=10),
        align='left',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=4,
        row=1, col=1,
    )

    fig.update_xaxes(title_text='D0 (fluorescence units)', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)

    # ── Right panel: Poisson expected vs observed ─────────────────────────
    categories = ['0', '1', '2', '3+']
    expected = ld_calibration['expected_counts']
    observed = ld_calibration['observed_bins']

    expected_vals = [expected[0], expected[1], expected[2], expected['3+']]
    observed_vals = [observed[0], observed[1], observed[2], observed['3+']]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=expected_vals,
            name='Expected (Poisson)',
            marker_color='#1f77b4',
            opacity=0.7,
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=observed_vals,
            name='Observed (D0-based)',
            marker_color='#ff7f0e',
            opacity=0.7,
        ),
        row=1, col=2,
    )

    # Note about observed estimation
    probs = ld_calibration['poisson_probs']
    fig.add_annotation(
        x=0.98, y=0.95,
        xref='x2 domain', yref='y2 domain',
        text=(
            f"\u03bb = {lambda_hat:.3f}<br>"
            f"P(0) = {probs['P0']:.3f}<br>"
            f"P(1) = {probs['P1']:.3f}<br>"
            f"P(2) = {probs['P2']:.3f}<br>"
            f"P(3+) = {probs['P3+']:.3f}<br>"
            f"<i>Observed 1/2/3+ estimated<br>from D0 binning</i>"
        ),
        showarrow=False,
        font=dict(size=9),
        align='right',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='gray',
        borderwidth=1,
        borderpad=4,
        row=1, col=2,
    )

    fig.update_xaxes(title_text='Copies per well', row=1, col=2)
    fig.update_yaxes(title_text='Number of wells', row=1, col=2)

    fig.update_layout(
        title='Limited Dilution Calibration Diagnostics',
        height=450,
        barmode='group',
        showlegend=True,
        legend=dict(
            x=0.55, y=0.02,
            bgcolor='rgba(255,255,255,0.8)',
        ),
    )

    return fig


# ── Replicate Overlay Plot ────────────────────────────────────────────────────


def plot_replicate_overlay(
    cycles: np.ndarray,
    all_samples: Dict[str, np.ndarray],
    results_list: List[Dict],
    group_wells: List[str],
    group_name: str,
) -> go.Figure:
    """
    Overlay raw fluorescence + MAK2 fits for all wells in a replicate group.

    Each replicate gets a distinct color. The legend shows well position
    and key parameters (D0, k, R²).

    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers used for fitting.
    all_samples : dict
        Full sample dict {well: fluorescence_array}.
    results_list : list of dict
        Full results list from batch fitting (includes 'fluor_data').
    group_wells : list of str
        Well positions belonging to this replicate group.
    group_name : str
        Display name for the group.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from mak2_model import MAK2Model

    fig = go.Figure()

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]

    model = MAK2Model()

    for i, well in enumerate(group_wells):
        color = colors[i % len(colors)]

        # Get fluorescence data
        fluor = all_samples.get(well)
        if fluor is None:
            continue

        # Find this well's result
        result = None
        for r in results_list:
            if r.get('Sample') == well:
                result = r
                break

        # Plot raw data (dots)
        fig.add_trace(go.Scatter(
            x=cycles[:len(fluor)],
            y=fluor,
            mode='markers',
            name=f'{well} data',
            marker=dict(size=4, color=color, opacity=0.5),
            showlegend=False,
        ))

        # Plot fit (line) if available
        if result and result.get('D0') is not None:
            D0 = result['D0']
            k = result['k']
            P0 = result['P0']
            F_bg_int = result.get('F_bg_intercept', result.get('F_bg', 0))
            F_bg_slope = result.get('F_bg_slope', 0)
            r2 = result.get('R2', np.nan)

            # Generate fit curve
            try:
                fit_cycles = cycles[:len(fluor)]
                params = {
                    'D0': D0, 'k': k, 'P0': P0,
                    'F_bg_intercept': F_bg_int, 'F_bg_slope': F_bg_slope
                }
                F_pred = model.predict(fit_cycles, params)

                label = (
                    f'{well}: D0={D0:.2e}, k={k:.4f}, R\u00b2={r2:.4f}'
                )
                fig.add_trace(go.Scatter(
                    x=fit_cycles,
                    y=F_pred,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                ))
            except Exception:
                # If prediction fails, just show data
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name=f'{well}: fit failed',
                    line=dict(color=color, width=2, dash='dot'),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name=f'{well}: no fit',
                line=dict(color=color, width=2, dash='dot'),
            ))

    fig.update_layout(
        title=f'Replicate Group: {group_name} ({len(group_wells)} replicates)',
        xaxis_title='Cycle',
        yaxis_title='Fluorescence',
        height=500,
        showlegend=True,
        legend=dict(
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.85)',
        ),
    )

    return fig


# ── Replicate Parameter Summary ──────────────────────────────────────────────


def calculate_replicate_param_summary(
    results_df: pd.DataFrame,
    group_wells: List[str],
    param_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate mean +/- SE for parameters across replicates in a group.

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch fitting results.
    group_wells : list of str
        Well positions in this replicate group.
    param_cols : list of str, optional
        Parameter columns to summarize. Default: D0, k, P0, Ct, plus Copies if present.

    Returns
    -------
    pd.DataFrame
        Summary with index = parameter names, columns = Mean, SE, N.
    """
    group_results = results_df[results_df['Sample'].isin(group_wells)]

    if param_cols is None:
        param_cols = ['D0', 'k', 'P0', 'Ct']
        if 'Copies_D0' in results_df.columns:
            param_cols.append('Copies_D0')
        if 'Copies_Ct' in results_df.columns:
            param_cols.append('Copies_Ct')

    summary = {}
    for col in param_cols:
        if col not in group_results.columns:
            continue
        vals = group_results[col].dropna()
        n = len(vals)
        if n > 0:
            mean_val = vals.mean()
            se_val = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        else:
            mean_val = np.nan
            se_val = np.nan
        summary[col] = {'Mean': mean_val, 'SE': se_val, 'N': n}

    return pd.DataFrame(summary).T
