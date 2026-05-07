"""Replicate and dilution-series analysis for qPCR results.

Sits downstream of the per-well fit. After every well in a plate has
its D0/Ct/R² recorded, the functions here aggregate technical
replicates and compare the relative precision of D0-based vs.
Ct-based quantification — the headline scientific argument for
mechanistic modelling over Ct calibration is that D0 has lower CV%
across replicates on the same template (because D0 doesn't depend on
threshold-line position or per-well efficiency assumptions).

Public API:
    parse_sample_groups       — group sample names into replicates
    calculate_replicate_stats — mean/SD/CV per group for D0 and Ct
    analyze_dilution_series   — fit standard curves and compare R²
    plot_dilution_series_comparison — side-by-side Plotly figure
    compare_precision         — convert Ct CV% to concentration-equivalent CV%
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_sample_groups(sample_names: List[str], pattern: Optional[str] = None) -> Dict[str, List[str]]:
    """Group sample names into replicates by splitting on a delimiter.

    qPCR sample naming follows a few well-known conventions; this
    function recognises three of them. The grouping logic is purely
    string-based — no metadata is consulted.

    Args:
        sample_names: All sample names on the plate.
        pattern: Which split convention to use:

            - ``'dot'`` or ``None`` (default): split on the **last**
              dot. ``"F1.1"`` → group ``"F1"``. Matches the qpcR R
              package's Boggy/Rutledge dataset naming.
            - ``'underscore'``: split on the last underscore.
              ``"Sample_A_rep1"`` → group ``"Sample_A"``.
            - ``'first_part'``: split on the **first** dot.
              ``"X1.R1.1"`` → group ``"X1"``. Used for nested
              replicate naming like the Rutledge dataset's
              experimental × technical replicate layout.
            - any other value: no grouping (each sample is its own
              group).

    Returns:
        ``{group_name: [sample, sample, ...]}``. Singleton groups
        are dropped — a group with only one sample isn't a replicate
        group.
    """
    groups = {}

    for name in sample_names:
        if pattern == 'dot' or pattern is None:
            # Split on last dot: "F1.1" → "F1"
            parts = name.rsplit('.', 1)
            group = parts[0] if len(parts) > 1 else name
        elif pattern == 'underscore':
            # Split on last underscore: "Sample_A_1" → "Sample_A"
            parts = name.rsplit('_', 1)
            group = parts[0] if len(parts) > 1 else name
        elif pattern == 'first_part':
            # Split on first dot: "X1.R1.1" → "X1"
            parts = name.split('.', 1)
            group = parts[0] if len(parts) > 1 else name
        else:
            # No grouping
            group = name

        if group not in groups:
            groups[group] = []
        groups[group].append(name)

    # Remove groups with only 1 sample (not replicates)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    return groups


def calculate_replicate_stats(
    results_df: pd.DataFrame,
    group_column: str = 'Group',
    metrics: List[str] = ['D0', 'Ct']
) -> pd.DataFrame:
    """Aggregate per-well results into per-group descriptive statistics.

    For each metric and each replicate group, computes mean, sample
    standard deviation (``ddof=1``), CV% (= SD/mean × 100), and
    min/max. NaN-only groups produce NaN stats rather than raising.

    The returned CV% is the standard biological-precision metric for
    a replicate set; it is what feeds into ``compare_precision`` for
    the D0-vs-Ct precision argument.

    Args:
        results_df: One row per well, must include ``group_column``
            and every name in ``metrics``.
        group_column: DataFrame column to group by. Default
            ``'Group'`` is the column produced by upstream code that
            applied ``parse_sample_groups``.
        metrics: Which numeric columns to summarise. Default
            ``['D0', 'Ct']`` covers the two quantification methods
            being compared.

    Returns:
        One row per group with columns ``Group``, ``N``, and
        ``{metric}_Mean``, ``{metric}_SD``, ``{metric}_CV``,
        ``{metric}_Min``, ``{metric}_Max`` for each metric.
    """
    stats = []

    for group, group_df in results_df.groupby(group_column):
        stat_row = {'Group': group, 'N': len(group_df)}

        for metric in metrics:
            if metric not in group_df.columns:
                stat_row[f'{metric}_Mean'] = np.nan
                stat_row[f'{metric}_SD'] = np.nan
                stat_row[f'{metric}_CV'] = np.nan
                continue
            values = group_df[metric].dropna()

            if len(values) == 0:
                stat_row[f'{metric}_Mean'] = np.nan
                stat_row[f'{metric}_SD'] = np.nan
                stat_row[f'{metric}_CV'] = np.nan
            else:
                mean_val = np.mean(values)
                sd_val = np.std(values, ddof=1)
                cv_val = (sd_val / mean_val * 100) if mean_val != 0 else np.nan

                stat_row[f'{metric}_Mean'] = mean_val
                stat_row[f'{metric}_SD'] = sd_val
                stat_row[f'{metric}_CV'] = cv_val
                stat_row[f'{metric}_Min'] = np.min(values)
                stat_row[f'{metric}_Max'] = np.max(values)

        stats.append(stat_row)

    return pd.DataFrame(stats)


def analyze_dilution_series(
    results_df: pd.DataFrame,
    dilution_factors: Optional[Dict[str, float]] = None,
    dilution_factor: float = 2,
    group_column: str = 'Group'
) -> Dict:
    """Fit standard curves for both Ct and D0 across a dilution series.

    Two regressions, both using log10(dilution) as the x-axis:

    - **Ct curve.** Standard qPCR practice: ``Ct = -m * log10(quantity) + b``.
      The slope ``m`` should be ``-3.32`` for 100% efficient
      doubling (one cycle delay per 10× dilution × log2(10) ≈ 3.32
      cycles). Reported efficiency = ``10^(-1/m) - 1``, expressed
      as a percentage.
    - **D0 curve.** Mechanistic equivalent: ``log10(D0) = m * log10(dilution) + b``.
      For a perfect dilution series the slope should be ``-1``
      (10× dilution → 10× lower D0).

    The R² values from the two fits are then compared. The headline
    finding from Boggy & Woolf (2010) is that the D0 curve is
    typically more linear than the Ct curve across instruments,
    chemistries, and template types — that's the empirical case for
    mechanistic quantification.

    Args:
        results_df: Per-well results. Must contain a column matching
            ``group_column`` and the columns ``D0`` and ``Ct``.
        dilution_factors: Explicit ``{group: dilution}`` mapping.
            Use this when group names don't reflect the dilution
            order. If ``None``, dilutions are auto-generated as
            ``dilution_factor**i`` for i = 0, 1, 2, …, matching the
            iteration order of ``groupby``.
        dilution_factor: Fold-dilution between consecutive groups
            (e.g., 2 for serial 1:2, 10 for serial 1:10). Only
            consulted when ``dilution_factors`` is None.
        group_column: Column to group on. Default ``'Group'``.

    Returns:
        Either an error dict (if fewer than 3 dilution points are
        available — a regression needs at least 3 points to be
        meaningful) or:

        - ``ct_analysis``: ``{slope, intercept, r2, efficiency,
          expected_slope, slope_error}``
        - ``d0_analysis``: ``{slope, intercept, r2, expected_slope,
          slope_error}``
        - ``comparison``: ``{ct_r2, d0_r2, better_linearity,
          r2_difference}``
        - ``data``: the per-group DataFrame used for the regression
          (passed forward to the plotting function).
    """
    # Group by replicate group
    grouped = results_df.groupby(group_column).agg({
        'Ct': ['mean', 'std'],
        'D0': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['Group', 'Ct_Mean', 'Ct_SD', 'D0_Mean', 'D0_SD']

    # Assign dilution factors
    if dilution_factors is None:
        # Auto-generate dilution factors (e.g., 1, 2, 4, 8 for 2-fold or 1, 10, 100, 1000 for 10-fold)
        dilution_factors = {row['Group']: dilution_factor**i for i, row in grouped.iterrows()}

    # Add dilution info
    grouped['Dilution'] = grouped['Group'].map(dilution_factors)
    grouped = grouped.dropna(subset=['Dilution']).sort_values('Dilution')

    # Ensure dilution values are float type for numpy operations
    grouped['Dilution'] = grouped['Dilution'].astype(float)

    if len(grouped) < 3:
        return {
            'error': 'Need at least 3 dilution points',
            'ct_analysis': None,
            'd0_analysis': None
        }

    # CT ANALYSIS: Standard curve approach
    # Plot Ct vs log10(1/dilution) = -log10(dilution)
    # For standard curve: Ct = -m * log10(quantity) + b
    # Efficiency = 10^(-1/m) - 1
    # Expected slope m ≈ -3.32 for 100% efficiency

    # Extract values as numpy float arrays
    dilution_values = np.array(grouped['Dilution'].values, dtype=float)
    log10_dilution = np.log10(dilution_values)
    ct_values = np.array(grouped['Ct_Mean'].values, dtype=float)

    # Regress Ct vs log10(dilution) - note: dilution increases means template decreases
    # So we use -log10(dilution) to get standard curve format (higher template = lower Ct)
    ct_slope, ct_intercept, ct_r2, _, _ = linregress(-log10_dilution, ct_values)

    # Efficiency from slope: E = 10^(-1/slope) - 1
    # For 100% efficiency, slope = -3.32 (doubling per cycle)
    ct_efficiency = (10 ** (-1 / ct_slope) - 1) * 100 if ct_slope != 0 else np.nan

    ct_analysis = {
        'slope': ct_slope,
        'intercept': ct_intercept,
        'r2': ct_r2 ** 2,
        'efficiency': ct_efficiency,
        'expected_slope': -3.32,  # For 100% efficiency
        'slope_error': abs(ct_slope - (-3.32))
    }

    # D0 ANALYSIS: log10(D0) vs log10(dilution)
    # For dilution series: log10(D0) should decrease linearly with log10(dilution)
    # Expected slope = -1 (10-fold dilution → 10-fold decrease in D0)
    log_d0 = np.log10(np.array(grouped['D0_Mean'].values, dtype=float))

    d0_slope, d0_intercept, d0_r2, _, _ = linregress(log10_dilution, log_d0)

    # Expected slope is -1 for perfect dilution series
    # (log10(D0) decreases by 1 for each log10 unit increase in dilution)
    expected_d0_slope = -1.0

    d0_analysis = {
        'slope': d0_slope,
        'intercept': d0_intercept,
        'r2': d0_r2 ** 2,
        'expected_slope': expected_d0_slope,
        'slope_error': abs(d0_slope - expected_d0_slope)
    }

    # COMPARISON
    comparison = {
        'ct_r2': ct_analysis['r2'],
        'd0_r2': d0_analysis['r2'],
        'better_linearity': 'D0' if d0_analysis['r2'] > ct_analysis['r2'] else 'Ct',
        'r2_difference': abs(d0_analysis['r2'] - ct_analysis['r2'])
    }

    return {
        'ct_analysis': ct_analysis,
        'd0_analysis': d0_analysis,
        'comparison': comparison,
        'data': grouped
    }


def plot_dilution_series_comparison(analysis_results: Dict) -> go.Figure:
    """Render side-by-side standard-curve plots for Ct and D0.

    Plotly figure with two subplots showing each method's regression
    line and replicate error bars on a shared log10(dilution) x-axis.
    Subplot titles include the R² and (for Ct) the back-calculated
    efficiency.

    Error bars are drawn manually as line traces rather than using
    Plotly's ``error_y`` because the latter doesn't render reliably
    in subplots across Plotly versions.

    Args:
        analysis_results: Output from ``analyze_dilution_series``.
            Must include the ``data`` DataFrame and both
            ``ct_analysis`` / ``d0_analysis`` regression dicts.

    Returns:
        Plotly Figure ready for ``st.plotly_chart`` display.
    """
    data = analysis_results['data']
    ct_analysis = analysis_results['ct_analysis']
    d0_analysis = analysis_results['d0_analysis']

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Ct Method (R² = {ct_analysis['r2']:.4f}, E = {ct_analysis['efficiency']:.1f}%)",
            f"D0 Method (R² = {d0_analysis['r2']:.4f})"
        ),
        x_title="Log10(Dilution Factor)"
    )

    # Use log10 for x-axis to match regression calculations
    # Ensure dilution values are float type for numpy operations
    log_dilution = np.log10(np.array(data['Dilution'].values, dtype=float))

    # Debug: Print error bar values to check they exist
    print(f"DEBUG - Ct SD values: {data['Ct_SD'].values}")
    print(f"DEBUG - D0 SD values: {data['D0_SD'].values}")

    # Plot Ct with EXPLICIT error bars (drawn as separate traces)
    ct_mean = data['Ct_Mean'].values
    ct_sd = data['Ct_SD'].values

    # Draw error bars as vertical lines
    for i in range(len(log_dilution)):
        # Vertical line from mean-SD to mean+SD
        fig.add_trace(
            go.Scatter(
                x=[log_dilution[i], log_dilution[i]],
                y=[ct_mean[i] - ct_sd[i], ct_mean[i] + ct_sd[i]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        # Top cap
        fig.add_trace(
            go.Scatter(
                x=[log_dilution[i] - 0.1, log_dilution[i] + 0.1],
                y=[ct_mean[i] + ct_sd[i], ct_mean[i] + ct_sd[i]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        # Bottom cap
        fig.add_trace(
            go.Scatter(
                x=[log_dilution[i] - 0.1, log_dilution[i] + 0.1],
                y=[ct_mean[i] - ct_sd[i], ct_mean[i] - ct_sd[i]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Now plot the data points on top
    fig.add_trace(
        go.Scatter(
            x=log_dilution,
            y=ct_mean,
            mode='markers',
            name='Ct (±SD)',
            marker=dict(size=3, color='blue', line=dict(width=1, color='darkblue'))
        ),
        row=1, col=1
    )

    # Ct regression line
    # ct_analysis slope is from -log10(dilution) regression (standard curve format)
    # So to plot: Ct = slope * (-log10_dilution) + intercept
    ct_fit = ct_analysis['slope'] * (-log_dilution) + ct_analysis['intercept']
    fig.add_trace(
        go.Scatter(
            x=log_dilution,
            y=ct_fit,
            mode='lines',
            name='Ct Fit',
            line=dict(color='blue', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )

    # Plot D0 with EXPLICIT error bars (drawn as separate traces)
    log_d0 = np.log10(data['D0_Mean'].values)
    log_d0_sd = data['D0_SD'] / (data['D0_Mean'] * np.log(10))  # Error propagation
    d0_error_bars = log_d0_sd.values if hasattr(log_d0_sd, 'values') else log_d0_sd

    # Draw error bars as vertical lines
    for i in range(len(log_dilution)):
        # Vertical line from mean-SD to mean+SD
        fig.add_trace(
            go.Scatter(
                x=[log_dilution[i], log_dilution[i]],
                y=[log_d0[i] - d0_error_bars[i], log_d0[i] + d0_error_bars[i]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        # Top cap
        fig.add_trace(
            go.Scatter(
                x=[log_dilution[i] - 0.1, log_dilution[i] + 0.1],
                y=[log_d0[i] + d0_error_bars[i], log_d0[i] + d0_error_bars[i]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        # Bottom cap
        fig.add_trace(
            go.Scatter(
                x=[log_dilution[i] - 0.1, log_dilution[i] + 0.1],
                y=[log_d0[i] - d0_error_bars[i], log_d0[i] - d0_error_bars[i]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )

    # Now plot the data points on top
    fig.add_trace(
        go.Scatter(
            x=log_dilution,
            y=log_d0,
            mode='markers',
            name='D0 (±SD)',
            marker=dict(size=3, color='red', line=dict(width=1, color='darkred'))
        ),
        row=1, col=2
    )

    # D0 regression line
    # d0_analysis slope is from log10(dilution) vs log10(D0)
    # So to plot: log10(D0) = slope * log10_dilution + intercept
    d0_fit = d0_analysis['slope'] * log_dilution + d0_analysis['intercept']
    fig.add_trace(
        go.Scatter(
            x=log_dilution,
            y=d0_fit,
            mode='lines',
            name='D0 Fit',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    # Update axes
    fig.update_yaxes(title_text="Ct", row=1, col=1)
    fig.update_yaxes(title_text="Log10(D0)", row=1, col=2)

    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Dilution Series Analysis: Ct vs D0 Quantification"
    )

    return fig


def compare_precision(stats_df: pd.DataFrame, efficiency: float = 0.95) -> pd.DataFrame:
    """Translate Ct CV% into a concentration-equivalent CV% for fair D0 comparison.

    A direct Ct CV% vs D0 CV% comparison is misleading: Ct lives on
    a log scale (each cycle delay halves quantity at 100% efficiency),
    so a 1% CV in Ct corresponds to a much larger CV in template
    quantity. To compare the precision of the two quantification
    methods on the **same scale** (template concentration), we have
    to push Ct's variability through the relationship
    ``quantity ∝ E^Ct``.

    First-order error propagation through ``q = E^Ct``:

        d(ln q) / d(Ct) = ln(E)
        ⇒ CV(q) ≈ |ln(E)| × SD(Ct)

    Numerically, with E = 1 + efficiency (default efficiency = 0.95
    → E = 1.95), ``ln(E) ≈ 0.668``, so a single-cycle Ct SD becomes
    a ~67% concentration CV. That's the apples-to-apples number to
    compare against D0's CV%.

    Args:
        stats_df: Output from ``calculate_replicate_stats``. Must
            include ``Ct_SD``, ``Ct_CV``, ``D0_CV``.
        efficiency: PCR efficiency in fractional units (0–1). Default
            0.95 (= 95%) is the typical fitted efficiency from
            standard-curve dilution series. Pass ``ct_analysis['efficiency']/100``
            from a real fit when available.

    Returns:
        Side-by-side DataFrame with original columns plus
        ``Ct_Conc_CV`` (the converted Ct CV%), ``Better_Precision``
        (string ``'Ct'`` or ``'D0'``), and ``CV_Difference`` (a
        positive value means D0 wins).
    """
    comparison = stats_df[['Group', 'N', 'Ct_Mean', 'Ct_SD', 'Ct_CV', 'D0_Mean', 'D0_CV']].copy()

    # Convert Ct variation to equivalent concentration CV%
    # For E^Ct, the CV% ≈ |ln(1+E) × Ct_SD| × 100
    # Simplified: CV% ≈ ln(1+E) × Ct_SD × 100
    E = efficiency
    comparison['Ct_Conc_CV'] = np.abs(np.log(1 + E) * comparison['Ct_SD']) * 100

    # Determine which method has better (lower) CV on concentration
    comparison['Better_Precision'] = comparison.apply(
        lambda row: 'D0' if row['D0_CV'] < row['Ct_Conc_CV'] else 'Ct',
        axis=1
    )

    comparison['CV_Difference'] = comparison['Ct_Conc_CV'] - comparison['D0_CV']

    return comparison


# Example usage
if __name__ == "__main__":
    # Simulate some results
    np.random.seed(42)

    samples = []
    for i in range(6):  # 6 dilution levels
        for rep in range(3):  # 3 replicates each
            true_d0 = 1e-6 * (0.5 ** i)  # 2-fold dilutions
            true_ct = 20 + i  # Should increase by 1 per dilution

            samples.append({
                'Sample': f'F{i+1}.{rep+1}',
                'Group': f'F{i+1}',
                'D0': true_d0 * (1 + np.random.normal(0, 0.05)),  # 5% CV
                'Ct': true_ct + np.random.normal(0, 0.3),  # 0.3 cycle SD
                'R2': 0.999
            })

    results_df = pd.DataFrame(samples)

    print("="*80)
    print("REPLICATE STATISTICS")
    print("="*80)
    stats = calculate_replicate_stats(results_df)
    print(stats.to_string(index=False))

    print("\n" + "="*80)
    print("PRECISION COMPARISON")
    print("="*80)
    precision = compare_precision(stats)
    print(precision.to_string(index=False))

    print("\n" + "="*80)
    print("DILUTION SERIES ANALYSIS")
    print("="*80)
    dilution_analysis = analyze_dilution_series(results_df)
    print(f"\nCt Analysis:")
    print(f"  R² = {dilution_analysis['ct_analysis']['r2']:.4f}")
    print(f"  Efficiency = {dilution_analysis['ct_analysis']['efficiency']:.1f}%")
    print(f"\nD0 Analysis:")
    print(f"  R² = {dilution_analysis['d0_analysis']['r2']:.4f}")
    print(f"  Slope = {dilution_analysis['d0_analysis']['slope']:.4f} (expected: {dilution_analysis['d0_analysis']['expected_slope']:.4f})")
    print(f"\n{dilution_analysis['comparison']['better_linearity']} has better linearity")
