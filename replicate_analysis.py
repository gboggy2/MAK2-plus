"""
Replicate and dilution series analysis for qPCR data.

Provides statistical comparison between Ct and D0 quantification methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_sample_groups(sample_names: List[str], pattern: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Group samples into replicates based on naming pattern.

    Supports common patterns:
    - F1.1, F1.2, F1.3 → group "F1"
    - Sample_A_rep1, Sample_A_rep2 → group "Sample_A"
    - Custom pattern matching

    Parameters
    ----------
    sample_names : list of str
        Sample names to group
    pattern : str, optional
        Pattern for grouping (e.g., "." splits on last dot)

    Returns
    -------
    groups : dict
        {group_name: [sample1, sample2, ...]}
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
    """
    Calculate statistics for replicate groups.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: Sample, Group, D0, Ct, etc.
    group_column : str
        Column name for grouping
    metrics : list of str
        Metrics to calculate stats for (e.g., ['D0', 'Ct'])

    Returns
    -------
    stats_df : pd.DataFrame
        Statistics by group: mean, SD, CV%, min, max
    """
    stats = []

    for group, group_df in results_df.groupby(group_column):
        stat_row = {'Group': group, 'N': len(group_df)}

        for metric in metrics:
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
    group_column: str = 'Group'
) -> Dict:
    """
    Analyze dilution series to assess linearity and efficiency.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with D0 and Ct values
    dilution_factors : dict, optional
        {group_name: dilution_factor} e.g., {'F1': 1, 'F2': 2, 'F3': 4}
        If None, assumes groups are ordered (1, 2, 4, 8, ...)
    group_column : str
        Column name for grouping

    Returns
    -------
    analysis : dict
        Contains:
        - 'ct_analysis': {slope, intercept, r2, efficiency}
        - 'd0_analysis': {slope, intercept, r2}
        - 'comparison': Which method has better linearity
    """
    # Group by replicate group
    grouped = results_df.groupby(group_column).agg({
        'Ct': ['mean', 'std'],
        'D0': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['Group', 'Ct_Mean', 'Ct_SD', 'D0_Mean', 'D0_SD']

    # Assign dilution factors
    if dilution_factors is None:
        # Assume 2-fold dilutions
        dilution_factors = {row['Group']: 2**i for i, row in grouped.iterrows()}

    # Add dilution info
    grouped['Dilution'] = grouped['Group'].map(dilution_factors)
    grouped = grouped.dropna(subset=['Dilution']).sort_values('Dilution')

    if len(grouped) < 3:
        return {
            'error': 'Need at least 3 dilution points',
            'ct_analysis': None,
            'd0_analysis': None
        }

    # CT ANALYSIS: ΔCt vs log2(dilution)
    # Expected: ΔCt = -log2(dilution) * (1/efficiency)
    # Slope = -1 for 100% efficiency
    log_dilution = np.log2(grouped['Dilution'].values)
    ct_values = grouped['Ct_Mean'].values

    # Regress Ct vs log2(dilution)
    ct_slope, ct_intercept, ct_r2, _, _ = linregress(log_dilution, ct_values)

    # Efficiency from slope: E = 2^(-1/slope) - 1
    ct_efficiency = (2 ** (-1 / ct_slope) - 1) * 100 if ct_slope != 0 else np.nan

    ct_analysis = {
        'slope': ct_slope,
        'intercept': ct_intercept,
        'r2': ct_r2 ** 2,
        'efficiency': ct_efficiency,
        'expected_slope': -1.0,
        'slope_error': abs(ct_slope - (-1.0))
    }

    # D0 ANALYSIS: log10(D0) vs log2(dilution)
    # Expected: perfect inverse relationship
    # Slope = -1 (doubling dilution halves D0)
    log_d0 = np.log10(grouped['D0_Mean'].values)

    d0_slope, d0_intercept, d0_r2, _, _ = linregress(log_dilution, log_d0)

    # Convert slope to efficiency-like metric
    # D0 should halve with each dilution → slope ≈ -0.301 (log10(0.5))
    expected_d0_slope = np.log10(0.5)  # -0.301

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
    """
    Create publication-quality plot comparing Ct vs D0 for dilution series.

    Parameters
    ----------
    analysis_results : dict
        Output from analyze_dilution_series()

    Returns
    -------
    fig : plotly Figure
        Comparison plot
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
        x_title="Log2(Dilution Factor)"
    )

    log_dilution = np.log2(data['Dilution'].values)

    # Plot Ct
    fig.add_trace(
        go.Scatter(
            x=log_dilution,
            y=data['Ct_Mean'],
            error_y=dict(type='data', array=data['Ct_SD']),
            mode='markers',
            name='Ct',
            marker=dict(size=10, color='blue')
        ),
        row=1, col=1
    )

    # Ct regression line
    ct_fit = ct_analysis['slope'] * log_dilution + ct_analysis['intercept']
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

    # Plot D0
    log_d0 = np.log10(data['D0_Mean'].values)
    log_d0_sd = data['D0_SD'] / (data['D0_Mean'] * np.log(10))  # Error propagation

    fig.add_trace(
        go.Scatter(
            x=log_dilution,
            y=log_d0,
            error_y=dict(type='data', array=log_d0_sd),
            mode='markers',
            name='D0',
            marker=dict(size=10, color='red')
        ),
        row=1, col=2
    )

    # D0 regression line
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


def compare_precision(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare precision (CV%) between Ct and D0 methods.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output from calculate_replicate_stats()

    Returns
    -------
    comparison_df : pd.DataFrame
        Side-by-side comparison with winner column
    """
    comparison = stats_df[['Group', 'N', 'Ct_Mean', 'Ct_CV', 'D0_Mean', 'D0_CV']].copy()

    # Determine which method has better (lower) CV
    comparison['Better_Precision'] = comparison.apply(
        lambda row: 'D0' if row['D0_CV'] < row['Ct_CV'] else 'Ct',
        axis=1
    )

    comparison['CV_Difference'] = comparison['Ct_CV'] - comparison['D0_CV']

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
