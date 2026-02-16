#!/usr/bin/env python3
"""
Benchmark and ablation analysis for MAK2 optimizer tiers.

Runs the Rutledge dataset (120 samples) through the optimizer with different
tier configurations to determine which tiers are necessary for achieving
R² ≥ 0.999 on all samples, and how much time each tier contributes.

Usage:
    python benchmark_tiers.py                    # Full benchmark + ablation
    python benchmark_tiers.py --baseline-only    # Just baseline (all tiers enabled)
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mak2_model import MAK2Model
from optimizer import MAK2Optimizer
from data_processing import detect_no_signal_samples


# ── Configuration ──────────────────────────────────────────────────────────────

DATA_FILE = project_root / "example_data" / "Rutledge.csv"
R2_THRESHOLD = 0.999
OUTPUT_DIR = project_root / "benchmark_results"

# Tier names used in disabled_tiers sets
ALL_TIERS = ['tier1.5', 'tier2', 'tier2.5', 'tier3', 'tier4']

# Ablation configurations: name → set of disabled tiers
ABLATION_CONFIGS = {
    'all_tiers':          set(),                           # Baseline: everything enabled
    'no_tier1.5':         {'tier1.5'},                     # Disable residual patterns
    'no_tier2':           {'tier2'},                       # Disable SSR retry
    'no_tier2.5':         {'tier2.5'},                     # Disable adaptive fallback
    'no_tier3':           {'tier3'},                       # Disable differential evolution
    'no_tier4':           {'tier4'},                       # Disable overshoot refit
    'tier1_only':         {'tier1.5', 'tier2', 'tier2.5', 'tier3', 'tier4'},  # Only multi-start
    'no_retry_tiers':     {'tier1.5', 'tier2'},            # No pattern/SSR retry
    'no_global':          {'tier3'},                       # No DE
    'minimal':            {'tier1.5', 'tier2', 'tier2.5', 'tier3'},  # Tier 1 + Tier 4 only
}


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_rutledge_data():
    """Load and parse Rutledge CSV dataset."""
    df = pd.read_csv(DATA_FILE)

    # Determine structure: first column is cycles, rest are samples
    cycle_col = df.columns[0]
    sample_cols = df.columns[1:]
    cycles = df[cycle_col].values.astype(float)

    all_samples = {}
    for col in sample_cols:
        all_samples[col] = df[col].values.astype(float)

    return cycles, all_samples


# ── Benchmark Runner ───────────────────────────────────────────────────────────

def run_single_sample(cycles, fluorescence, disabled_tiers=None):
    """
    Fit a single sample and return results + tier log.

    Returns dict with: r2, params, tier_log, total_time, success
    """
    model = MAK2Model()
    optimizer = MAK2Optimizer(model)

    start = time.perf_counter()
    try:
        params = optimizer.fit(
            cycles, fluorescence,
            cycles_after_max=3,
            auto_truncate=True,
            r2_threshold=R2_THRESHOLD,
            verbose=False,
            fix_background=True,
            disabled_tiers=disabled_tiers,
        )
        elapsed = time.perf_counter() - start
        r2 = optimizer.metrics['r_squared']

        return {
            'r2': r2,
            'params': params,
            'tier_log': optimizer.tier_log,
            'total_time': elapsed,
            'success': r2 >= R2_THRESHOLD,
            'error': None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'r2': 0.0,
            'params': None,
            'tier_log': optimizer.tier_log,
            'total_time': elapsed,
            'success': False,
            'error': str(e),
        }


def run_benchmark(cycles, valid_samples, disabled_tiers=None, label="baseline"):
    """
    Run all samples through optimizer with given tier configuration.

    Returns list of per-sample result dicts.
    """
    results = []
    n_total = len(valid_samples)

    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"  Disabled tiers: {disabled_tiers or 'none'}")
    print(f"  Samples: {n_total}")
    print(f"{'='*70}")

    for i, (name, fluor) in enumerate(valid_samples.items()):
        result = run_single_sample(cycles, fluor, disabled_tiers=disabled_tiers)
        result['sample_name'] = name
        results.append(result)

        status = "✓" if result['success'] else "✗"
        r2_str = f"{result['r2']:.6f}" if result['r2'] else "FAIL"

        # Progress indicator every 20 samples
        if (i + 1) % 20 == 0 or not result['success']:
            print(f"  [{i+1:3d}/{n_total}] {status} {name}: R²={r2_str} ({result['total_time']:.2f}s)")

    # Summary
    n_pass = sum(1 for r in results if r['success'])
    n_fail = n_total - n_pass
    total_time = sum(r['total_time'] for r in results)
    avg_time = total_time / n_total if n_total > 0 else 0

    print(f"\n  Results: {n_pass}/{n_total} passed (R² ≥ {R2_THRESHOLD})")
    if n_fail > 0:
        print(f"  ❌ FAILURES:")
        for r in results:
            if not r['success']:
                print(f"     {r['sample_name']}: R²={r['r2']:.6f}")
    print(f"  Total time: {total_time:.1f}s, Avg: {avg_time:.2f}s/sample")

    return results


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze_tier_stats(results, label="baseline"):
    """Analyze tier firing rates and timing from a benchmark run."""
    tier_stats = defaultdict(lambda: {
        'fired_count': 0,
        'improved_count': 0,
        'total_time': 0.0,
        'times': [],
    })

    for r in results:
        for entry in r['tier_log']:
            tier = entry['tier']
            tier_stats[tier]['total_time'] += entry['time_seconds']
            tier_stats[tier]['times'].append(entry['time_seconds'])
            if entry['fired']:
                tier_stats[tier]['fired_count'] += 1
            if entry.get('improved', False):
                tier_stats[tier]['improved_count'] += 1

    n_samples = len(results)
    print(f"\n{'─'*70}")
    print(f"  Tier Statistics for: {label}")
    print(f"{'─'*70}")
    print(f"  {'Tier':<30s} {'Fired':>7s} {'Improved':>9s} {'Avg Time':>10s} {'Total':>8s}")
    print(f"  {'─'*30} {'─'*7} {'─'*9} {'─'*10} {'─'*8}")

    for tier_name in ['tier1_multistart', 'tier1.5_residual_patterns', 'tier2_ssr_retry',
                      'tier2.5_adaptive_fallback', 'tier3_differential_evolution',
                      'tier4_overshoot_refit']:
        stats = tier_stats.get(tier_name)
        if stats is None:
            continue
        fired = stats['fired_count']
        improved = stats['improved_count']
        avg_t = stats['total_time'] / n_samples if n_samples > 0 else 0
        total_t = stats['total_time']
        pct_fired = f"{fired}/{n_samples}" if tier_name != 'tier1_multistart' else f"{n_samples}/{n_samples}"

        print(f"  {tier_name:<30s} {pct_fired:>7s} {improved:>6d}    {avg_t:>8.3f}s {total_t:>7.1f}s")

    return dict(tier_stats)


def generate_report(all_results, output_path):
    """Generate a markdown report summarizing all benchmark runs."""
    lines = []
    lines.append("# MAK2 Optimizer Tier Benchmark Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Dataset: Rutledge.csv\n")
    lines.append(f"R² threshold: {R2_THRESHOLD}\n")

    # Summary table
    lines.append("\n## Summary\n")
    lines.append("| Configuration | Passed | Failed | Total Time | Avg Time/Sample |")
    lines.append("|---|---|---|---|---|")

    for label, results in all_results.items():
        n_pass = sum(1 for r in results if r['success'])
        n_total = len(results)
        n_fail = n_total - n_pass
        total_t = sum(r['total_time'] for r in results)
        avg_t = total_t / n_total if n_total > 0 else 0
        status = "✅" if n_fail == 0 else f"❌ ({n_fail} failed)"
        lines.append(f"| {label} | {n_pass}/{n_total} | {status} | {total_t:.1f}s | {avg_t:.2f}s |")

    # Tier firing details for baseline
    if 'all_tiers' in all_results:
        lines.append("\n## Baseline Tier Details\n")
        lines.append("| Tier | Fired | Improved | Avg Time | Total Time |")
        lines.append("|---|---|---|---|---|")

        baseline_results = all_results['all_tiers']
        n_samples = len(baseline_results)
        tier_stats = defaultdict(lambda: {
            'fired_count': 0, 'improved_count': 0, 'total_time': 0.0,
        })

        for r in baseline_results:
            for entry in r['tier_log']:
                tier = entry['tier']
                tier_stats[tier]['total_time'] += entry['time_seconds']
                if entry['fired']:
                    tier_stats[tier]['fired_count'] += 1
                if entry.get('improved', False):
                    tier_stats[tier]['improved_count'] += 1

        for tier_name in ['tier1_multistart', 'tier1.5_residual_patterns', 'tier2_ssr_retry',
                          'tier2.5_adaptive_fallback', 'tier3_differential_evolution',
                          'tier4_overshoot_refit']:
            stats = tier_stats.get(tier_name)
            if stats is None:
                continue
            fired = stats['fired_count']
            improved = stats['improved_count']
            avg_t = stats['total_time'] / n_samples
            total_t = stats['total_time']
            lines.append(f"| {tier_name} | {fired}/{n_samples} | {improved} | {avg_t:.3f}s | {total_t:.1f}s |")

    # Failed samples detail
    lines.append("\n## Failed Samples by Configuration\n")
    for label, results in all_results.items():
        failures = [r for r in results if not r['success']]
        if failures:
            lines.append(f"\n### {label}\n")
            for r in failures:
                lines.append(f"- **{r['sample_name']}**: R²={r['r2']:.6f}")
        else:
            lines.append(f"\n### {label}: All passed ✅\n")

    # Recommendations
    lines.append("\n## Recommendations\n")
    lines.append("Based on the ablation results:\n")

    baseline_pass = sum(1 for r in all_results.get('all_tiers', []) if r['success'])
    for label, results in all_results.items():
        if label == 'all_tiers':
            continue
        n_pass = sum(1 for r in results if r['success'])
        total_t = sum(r['total_time'] for r in results)
        baseline_t = sum(r['total_time'] for r in all_results.get('all_tiers', []))

        if n_pass == baseline_pass:
            speedup = baseline_t / total_t if total_t > 0 else 0
            lines.append(f"- **{label}**: Same pass rate, {speedup:.1f}x speed change → ✅ tier may be removable")
        else:
            lines.append(f"- **{label}**: Lost {baseline_pass - n_pass} samples → ❌ tier is needed")

    report_text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n📄 Report saved to: {output_path}")
    return report_text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    baseline_only = '--baseline-only' in sys.argv

    print("="*70)
    print("  MAK2 Optimizer Tier Benchmark")
    print("="*70)

    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    cycles, all_samples = load_rutledge_data()
    print(f"  Loaded {len(all_samples)} samples, {len(cycles)} cycles each")

    # Filter out no-signal samples
    print("\nDetecting no-signal samples...")
    valid_samples, no_signal, plate_stats = detect_no_signal_samples(
        cycles, all_samples, verbose=False
    )
    print(f"  {len(valid_samples)} valid samples, {len(no_signal)} no-signal")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Run baseline
    all_results = {}
    all_results['all_tiers'] = run_benchmark(cycles, valid_samples, label='all_tiers')
    baseline_stats = analyze_tier_stats(all_results['all_tiers'], 'all_tiers')

    if not baseline_only:
        # Run ablation tests
        print(f"\n\n{'#'*70}")
        print(f"  ABLATION ANALYSIS")
        print(f"{'#'*70}")

        for config_name, disabled in ABLATION_CONFIGS.items():
            if config_name == 'all_tiers':
                continue  # Already ran baseline
            all_results[config_name] = run_benchmark(
                cycles, valid_samples,
                disabled_tiers=disabled,
                label=config_name
            )
            analyze_tier_stats(all_results[config_name], config_name)

    # Generate report
    report_path = OUTPUT_DIR / "tier_benchmark_report.md"
    generate_report(all_results, report_path)

    # Also save raw data as JSON for further analysis
    json_path = OUTPUT_DIR / "tier_benchmark_data.json"
    json_data = {}
    for label, results in all_results.items():
        json_data[label] = {
            'n_pass': sum(1 for r in results if r['success']),
            'n_total': len(results),
            'total_time': sum(r['total_time'] for r in results),
            'samples': [
                {
                    'name': r['sample_name'],
                    'r2': r['r2'],
                    'success': r['success'],
                    'time': r['total_time'],
                    'tier_log': r['tier_log'],
                    'error': r['error'],
                }
                for r in results
            ]
        }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"📊 Raw data saved to: {json_path}")

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*70}")
    baseline_n_pass = sum(1 for r in all_results['all_tiers'] if r['success'])
    baseline_n_total = len(all_results['all_tiers'])
    print(f"  Baseline: {baseline_n_pass}/{baseline_n_total} passed")

    if not baseline_only:
        print(f"\n  Ablation results:")
        for config_name, results in all_results.items():
            if config_name == 'all_tiers':
                continue
            n_pass = sum(1 for r in results if r['success'])
            total_t = sum(r['total_time'] for r in results)
            emoji = "✅" if n_pass == baseline_n_pass else "❌"
            print(f"    {emoji} {config_name:<25s}: {n_pass}/{baseline_n_total} passed, {total_t:.1f}s")


if __name__ == '__main__':
    main()
