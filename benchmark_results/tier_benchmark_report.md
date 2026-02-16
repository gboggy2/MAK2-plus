# MAK2 Optimizer Tier Benchmark Report

Generated: 2026-02-16 10:50:16

Dataset: Rutledge.csv

R² threshold: 0.999


## Summary

| Configuration | Passed | Failed | Total Time | Avg Time/Sample |
|---|---|---|---|---|
| all_tiers | 120/120 | ✅ | 72.2s | 0.60s |

## Baseline Tier Details

| Tier | Fired | Improved | Avg Time | Total Time |
|---|---|---|---|---|
| tier1_multistart | 120/120 | 120 | 0.039s | 4.7s |
| tier1.5_residual_patterns | 31/120 | 11 | 0.002s | 0.2s |
| tier2_ssr_retry | 4/120 | 0 | 0.001s | 0.1s |
| tier2.5_adaptive_fallback | 26/120 | 19 | 0.008s | 0.9s |
| tier3_differential_evolution | 15/120 | 15 | 0.340s | 40.7s |
| tier4_overshoot_refit | 33/120 | 22 | 0.213s | 25.6s |

## Failed Samples by Configuration


### all_tiers: All passed ✅


## Recommendations

Based on the ablation results:
