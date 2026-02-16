# MAK2 Optimizer Tier Benchmark Report

Generated: 2026-02-16 09:07:17

Dataset: Rutledge.csv

R² threshold: 0.999


## Summary

| Configuration | Passed | Failed | Total Time | Avg Time/Sample |
|---|---|---|---|---|
| all_tiers | 120/120 | ✅ | 52.2s | 0.44s |
| no_tier1.5 | 120/120 | ✅ | 52.1s | 0.43s |
| no_tier2 | 119/120 | ❌ (1 failed) | 46.4s | 0.39s |
| no_tier2.5 | 120/120 | ✅ | 76.2s | 0.64s |
| no_tier3 | 110/120 | ❌ (10 failed) | 20.9s | 0.17s |
| no_tier4 | 120/120 | ✅ | 34.1s | 0.28s |
| tier1_only | 89/120 | ❌ (31 failed) | 3.5s | 0.03s |
| no_retry_tiers | 119/120 | ❌ (1 failed) | 48.5s | 0.40s |
| no_global | 110/120 | ❌ (10 failed) | 20.5s | 0.17s |
| minimal | 89/120 | ❌ (31 failed) | 3.7s | 0.03s |

## Baseline Tier Details

| Tier | Fired | Improved | Avg Time | Total Time |
|---|---|---|---|---|
| tier1_multistart | 120/120 | 120 | 0.029s | 3.4s |
| tier1.5_residual_patterns | 31/120 | 11 | 0.001s | 0.1s |
| tier2_ssr_retry | 4/120 | 0 | 0.001s | 0.1s |
| tier2.5_adaptive_fallback | 26/120 | 19 | 0.006s | 0.7s |
| tier3_differential_evolution | 15/120 | 15 | 0.246s | 29.5s |
| tier4_overshoot_refit | 33/120 | 22 | 0.153s | 18.4s |

## Failed Samples by Configuration


### all_tiers: All passed ✅


### no_tier1.5: All passed ✅


### no_tier2

- **X5.R4.1**: R²=0.998962

### no_tier2.5: All passed ✅


### no_tier3

- **X5.R3.1**: R²=0.997743
- **X5.R4.1**: R²=0.995094
- **X5.R4.3**: R²=0.998603
- **X5.R5.3**: R²=0.997347
- **X5.R5.4**: R²=0.998189
- **X6.R1.1**: R²=0.998601
- **X6.R1.2**: R²=0.998882
- **X6.R1.4**: R²=0.998941
- **X6.R2.4**: R²=0.998723
- **X6.R4.2**: R²=0.998730

### no_tier4: All passed ✅


### tier1_only

- **X5.R1.1**: R²=0.996344
- **X5.R1.2**: R²=0.990860
- **X5.R1.3**: R²=0.986036
- **X5.R1.4**: R²=0.994516
- **X5.R2.1**: R²=0.996841
- **X5.R2.2**: R²=0.998982
- **X5.R2.3**: R²=0.991354
- **X5.R2.4**: R²=0.961504
- **X5.R3.1**: R²=0.997743
- **X5.R3.2**: R²=0.990074
- **X5.R3.4**: R²=0.993265
- **X5.R4.1**: R²=0.995094
- **X5.R4.2**: R²=0.992889
- **X5.R4.3**: R²=0.992310
- **X5.R4.4**: R²=0.998696
- **X5.R5.1**: R²=0.991609
- **X5.R5.2**: R²=0.981385
- **X5.R5.3**: R²=0.996381
- **X5.R5.4**: R²=0.992877
- **X6.R1.1**: R²=0.997582
- **X6.R1.2**: R²=0.998852
- **X6.R1.3**: R²=0.997288
- **X6.R1.4**: R²=0.998941
- **X6.R2.2**: R²=0.998999
- **X6.R2.4**: R²=0.998723
- **X6.R3.2**: R²=0.998939
- **X6.R4.2**: R²=0.998730
- **X6.R4.3**: R²=0.998710
- **X6.R4.4**: R²=0.997865
- **X6.R5.3**: R²=0.998926
- **X6.R5.4**: R²=0.997208

### no_retry_tiers

- **X5.R4.1**: R²=0.998962

### no_global

- **X5.R3.1**: R²=0.997743
- **X5.R4.1**: R²=0.995094
- **X5.R4.3**: R²=0.998603
- **X5.R5.3**: R²=0.997347
- **X5.R5.4**: R²=0.998189
- **X6.R1.1**: R²=0.998601
- **X6.R1.2**: R²=0.998882
- **X6.R1.4**: R²=0.998941
- **X6.R2.4**: R²=0.998723
- **X6.R4.2**: R²=0.998730

### minimal

- **X5.R1.1**: R²=0.996344
- **X5.R1.2**: R²=0.990860
- **X5.R1.3**: R²=0.986036
- **X5.R1.4**: R²=0.994516
- **X5.R2.1**: R²=0.996841
- **X5.R2.2**: R²=0.998982
- **X5.R2.3**: R²=0.991354
- **X5.R2.4**: R²=0.961504
- **X5.R3.1**: R²=0.997743
- **X5.R3.2**: R²=0.990074
- **X5.R3.4**: R²=0.993265
- **X5.R4.1**: R²=0.995094
- **X5.R4.2**: R²=0.992889
- **X5.R4.3**: R²=0.992310
- **X5.R4.4**: R²=0.998696
- **X5.R5.1**: R²=0.991609
- **X5.R5.2**: R²=0.981385
- **X5.R5.3**: R²=0.996381
- **X5.R5.4**: R²=0.992877
- **X6.R1.1**: R²=0.997582
- **X6.R1.2**: R²=0.998852
- **X6.R1.3**: R²=0.997288
- **X6.R1.4**: R²=0.998941
- **X6.R2.2**: R²=0.998999
- **X6.R2.4**: R²=0.998723
- **X6.R3.2**: R²=0.998939
- **X6.R4.2**: R²=0.998730
- **X6.R4.3**: R²=0.998710
- **X6.R4.4**: R²=0.997865
- **X6.R5.3**: R²=0.998926
- **X6.R5.4**: R²=0.997208

## Recommendations

Based on the ablation results:

- **no_tier1.5**: Same pass rate, 1.0x speed change → ✅ tier may be removable
- **no_tier2**: Lost 1 samples → ❌ tier is needed
- **no_tier2.5**: Same pass rate, 0.7x speed change → ✅ tier may be removable
- **no_tier3**: Lost 10 samples → ❌ tier is needed
- **no_tier4**: Same pass rate, 1.5x speed change → ✅ tier may be removable
- **tier1_only**: Lost 31 samples → ❌ tier is needed
- **no_retry_tiers**: Lost 1 samples → ❌ tier is needed
- **no_global**: Lost 10 samples → ❌ tier is needed
- **minimal**: Lost 31 samples → ❌ tier is needed