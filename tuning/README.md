# MAK2 quality-gate tuning

Tooling for tuning the per-well quality-gate thresholds in
``config.QualityGateConfig`` against the PCRedux labelled curve
dataset.

## Goal

Find the gate parameter values that maximise agreement between
MAK2+'s per-well status verdicts (PASS / FAIL / INDETERMINATE) and
the rater-consensus labels in the PCRedux dataset (`y` / `n` / `a`).

The gates themselves stay rule-based — same code, same architecture
— this is *parameter optimisation*, not ML model training.

## Scripts

| Script | Purpose |
|---|---|
| `score_gates.py` | Score one ``QualityGateConfig`` against PCRedux. Fits every curve, then runs ``run_quality_gates`` with that config and reports confusion matrix + balanced accuracy. Used both as the optimiser's objective function and as a standalone diagnostic. |
| `tune_gates.py` | Outer optimisation loop. Differential-evolution search over the 10-knob parameter space, with asymmetric cost (false PASS hurts more than false FAIL). Writes best config + full trial history to `results/`. |

## Quick start

```bash
# 1. Score the current DEFAULT_GATES against PCRedux
python tuning/score_gates.py

# 2. Run the optimiser (DE; default 30 generations × popsize 15 = ~450 trials)
python tuning/tune_gates.py

# 3. Compare a custom config (e.g. one of the produced best.json files)
python -c "
import json
from config import QualityGateConfig
from tuning.score_gates import score_config, _build_cache, LABELS_CSV, CURVES_CSV
import pandas as pd
labels = pd.read_csv(LABELS_CSV)
curves = pd.read_csv(CURVES_CSV)
cycles = curves[curves['curve_id']==labels['curve_id'].iloc[0]]['cycle'].to_numpy(float)
import pickle
with open('tuning/_score_cache.pkl','rb') as f: cache = pickle.load(f)
gates = QualityGateConfig(**json.load(open('tuning/results/<timestamp>-best.json')))
result = score_config(gates, cache, labels, cycles)
print(f'BA = {result[\"balanced_accuracy\"]:.3f}')
"
```

## Performance

Fitting the full PCRedux subset (1023 curves) takes ~10 minutes on a
modern laptop. The fit results are cached to `_score_cache.pkl`, so
subsequent tuning trials are *much* faster — gate scoring on a
cached fit takes ~50ms per curve (≈ 1 min per full evaluation of the
1023-curve dataset, ≈ 7 hours for a 450-trial DE run with 1
worker).

For faster iteration during development, restrict the dataset:

```python
# In score_gates.py main(), add: labels = labels.head(200)
```

200 curves cover ~80% of the failure modes in the full set and
score in ~10s per trial.

## Class balance caveat

The PCRedux subset is heavily skewed:

|  | Count | Fraction |
|---|---|---|
| `n` (negative) | 891 | 87% |
| `y` (positive) | 113 | 11% |
| `a` (ambiguous) | 19 | 2% |

Implications for tuning:

1. **Don't use raw accuracy** — a "predict everything `n`" classifier
   scores 87% on raw accuracy. Use balanced accuracy or F1-weighted.
2. **Ambiguous-class tuning is unreliable** — only 19 examples;
   noise dominates.
3. **Positive-class tuning is also light** — 113 examples across 5
   sub-datasets. Cross-instrument generalisation can't be
   convincingly established from PCRedux alone; supplement with real
   ABI plates as they become available.

See `tests/fixtures/pcredux/README.md` for the underlying dataset
caveats.

## Search space

The 10 knobs in ``QualityGateConfig`` are searched over the bounds
defined at the top of `tune_gates.py`. Bounds bracket the current
defaults by roughly 2× in each direction; physically-nonsensical
regions (R² floors below 0.5, fit windows below 5 cycles, etc.) are
excluded.

To restrict tuning to a subset of knobs (e.g. only Gate 0's R²
floors), edit `SEARCH_SPACE` in `tune_gates.py` and remove the rows
you don't want optimised.

## Asymmetric cost

The default objective penalises false PASS (predicting `y` when the
expert label was `n`) at 2× the rate of false FAIL (predicting `n`
when the label was `y`). Rationale: a missed bad well that
propagates downstream is worse for users than a missed good well
they can manually inspect.

Override via CLI:

```bash
python tuning/tune_gates.py --false-pass-weight 5.0 --false-fail-weight 1.0
```

## Output

Each tuning run writes:

  - `results/<timestamp>-best.json` — the winning gate config
  - `results/<timestamp>-history.csv` — every trial's parameters +
    cost + balanced accuracy

The `results/` directory is **not** committed by default (the
artefacts are reproducible from the cached fits + the script). Add to
`.gitignore` if needed.

## Landing a tuned config

Once a winning config validates well against PCRedux *and* against
the existing MAK2 regression fixtures, update the defaults in
`config.QualityGateConfig` to match. Then:

1. Re-run the Phase 0.5 regression test — its reference fixture was
   captured under the old defaults, so expect status flips on some
   wells.
2. Re-capture `tests/fixtures/boggy_reference.json` via
   `tests/fixtures/build_fixtures.py`.
3. Commit the dataclass update + the new reference fixture in one
   commit, with a clear message documenting the BA improvement on
   PCRedux and the diff against Boggy.
