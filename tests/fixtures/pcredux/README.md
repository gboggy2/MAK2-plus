# PCRedux dataset fixtures

Labelled qPCR curves from **Rödiger et al., bioRxiv 2021**
([doi:10.1101/2021.03.31.437921](https://doi.org/10.1101/2021.03.31.437921)),
used by MAK2+ to tune the quality-gate thresholds against expert
classification labels.

## Files

| File | Source / generated | What |
|---|---|---|
| `media-1.xlsx` | source | The paper's supplementary table, 9 sheets |
| `convert.py` | source | One-shot converter: xlsx → CSVs |
| `pcredux_curves.csv` | generated | Long format: `(curve_id, dataset, cycle, fluorescence)` |
| `pcredux_labels.csv` | generated | One row per curve: consensus + per-rater labels |

Regenerate the CSVs after editing `convert.py`:

```bash
python tests/fixtures/pcredux/convert.py
```

The CSVs are committed for tuning-script convenience — they live a
short `git mv` away from being the source-of-truth if `media-1.xlsx`
is ever swapped for a richer dataset.

## What's in the fixture

**1023 curves** spanning 5 sub-datasets, all TaqMan chemistry:

| Sub-dataset | Curves | Source |
|---|---|---|
| `batsch1` | 15 | Batsch et al, BMC Bioinf (2008) — RNA dilution series |
| `batsch3` | 15 | Batsch et al — pig EMT |
| `maro1` | 384 | Marongiu et al, PLOS One (2016) — clinical faecal samples |
| `maro2` | 384 | Marongiu et al — clinical faecal samples |
| `maro3` | 225 | Marongiu et al — clinical faecal samples (partial; paper has 384, only 225 in supp xlsx) |

The paper's Human Rating sheet labels another **2158 curves** (RAS002,
RAS003, competimer, karlen1, kbqPCR, reps, vermeulen2, plus the
missing 159 maro3 curves) but their fluorescence isn't in the
supplementary xlsx. Those labels are dropped during conversion since
we can't run MAK2 against them.

## Label vocabulary

Three independent expert raters classified each curve; the
`label` column is the rater consensus (rater #3's `Consens` field
in the xlsx). Possible values:

| Label | Meaning | Maps to MAK2 status |
|---|---|---|
| `y` | Positive amplification | `PASS` |
| `n` | No amplification | `FAIL` |
| `a` | Ambiguous | `INDETERMINATE` |

The `rater_conformity` column is `True` when all three raters agreed;
∼95% of curves in this subset are conformant. Tuning experiments can
optionally weight conformant curves higher (the disagreements are
typically genuinely hard cases that even humans can't classify
consistently).

## Class balance — important caveat for tuning

This subset is **heavily skewed toward negatives**:

|  | Count | Fraction |
|---|---|---|
| `n` (negative) | 891 | 87% |
| `y` (positive) | 113 | 11% |
| `a` (ambiguous) | 19 | 2% |

(The full Human Rating sheet — including the 2158 curves without
fluorescence — is closer to 52% / 44% / 4%. The raw-fluorescence
subset is dominated by Marongiu's clinical-sample negatives.)

**Consequences for tuning**:

1. **Use balanced accuracy or F1-weighted, not raw accuracy.** Raw
   accuracy is satisfied by predicting `n` for everything (87% score
   without doing anything useful).
2. **The ambiguous class is statistically thin** (19 curves).
   Don't expect to tune the INDETERMINATE-producing gates with high
   confidence from this dataset alone.
3. **Positive-class tuning is also light** (113 curves spanning
   5 sub-datasets). Cross-dataset generalisation can't really be
   established here.
4. **To strengthen tuning**, augment with curves from MAK2's own
   PlateA fixture (once captured), the qpcR R package's
   `data.kissler` / `data.rutledge`, or a private real-plate
   collection.

## Quick exploration

```python
import pandas as pd

labels = pd.read_csv("tests/fixtures/pcredux/pcredux_labels.csv")
curves = pd.read_csv("tests/fixtures/pcredux/pcredux_curves.csv")

# Pull one curve
c = curves[curves["curve_id"] == "batsch1.1"]
cycles = c["cycle"].values
fluor = c["fluorescence"].values
expected_label = labels.loc[labels["curve_id"] == "batsch1.1", "label"].iloc[0]
```
