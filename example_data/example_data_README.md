# MAK2+ Example Datasets

This folder contains real qPCR datasets for demonstrating and testing MAK2+ analysis capabilities.

## Available Datasets

### 1. **Boggy.csv** - Dilution Series (Recommended for new users)
- **Description**: Classic 2-fold dilution series with 6 concentration points
- **Samples**: 12 wells (6 dilutions × 2 technical replicates)
- **Cycles**: 40
- **Source**: Boggy & Woolf (2010) PLOS ONE
- **Best for**: Learning MAK2+ basics, understanding primer depletion effects

**Sample layout:**
```
F1.1, F1.2  →  Highest concentration
F2.1, F2.2  →  2-fold dilution
F3.1, F3.2  →  4-fold dilution
F4.1, F4.2  →  8-fold dilution
F5.1, F5.2  →  16-fold dilution
F6.1, F6.2  →  32-fold dilution (lowest)
```

**Expected behavior:**
- Strong amplification in F1-F4
- Late amplification in F5-F6
- Clear primer depletion visible in F1 (plateau phase)
- R² > 0.995 for most wells

---

### 2. **Rutledge.csv** - High-Throughput Screen
- **Description**: Large-scale experiment with multiple concentration levels and replicates
- **Samples**: 120 wells (6 concentrations × 5 exp. replicates × 4 tech. replicates)
- **Cycles**: 45
- **Source**: Rutledge qPCR efficiency studies
- **Best for**: Batch processing, testing bootstrap at low signal, quality control

**Sample layout:**
```
X1.R1.1 to X1.R5.4  →  Highest concentration (20 wells)
X2.R1.1 to X2.R5.4  →  Level 2 (20 wells)
X3.R1.1 to X3.R5.4  →  Level 3 (20 wells)
X4.R1.1 to X4.R5.4  →  Level 4 (20 wells)
X5.R1.1 to X5.R5.4  →  Level 5 (20 wells)
X6.R1.1 to X6.R5.4  →  Lowest concentration (20 wells)
```

**Expected behavior:**
- X1-X3: Excellent fits, tight CIs
- X4: Good fits, acceptable CIs
- X5-X6: Some wells may fail or have wide CIs (low template)
- Bootstrap convergence rate >95% for X1-X4, may drop for X5-X6

---

### 3. **reps.csv** - Technical Replicates Study
- **Description**: Seven concentration levels with quad replicates for precision analysis
- **Samples**: 28 wells (7 concentrations × 4 technical replicates)
- **Cycles**: 49
- **Source**: Internal validation dataset
- **Best for**: Reproducibility testing, CV analysis, quality metrics

**Sample layout:**
```
F1.1 to F1.4  →  Highest concentration
F2.1 to F2.4  →  High
F3.1 to F3.4  →  Medium-high
F4.1 to F4.4  →  Medium
F5.1 to F5.4  →  Medium-low
F6.1 to F6.4  →  Low
F7.1 to F7.4  →  Lowest concentration
```

**Expected behavior:**
- Technical replicates should cluster tightly
- CV(D0) < 5% at high concentrations
- CV increases at low concentrations (F6-F7)
- Good for demonstrating reproducibility metrics

---

## File Format

All datasets use the same CSV format:
```
"Cycles","Sample1.Rep1","Sample1.Rep2",...
1,0.2003,0.1989,...
2,0.2011,0.1995,...
...
```

**Column 1:** Cycle number  
**Columns 2+:** Fluorescence values (one column per well)

**Column naming convention:** `[Sample_ID].[Replicate_Number]`
- Example: `F1.1` = Sample F1, replicate 1
- Example: `X2.R3.4` = Concentration X2, experimental replicate R3, technical replicate 4

---

## Using These Datasets in MAK2+

### Via Web Interface:
1. Click "Load Example Data" dropdown
2. Select desired dataset
3. Click "Load Data"
4. Proceed with analysis

### Via Python:
```python
import pandas as pd

# Load data
data = pd.read_csv('example_data/Boggy.csv')

# Extract cycle and fluorescence data
cycles = data['Cycles'].values
fluorescence = data.iloc[:, 1:].values  # All columns except first

# Run MAK2+ analysis
# ... (see main documentation)
```

---

## Interpreting Results

### Good Quality Indicators:
- ✅ R² > 0.995
- ✅ Bootstrap convergence rate > 95%
- ✅ Residuals randomly distributed around zero
- ✅ Confidence intervals span < 0.5 log₁₀ units
- ✅ Technical replicates: CV(D0) < 10%

### Warning Signs:
- ⚠️ R² < 0.990
- ⚠️ Bootstrap convergence < 90%
- ⚠️ Systematic residual patterns
- ⚠️ Wide confidence intervals (>1 log₁₀ unit)
- ⚠️ High CV in technical replicates (>15%)

### Expected Failures:
- ❌ No template control (NTC) wells
- ❌ Very late amplification (Ct > 38)
- ❌ Noisy baseline
- ❌ Abnormal curve shapes

---

## Citation

If you use these datasets in publications, please cite:

**Software:**
```
Boggy, G. (2024). MAK2+: Open-source computational tool for absolute 
nucleic acid quantification. GitHub. https://github.com/gboggy2/MAK2-plus
```

**Original MAK2 model (for Boggy.csv):**
```
Boggy, G.J., & Woolf, P.J. (2010). A Mechanistic Model of PCR for 
Accurate Quantification of Quantitative PCR Data. PLOS ONE, 5(8), e12355.
https://doi.org/10.1371/journal.pone.0012355
```

---

## Questions or Issues?

- **GitHub Issues**: https://github.com/gboggy2/MAK2-plus/issues
- **Email**: gjboggy@gmail.com

---

**Last updated:** December 21, 2024
