# qPCR MAK2 Analyzer

A Python-based tool for fitting mechanistic PCR models to quantitative PCR (qPCR) data, with extensions for primer depletion.

## Overview

This implementation is based on the MAK2 model from Boggy & Woolf (2010) with additional features:
- **Primer depletion tracking**: Monitors primer consumption across PCR cycles
- **Background fluorescence modeling**: Linear model for baseline drift
- **Interactive visualization**: Streamlit web interface for easy analysis

## Model Description

### MAK2 with Primer Depletion

The extended MAK2 model tracks DNA amplification while accounting for primer consumption:

**Within each cycle** (constant primer concentration P_n):
```
D_n = D_{n-1} + (k × P_n) × ln(1 + D_{n-1} / (k × P_n))
```

**Between cycles** (primer update):
```
P_n = P_{n-1} - 2 × (D_n - D_{n-1})
```

**Smart Truncation Strategy**: While the model includes primer depletion, the plateau phase 
may involve other mechanisms (enzyme degradation, dNTP depletion, product inhibition). 
The implementation offers two truncation methods:

1. **Fluorescence threshold** (default: 85% of max) - Truncates at signal level
2. **Slope threshold** (default: 10% of max slope) - Truncates when growth rate drops

Both methods capture:
- Full exponential growth
- Early plateau (primer-limited)
- While avoiding deep plateau artifacts

Thresholds are user-configurable (50-100% for fluorescence, 5-50% for slope) in the web app.

Where:
- `D_n`: DNA concentration at cycle n
- `P_n`: Primer concentration at cycle n  
- `k`: PCR characteristic constant (ratio of primer binding to reannealing rate)

**Fluorescence model**:
```
F_n = D_n + F_bg_intercept + F_bg_slope × n
```

### Fitted Parameters

1. **D₀**: Initial DNA template concentration
2. **k**: PCR constant characterizing reaction kinetics
3. **P₀**: Initial primer concentration
4. **F_bg_intercept**: Background fluorescence (intercept)
5. **F_bg_slope**: Background fluorescence (linear slope)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

### Python API

```python
from mak2_model import MAK2Model
from optimizer import MAK2Optimizer
import numpy as np

# Load your data
cycles = np.array([1, 2, 3, ...])
fluorescence = np.array([0.1, 0.15, 0.25, ...])

# Fit the model
optimizer = MAK2Optimizer()
params = optimizer.fit(cycles, fluorescence)

# View results
print(f"Initial DNA (D₀): {params['D0']:.2e}")
print(f"PCR constant (k): {params['k']:.4f}")
print(f"Initial primer (P₀): {params['P0']:.2e}")

# Make predictions
F_pred = optimizer.predict(cycles)
```

## File Structure

```
├── app.py                  # Streamlit web interface
├── mak2_model.py          # Core MAK2 model implementation
├── optimizer.py           # Parameter fitting via optimization
├── data_processing.py     # Data loading and export utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

### Data Input Options
- **Example datasets**: Pre-loaded synthetic data with known parameters
- **File upload**: CSV or Excel files with cycle/fluorescence columns
- **Manual entry**: Direct input of data points

### Fitting Options
- **Fluorescence threshold truncation**: Configurable 50-100% (default: 85%)
- **Auto-truncation**: Automatically stops at threshold
- **Manual truncation**: User-specified truncation point
- **Optimization methods**: 
  - Differential Evolution (global optimization, slower but robust)
  - Nelder-Mead (local optimization, faster but needs good initial guess)
- **Custom bounds**: Advanced users can specify parameter search ranges

### Visualizations
- qPCR curve with fitted model overlay
- Residual plot for fit quality assessment
- Amplification efficiency over cycles
- Exportable results in CSV format

## Theory

### Why MAK2?

Traditional qPCR quantification methods either:
1. Require standard curves (time/sample intensive)
2. Assume constant amplification efficiency (mechanistically incorrect)

MAK2 provides:
- **Mechanistic accuracy**: Based on actual PCR chemistry
- **Single-assay quantification**: No standard curve needed
- **Declining efficiency prediction**: Naturally models efficiency changes

### Model Assumptions

The model is valid when:
1. Reactions occur according to mass action kinetics
2. Enzyme (polymerase) is not saturated
3. Errors during PCR are negligible
4. No significant off-target amplification

**Important**: Unlike the original MAK2, this extended version does NOT require truncation 
before the plateau phase. The primer depletion mechanism naturally models the plateau.

## Limitations & Future Work

**Current limitations:**
- Parameter identifiability: k and P₀ can trade off (k×P₀ forms effective constant)
- Optimization can be sensitive to initial bounds
- No confidence intervals on parameters (yet)

**Planned improvements:**
- Bootstrap confidence intervals
- Multi-curve fitting for improved parameter estimation
- Export to other formats (JSON, MATLAB)
- Batch processing of multiple samples

## References

Boggy, G.J., & Woolf, P.J. (2010). A Mechanistic Model of PCR for Accurate Quantification of Quantitative PCR Data. PLOS ONE, 5(8), e12355.
https://doi.org/10.1371/journal.pone.0012355

## License

MIT License - feel free to use and modify for your research!

## Contributing

Contributions welcome! Areas for improvement:
- Better optimization strategies
- Parameter uncertainty quantification
- Additional visualization options
- Support for multiple dyes/targets

## Contact

Built with Claude (Anthropic) for qPCR data analysis.
