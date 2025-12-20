"""
Data processing utilities for qPCR data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_qpcr_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load qPCR data from CSV or Excel file.
    
    Expected format:
    - Column 1: Cycle number
    - Column 2: Fluorescence
    
    Parameters
    ----------
    file_path : str
        Path to data file
        
    Returns
    -------
    cycles : np.ndarray
        Cycle numbers
    fluorescence : np.ndarray
        Fluorescence values
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("File must be CSV or Excel format")
    
    # Assume first two columns are cycle and fluorescence
    cycles = df.iloc[:, 0].values
    fluorescence = df.iloc[:, 1].values
    
    return cycles, fluorescence


def prepare_example_data() -> dict:
    """
    Generate example qPCR datasets for testing.
    
    Returns
    -------
    examples : dict
        Dictionary of example datasets
    """
    from mak2_model import MAK2Model
    
    model = MAK2Model()
    examples = {}
    
    # Example 1: High initial concentration
    cycles1, D1, F1 = model.simulate_cycles(
        D0=1000, k=0.5, P0=1e7, n_cycles=35,
        F_bg_intercept=0.1, F_bg_slope=0.001
    )
    examples['High Concentration'] = {
        'cycles': cycles1,
        'fluorescence': F1,
        'true_params': {
            'D0': 1000, 'k': 0.5, 'P0': 1e7,
            'F_bg_intercept': 0.1, 'F_bg_slope': 0.001
        }
    }
    
    # Example 2: Low initial concentration
    cycles2, D2, F2 = model.simulate_cycles(
        D0=10, k=0.3, P0=5e6, n_cycles=40,
        F_bg_intercept=0.05, F_bg_slope=0.0005
    )
    examples['Low Concentration'] = {
        'cycles': cycles2,
        'fluorescence': F2,
        'true_params': {
            'D0': 10, 'k': 0.3, 'P0': 5e6,
            'F_bg_intercept': 0.05, 'F_bg_slope': 0.0005
        }
    }
    
    # Example 3: Primer limiting
    cycles3, D3, F3 = model.simulate_cycles(
        D0=100, k=0.4, P0=1e5, n_cycles=35,
        F_bg_intercept=0.08, F_bg_slope=0.002
    )
    examples['Primer Limiting'] = {
        'cycles': cycles3,
        'fluorescence': F3,
        'true_params': {
            'D0': 100, 'k': 0.4, 'P0': 1e5,
            'F_bg_intercept': 0.08, 'F_bg_slope': 0.002
        }
    }
    
    return examples


def export_results(
    cycles: np.ndarray,
    fluorescence_data: np.ndarray,
    fluorescence_fit: np.ndarray,
    params: dict,
    output_path: str
):
    """
    Export fitting results to CSV.
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    fluorescence_data : np.ndarray
        Measured fluorescence
    fluorescence_fit : np.ndarray
        Fitted fluorescence
    params : dict
        Fitted parameters
    output_path : str
        Path for output CSV file
    """
    # Create DataFrame with data
    df = pd.DataFrame({
        'Cycle': cycles,
        'Fluorescence_Data': fluorescence_data,
        'Fluorescence_Fit': fluorescence_fit,
        'Residual': fluorescence_data - fluorescence_fit
    })
    
    # Add parameters as metadata in separate rows
    param_df = pd.DataFrame({
        'Parameter': list(params.keys()),
        'Value': list(params.values())
    })
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("# MAK2 Fitting Results\n")
        f.write("# Parameters:\n")
        param_df.to_csv(f, index=False)
        f.write("\n# Data:\n")
        df.to_csv(f, index=False)
    
    print(f"Results exported to {output_path}")


if __name__ == "__main__":
    # Test example data generation
    examples = prepare_example_data()
    
    print("Generated example datasets:")
    for name, data in examples.items():
        print(f"\n{name}:")
        print(f"  Cycles: {len(data['cycles'])}")
        print(f"  Fluorescence range: {data['fluorescence'].min():.2f} - {data['fluorescence'].max():.2f}")
        print(f"  True params: D0={data['true_params']['D0']}, k={data['true_params']['k']}, P0={data['true_params']['P0']:.2e}")
