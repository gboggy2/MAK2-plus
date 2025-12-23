"""
Example dataset loader for MAK2+ application.

Provides convenient access to curated qPCR datasets for demonstration,
testing, and validation of the MAK2+ analysis pipeline.

Author: Greg Boggy, PhD
Date: December 21, 2024
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class ExampleDataLoader:
    """
    Load and manage example qPCR datasets for MAK2+ demonstration.
    
    Provides access to three curated datasets:
    - Boggy.csv: Classic dilution series (recommended for new users)
    - Rutledge.csv: High-throughput screen (batch processing)
    - reps.csv: Technical replicates study (reproducibility)
    """
    
    def __init__(self, data_dir: str = "example_data"):
        """
        Initialize the example data loader.
        
        Parameters
        ----------
        data_dir : str
            Directory containing example CSV files and metadata
        """
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file."""
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            # Return default metadata if file doesn't exist
            return self._get_default_metadata()
    
    def _get_default_metadata(self) -> Dict:
        """Return default metadata structure."""
        return {
            "datasets": {
                "Boggy.csv": {
                    "display_name": "Boggy et al. - Dilution Series",
                    "description": "qPCR dilution series demonstrating MAK2 model fitting",
                    "n_samples": 12,
                    "n_cycles": 40
                },
                "Rutledge.csv": {
                    "display_name": "Rutledge - High-Throughput Screen",
                    "description": "Large-scale qPCR dataset with multiple replicates",
                    "n_samples": 120,
                    "n_cycles": 45
                },
                "reps.csv": {
                    "display_name": "Technical Replicates Study",
                    "description": "Seven concentration levels with quad replicates",
                    "n_samples": 28,
                    "n_cycles": 49
                }
            }
        }
    
    def get_available_datasets(self) -> Dict[str, str]:
        """
        Get list of available example datasets with descriptions.
        
        Returns
        -------
        dict
            Dictionary mapping filenames to display names
        """
        datasets = {}
        for filename, info in self.metadata["datasets"].items():
            datasets[filename] = info["display_name"]
        return datasets
    
    def get_dataset_info(self, filename: str) -> Dict:
        """
        Get detailed information about a specific dataset.
        
        Parameters
        ----------
        filename : str
            Name of the dataset file (e.g., 'Boggy.csv')
        
        Returns
        -------
        dict
            Dataset metadata including description, characteristics, etc.
        """
        if filename in self.metadata["datasets"]:
            return self.metadata["datasets"][filename]
        else:
            raise ValueError(f"Dataset '{filename}' not found in metadata")
    
    def load_dataset(
        self, 
        filename: str,
        add_offset: bool = True,
        offset_value: float = 1e-5
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """
        Load an example dataset.
        
        Parameters
        ----------
        filename : str
            Name of the dataset file to load
        add_offset : bool, default=True
            Add small offset to avoid zero values (important for log transforms)
        offset_value : float, default=1e-5
            Small value to add if add_offset=True
        
        Returns
        -------
        cycles : np.ndarray
            Array of cycle numbers
        fluorescence_df : pd.DataFrame
            DataFrame with fluorescence data (columns = wells)
        metadata : dict
            Dataset metadata and information
        
        Example
        -------
        >>> loader = ExampleDataLoader()
        >>> cycles, fluor_df, info = loader.load_dataset('Boggy.csv')
        >>> print(f"Loaded {len(fluor_df.columns)} wells, {len(cycles)} cycles")
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}\n"
                f"Available datasets: {', '.join(self.get_available_datasets().keys())}"
            )
        
        # Load CSV file
        data = pd.read_csv(filepath)
        
        # Extract cycles (first column)
        cycles = data.iloc[:, 0].values
        
        # Extract fluorescence data (all other columns)
        fluorescence_df = data.iloc[:, 1:]
        
        # Add small offset to avoid zeros (important for background-subtracted data)
        if add_offset:
            fluorescence_df = fluorescence_df + offset_value
        
        # Get metadata
        metadata = self.get_dataset_info(filename)
        
        return cycles, fluorescence_df, metadata
    
    def get_recommended_dataset(self, purpose: str = "learning") -> str:
        """
        Get recommended dataset filename for a specific purpose.
        
        Parameters
        ----------
        purpose : str
            One of: 'learning', 'batch', 'reproducibility'
        
        Returns
        -------
        str
            Recommended dataset filename
        """
        recommendations = {
            "learning": "Boggy.csv",
            "batch": "Rutledge.csv",
            "reproducibility": "reps.csv"
        }
        
        if purpose not in recommendations:
            raise ValueError(
                f"Unknown purpose '{purpose}'. "
                f"Choose from: {', '.join(recommendations.keys())}"
            )
        
        return recommendations[purpose]
    
    def create_streamlit_selector(self) -> Tuple[str, str]:
        """
        Create data for Streamlit selectbox.
        
        Returns
        -------
        display_names : list
            List of display names for dropdown
        filename_map : dict
            Mapping from display name to filename
        
        Example (in Streamlit app)
        -------
        >>> loader = ExampleDataLoader()
        >>> display_names, name_map = loader.create_streamlit_selector()
        >>> selected = st.selectbox("Choose example dataset:", display_names)
        >>> filename = name_map[selected]
        >>> cycles, data, info = loader.load_dataset(filename)
        """
        datasets = self.get_available_datasets()
        display_names = list(datasets.values())
        filename_map = {v: k for k, v in datasets.items()}
        
        return display_names, filename_map


def load_example_data(filename: str, data_dir: str = "example_data") -> Tuple:
    """
    Convenience function to quickly load an example dataset.
    
    Parameters
    ----------
    filename : str
        Name of dataset to load (e.g., 'Boggy.csv')
    data_dir : str
        Directory containing example data
    
    Returns
    -------
    cycles : np.ndarray
        Cycle numbers
    fluorescence_df : pd.DataFrame
        Fluorescence data
    metadata : dict
        Dataset information
    
    Example
    -------
    >>> cycles, fluor, info = load_example_data('Boggy.csv')
    >>> print(info['description'])
    """
    loader = ExampleDataLoader(data_dir=data_dir)
    return loader.load_dataset(filename)


def get_example_for_tutorial() -> Tuple:
    """
    Load the recommended dataset for tutorials and learning.
    
    Returns
    -------
    cycles : np.ndarray
        Cycle numbers
    fluorescence_df : pd.DataFrame
        Fluorescence data
    metadata : dict
        Dataset information
    
    Example
    -------
    >>> cycles, fluor, info = get_example_for_tutorial()
    # This loads Boggy.csv by default
    """
    loader = ExampleDataLoader()
    recommended = loader.get_recommended_dataset("learning")
    return loader.load_dataset(recommended)


if __name__ == "__main__":
    # Demo usage
    loader = ExampleDataLoader()
    
    print("Available datasets:")
    for filename, display_name in loader.get_available_datasets().items():
        print(f"  - {display_name} ({filename})")
    
    print("\nLoading Boggy.csv...")
    cycles, fluor_df, metadata = loader.load_dataset("Boggy.csv")
    
    print(f"\nDataset info:")
    print(f"  Name: {metadata['display_name']}")
    print(f"  Description: {metadata['description']}")
    print(f"  Samples: {len(fluor_df.columns)}")
    print(f"  Cycles: {len(cycles)}")
    print(f"\nFirst 3 wells: {list(fluor_df.columns[:3])}")
