"""
Streamlit UI components for sample selection and data preview

Provides interactive widgets for:
- Viewing loaded sample information
- Selecting which samples to fit
- Preview plots of selected samples
- Batch vs. individual fitting mode selection

Author: Greg Boggy, PhD
Date: January 28, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple


class SampleSelector:
    """
    Interactive sample selection widget for Streamlit.
    
    Handles multi-sample datasets and provides:
    - Sample information table
    - Filter/search functionality
    - Multi-select for batch processing
    - Single-select for individual fitting
    - Preview visualization
    """
    
    def __init__(self, cycles: np.ndarray, samples: Dict[str, np.ndarray]):
        """
        Initialize sample selector.
        
        Parameters
        ----------
        cycles : np.ndarray
            Cycle numbers
        samples : Dict[str, np.ndarray]
            Sample fluorescence data
        """
        self.cycles = cycles
        self.samples = samples
        self.sample_names = list(samples.keys())
        self.n_samples = len(samples)
    
    def show_sample_table(self, show_filters: bool = True) -> pd.DataFrame:
        """
        Display interactive sample information table.
        
        Parameters
        ----------
        show_filters : bool
            Show filtering options
            
        Returns
        -------
        pd.DataFrame
            Sample information
        """
        # Calculate sample statistics
        info_list = []
        for name in self.sample_names:
            fluor = self.samples[name]
            info_list.append({
                'Sample': name,
                'Min Fluor': f"{fluor.min():.3f}",
                'Max Fluor': f"{fluor.max():.3f}",
                'Range': f"{fluor.max() - fluor.min():.3f}",
                'Signal Ratio': f"{fluor.max() / fluor.min():.2f}",
                'Mean': f"{fluor.mean():.3f}"
            })
        
        info_df = pd.DataFrame(info_list)
        
        st.markdown("### 📊 Sample Information")
        st.dataframe(info_df, use_container_width=True, height=min(400, 35 * (len(info_df) + 1)))
        
        return info_df
    
    def select_samples_for_fitting(
        self, 
        mode: str = "batch",
        default_samples: Optional[List[str]] = None
    ) -> List[str]:
        """
        Create sample selection interface.
        
        Parameters
        ----------
        mode : str
            'batch' for multi-select, 'single' for single sample
        default_samples : List[str], optional
            Default selected samples
            
        Returns
        -------
        List[str]
            Selected sample names
        """
        st.markdown("### 🎯 Select Samples to Fit")
        
        if mode == "batch":
            # Multi-select with "Select All" option
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if default_samples is None:
                    default_samples = self.sample_names  # All selected by default
                
                selected = st.multiselect(
                    "Choose samples for batch fitting:",
                    options=self.sample_names,
                    default=default_samples,
                    help="Select multiple samples to fit them all at once"
                )
            
            with col2:
                if st.button("Select All", use_container_width=True):
                    selected = self.sample_names
                    st.rerun()
                
                if st.button("Clear All", use_container_width=True):
                    selected = []
                    st.rerun()
            
            if selected:
                st.success(f"✅ Selected {len(selected)} sample(s)")
            else:
                st.warning("⚠️ No samples selected")
            
            return selected
        
        else:  # single mode
            selected_name = st.selectbox(
                "Choose sample to fit:",
                options=self.sample_names,
                help="Select one sample to analyze"
            )
            return [selected_name] if selected_name else []
    
    def preview_selected_samples(
        self, 
        selected_samples: List[str],
        max_preview: int = 10
    ):
        """
        Show preview plot of selected samples.
        
        Parameters
        ----------
        selected_samples : List[str]
            Sample names to preview
        max_preview : int
            Maximum number of samples to show in preview
        """
        if not selected_samples:
            st.info("👆 Select samples above to preview")
            return
        
        st.markdown("### 👁️ Data Preview")
        
        # Limit preview if too many samples
        preview_samples = selected_samples[:max_preview]
        if len(selected_samples) > max_preview:
            st.info(f"Showing first {max_preview} of {len(selected_samples)} selected samples")
        
        # Create plotly figure
        fig = go.Figure()
        
        for name in preview_samples:
            fluor = self.samples[name]
            fig.add_trace(go.Scatter(
                x=self.cycles,
                y=fluor,
                mode='lines+markers',
                name=name,
                marker=dict(size=4),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Raw Fluorescence Curves",
            xaxis_title="Cycle Number",
            yaxis_title="Fluorescence",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        with st.expander("📈 Preview Statistics"):
            stats_list = []
            for name in preview_samples:
                fluor = self.samples[name]
                stats_list.append({
                    'Sample': name,
                    'Min': fluor.min(),
                    'Max': fluor.max(),
                    'Mean': fluor.mean(),
                    'Std': fluor.std(),
                    'Range': fluor.max() - fluor.min()
                })
            
            stats_df = pd.DataFrame(stats_list)
            st.dataframe(
                stats_df.style.format({
                    'Min': '{:.3f}',
                    'Max': '{:.3f}',
                    'Mean': '{:.3f}',
                    'Std': '{:.3f}',
                    'Range': '{:.3f}'
                }),
                use_container_width=True
            )
    
    def filter_samples_by_criteria(
        self,
        min_signal_ratio: float = 2.0,
        min_max_fluor: Optional[float] = None,
        max_initial_fluor: Optional[float] = None
    ) -> List[str]:
        """
        Filter samples based on quality criteria.
        
        Parameters
        ----------
        min_signal_ratio : float
            Minimum max/min ratio
        min_max_fluor : float, optional
            Minimum maximum fluorescence
        max_initial_fluor : float, optional
            Maximum initial fluorescence
            
        Returns
        -------
        List[str]
            Filtered sample names
        """
        filtered_names = []
        
        for name in self.sample_names:
            fluor = self.samples[name]
            
            # Check signal ratio
            signal_ratio = fluor.max() / fluor.min()
            if signal_ratio < min_signal_ratio:
                continue
            
            # Check max fluorescence
            if min_max_fluor is not None and fluor.max() < min_max_fluor:
                continue
            
            # Check initial fluorescence
            if max_initial_fluor is not None and fluor[0] > max_initial_fluor:
                continue
            
            filtered_names.append(name)
        
        return filtered_names
    
    def create_full_interface(
        self,
        key_prefix: str = "selector",
        show_preview: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Create complete sample selection interface.
        
        Parameters
        ----------
        key_prefix : str
            Prefix for Streamlit widget keys
        show_preview : bool
            Show preview plot
            
        Returns
        -------
        mode : str
            'batch' or 'single'
        selected_samples : List[str]
            Selected sample names
        """
        # Show sample table
        self.show_sample_table()
        
        st.markdown("---")
        
        # Mode selection
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            mode = st.radio(
                "Fitting mode:",
                options=["Batch (All)", "Single Sample"],
                key=f"{key_prefix}_mode",
                help="Batch mode fits all selected samples. Single mode fits one at a time."
            )
        
        mode_key = "batch" if "Batch" in mode else "single"
        
        with col2:
            st.markdown("**Quick Filters**")
            if st.button("📊 High Signal Only", key=f"{key_prefix}_high_signal"):
                # Filter for samples with good signal
                filtered = self.filter_samples_by_criteria(min_signal_ratio=5.0)
                st.session_state[f"{key_prefix}_filtered"] = filtered
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset Selection", key=f"{key_prefix}_reset"):
                if f"{key_prefix}_filtered" in st.session_state:
                    del st.session_state[f"{key_prefix}_filtered"]
                st.rerun()
        
        st.markdown("---")
        
        # Get default samples
        default_samples = st.session_state.get(f"{key_prefix}_filtered", None)
        
        # Sample selection
        selected_samples = self.select_samples_for_fitting(
            mode=mode_key,
            default_samples=default_samples
        )
        
        # Preview
        if show_preview and selected_samples:
            st.markdown("---")
            self.preview_selected_samples(selected_samples)
        
        return mode_key, selected_samples


def create_sample_selector_widget(
    cycles: np.ndarray,
    samples: Dict[str, np.ndarray],
    key_prefix: str = "main"
) -> Tuple[str, List[str]]:
    """
    Convenience function to create sample selector in Streamlit app.
    
    Parameters
    ----------
    cycles : np.ndarray
        Cycle numbers
    samples : Dict[str, np.ndarray]
        Sample fluorescence data
    key_prefix : str
        Prefix for widget keys
        
    Returns
    -------
    mode : str
        'batch' or 'single'
    selected_samples : List[str]
        Selected sample names
        
    Example
    -------
    >>> cycles, samples, info = load_qpcr_file('data.xlsx')
    >>> mode, selected = create_sample_selector_widget(cycles, samples)
    >>> if selected:
    >>>     st.write(f"Ready to fit {len(selected)} sample(s) in {mode} mode")
    """
    selector = SampleSelector(cycles, samples)
    return selector.create_full_interface(key_prefix=key_prefix)


if __name__ == "__main__":
    print("Sample Selector UI Module")
    print("This module provides Streamlit widgets for sample selection.")
    print("Import and use in your Streamlit app with:")
    print("  from sample_selector import create_sample_selector_widget")
