"""
Streamlit app for qPCR data analysis using MAK2 model.
"""

import sys
import os

# Add current directory to Python path for imports
if os.path.dirname(__file__):
    sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mak2_model import MAK2Model, calculate_amplification_efficiency
from optimizer import MAK2Optimizer
from bootstrap import bootstrap_parameter_uncertainty, BootstrapAnalyzer
from example_data_loader import ExampleDataLoader
from qpcr_data_converter import load_qpcr_file, QPCRDataConverter
from sample_selector_ui import create_sample_selector_widget
from replicate_analysis import (
    parse_sample_groups,
    calculate_replicate_stats,
    analyze_dilution_series,
    compare_precision,
    plot_dilution_series_comparison
)

# Page config
st.set_page_config(
    page_title="qPCR MAK2 Analyzer",
    page_icon="🧬",
    layout="wide"
)

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🧬 qPCR Model Fitting with MAK2+")

# Community signup banner
st.info("""
👋 **MAK2+ is in open beta!** Free to use for research and education.
📬 [Join our mailing list](https://docs.google.com/forms/d/e/PLACEHOLDER/viewform) for updates | 💬 [GitHub Discussions](https://github.com/gboggy2/MAK2-plus/discussions) for questions | 🌟 [Community Guide](https://github.com/gboggy2/MAK2-plus/blob/main/COMMUNITY.md)
""", icon="🧬")
st.markdown("""
This tool fits the MAK2 mechanistic model to qPCR data, including primer depletion effects.

**Two Smart Truncation Methods:**
- **Fluorescence threshold**: Truncate at % of max fluorescence (default: 85%)
- **Slope threshold**: Truncate at max derivative + N cycles (default: 3 cycles)

Both avoid deep plateau artifacts (enzyme degradation, dNTP depletion) not modeled by primer depletion.

Based on Boggy & Woolf (2010) with extensions for primer concentration tracking.
""")

# Sidebar for data input
st.sidebar.header("Data Input")

# Initialize example data loader (once)
if 'example_loader' not in st.session_state:
    st.session_state.example_loader = ExampleDataLoader(data_dir="example_data")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Example Data", "Upload File", "Manual Entry"]
)

cycles = None
fluorescence = None
all_samples = {}  # For batch processing
batch_mode = False

if data_source == "Example Data":
    # Get available datasets
    display_names, filename_map = st.session_state.example_loader.create_streamlit_selector()
    
    # Create dropdown selector
    selected_display = st.sidebar.selectbox(
        "Select example dataset:",
        display_names,
        key="example_dataset_selector"
    )
    
    # Get the actual filename
    selected_filename = filename_map[selected_display]
    
    # Show dataset info in an expander
    dataset_info = st.session_state.example_loader.get_dataset_info(selected_filename)
    
    with st.sidebar.expander("ℹ️ About this dataset", expanded=False):
        st.markdown(f"**{dataset_info['display_name']}**")
        st.write(dataset_info['description'])
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Samples", dataset_info['n_samples'])
        col2.metric("Cycles", dataset_info['n_cycles'])
        
        if 'characteristics' in dataset_info:
            st.write("**Characteristics:**")
            for char in dataset_info['characteristics']:
                st.write(f"• {char}")
        
        if 'expected_results' in dataset_info:
            st.info(f"**Expected:** {dataset_info['expected_results']}")
    
    # Load button
    if st.sidebar.button("📥 Load Example Data", type="primary"):
        try:
            # Load the dataset
            loaded_cycles, fluorescence_df, metadata = st.session_state.example_loader.load_dataset(
                selected_filename
            )
            
            # Store in session state immediately
            st.session_state.loaded_cycles = loaded_cycles
            st.session_state.fluorescence_df = fluorescence_df
            st.session_state.dataset_name = selected_display
            st.session_state.data_loaded = True
            
            # Check if multiple samples (batch mode)
            if fluorescence_df.shape[1] > 1:
                st.session_state.batch_mode_available = True
            else:
                st.session_state.batch_mode_available = False
            
            st.sidebar.success(f"✅ Loaded {selected_display}")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {str(e)}")
    
    # After loading, handle batch mode and sample selection
    if st.session_state.get('data_loaded', False):
        loaded_cycles = st.session_state.loaded_cycles
        fluorescence_df = st.session_state.fluorescence_df
        
        if st.session_state.get('batch_mode_available', False):
            st.sidebar.info(f"📊 Loaded {fluorescence_df.shape[1]} samples")
            batch_mode = st.sidebar.checkbox("Batch fit all samples", value=True, key="example_batch_mode")
            
            if batch_mode:
                # Store all samples
                cycles = loaded_cycles
                for col in fluorescence_df.columns:
                    all_samples[col] = fluorescence_df[col].values
                st.sidebar.success(f"Loaded {len(all_samples)} samples")
                
                # Select one to preview
                preview_sample = st.sidebar.selectbox("Preview sample:", list(all_samples.keys()))
                fluorescence = all_samples[preview_sample]
            else:
                # Single sample mode - let user select
                sample_col = st.sidebar.selectbox("Select sample column:", fluorescence_df.columns)
                cycles = loaded_cycles
                fluorescence = fluorescence_df[sample_col].values
        else:
            # Single sample dataset
            cycles = loaded_cycles
            fluorescence = fluorescence_df.iloc[:, 0].values
    elif data_source == "Example Data":
        # Show instruction if no data loaded yet
        st.sidebar.info("👆 Click 'Load Example Data' to begin")

elif data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Supports various qPCR formats: simple, wide, Bio-Rad CFX, QuantStudio"
    )
    
    if uploaded_file is not None:
        # Clear fitted results when new file uploaded (use name+size to detect re-uploads)
        # Note: id(uploaded_file) changes on every rerun, so we use name+size instead
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_uploaded_file_key' not in st.session_state:
            st.session_state['last_uploaded_file_key'] = file_key
        elif st.session_state['last_uploaded_file_key'] != file_key:
            st.session_state['last_uploaded_file_key'] = file_key
            # Clear all previous results for the new file
            for key in ['fitted_params', 'optimizer', 'bootstrap_results', 'batch_results',
                       'uploaded_cycles', 'uploaded_samples', 'uploaded_metadata',
                       'selected_target', 'target_confirmed']:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Process file with enhanced converter on first load
        if 'uploaded_cycles' not in st.session_state:
            try:
                with st.sidebar.status("📂 Processing file...", expanded=True) as status:
                    st.write("Reading file...")
                    
                    # Use enhanced converter
                    converter = QPCRDataConverter(add_offset=True, offset_value=1e-5)
                    loaded_cycles, loaded_samples, metadata = converter.load_from_file(uploaded_file)
                    
                    st.write(f"✅ Detected format: **{metadata['format']}**")
                    st.write(f"✅ Loaded {metadata['n_samples']} samples")
                    st.write(f"✅ {metadata['n_cycles']} cycles per sample")
                    
                    # Store in session state
                    st.session_state.uploaded_cycles = loaded_cycles
                    st.session_state.uploaded_samples = loaded_samples
                    st.session_state.uploaded_metadata = metadata
                    
                    status.update(label="✅ File processed!", state="complete")
                
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                st.sidebar.info("💡 Expected format: First column = cycles, other columns = samples")
                import traceback
                with st.sidebar.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
        
        # After loading, show sample selector in main area
        if 'uploaded_cycles' in st.session_state:
            loaded_cycles = st.session_state.uploaded_cycles
            loaded_samples = st.session_state.uploaded_samples
            metadata = st.session_state.uploaded_metadata
            
            
            # Check if we need target selection first
            if metadata.get('requires_target_selection', False) and 'selected_target' not in st.session_state:
                # Show target selector
                st.markdown("## 🎯 Select Target to Analyze")
                st.info(f"📊 This file contains **multiplexed qPCR data** with multiple targets per well")
                
                extra_info = metadata['extra_info']
                targets = extra_info['targets']
                
                st.markdown("### Available Targets")
                
                # Show target info
                target_counts = {}
                raw_df = extra_info['raw_df']
                for target in targets:
                    count = raw_df[raw_df['Target'] == target]['Well_Position'].nunique()
                    target_counts[target] = count
                
                # Display as table
                target_df = pd.DataFrame({
                    'Target Name': targets,
                    'Wells': [target_counts[t] for t in targets]
                })
                st.dataframe(target_df, use_container_width=True)
                
                # Selection with auto-trigger on change
                st.markdown("### Choose Target")
                
                def on_target_change():
                    """Callback when target selection changes"""
                    target = st.session_state.target_selectbox
                    st.session_state.target_confirmed = False  # Reset confirmation
                
                selected_target = st.selectbox(
                    "Select which target/gene to analyze:",
                    targets,
                    key="target_selectbox",
                    on_change=on_target_change,
                    help="Choose the target gene you want to fit with MAK2+"
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Use a separate button that just sets a flag
                    if st.button("📥 Load Samples for This Target", type="primary", use_container_width=True, key="load_target_btn"):
                        st.session_state.target_confirmed = True
                
                # Check if we should process (outside the button callback)
                if st.session_state.get('target_confirmed', False):
                    try:
                        # Filter data by target
                        with st.spinner(f"Loading {selected_target} data..."):
                            converter = QPCRDataConverter(add_offset=True, offset_value=1e-5)
                            filtered_cycles, filtered_samples = converter.filter_by_target(extra_info, selected_target)
                            
                            # Update session state with filtered data
                            st.session_state.selected_target = selected_target
                            st.session_state.uploaded_cycles = filtered_cycles
                            st.session_state.uploaded_samples = filtered_samples
                            
                            # Create new metadata without extra_info (no longer needed)
                            st.session_state.uploaded_metadata = {
                                'format': metadata['format'],
                                'n_samples': len(filtered_samples),
                                'n_cycles': len(filtered_cycles),
                                'sample_names': list(filtered_samples.keys()),
                                'cycle_range': (filtered_cycles.min(), filtered_cycles.max()),
                                'requires_target_selection': False
                            }
                            
                            # Clear the confirmation flag
                            st.session_state.target_confirmed = False
                            
                            st.success(f"✅ Loaded {len(filtered_samples)} samples for {selected_target}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error filtering target: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.session_state.target_confirmed = False
                
                # Don't proceed further until target is selected
                st.stop()
            
            # Show file info in sidebar
            target_label = f" ({st.session_state.get('selected_target', 'all')})" if 'selected_target' in st.session_state else ""
            st.sidebar.success(f"✅ {metadata['n_samples']} samples loaded{target_label}")
            with st.sidebar.expander("📋 File Details"):
                st.write(f"**Format:** {metadata['format']}")
                st.write(f"**Cycles:** {metadata['n_cycles']}")
                st.write(f"**Cycle range:** {metadata['cycle_range'][0]} - {metadata['cycle_range'][1]}")
                if 'selected_target' in st.session_state:
                    st.write(f"**Target:** {st.session_state.selected_target}")
                
                # Debug info
                with st.expander("🔍 Debug Info"):
                    st.write(f"loaded_samples dict size: {len(loaded_samples)}")
                    st.write(f"metadata n_samples: {metadata['n_samples']}")
                    st.write(f"requires_target_selection: {metadata.get('requires_target_selection', 'N/A')}")
                    if len(loaded_samples) > 0:
                        st.write(f"Sample names: {list(loaded_samples.keys())[:5]}")
            
            # Show sample selector in main area if no data fitted yet
            if cycles is None or fluorescence is None:
                # Check if we have samples loaded
                if len(loaded_samples) == 0:
                    st.warning("⚠️ No samples loaded. Please select a target first.")
                    st.info("👆 Use the sidebar to upload a file and select a target")
                else:
                    st.markdown("## 📂 Select Samples to Analyze")
                    st.info(f"📊 Loaded **{metadata['n_samples']} samples** from `{uploaded_file.name}`")
                    
                    # Create sample selector interface
                    from sample_selector_ui import SampleSelector
                    selector = SampleSelector(loaded_cycles, loaded_samples)
                    
                    # Show full interface
                    mode_key, selected_samples = selector.create_full_interface(
                        key_prefix="upload",
                        show_preview=True
                    )
                    
                    # Add "Ready to Fit" button
                    if selected_samples:
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("🚀 Ready to Fit Selected Samples", type="primary", use_container_width=True):
                                # Store selected samples for fitting
                                if mode_key == "batch":
                                    # Batch mode
                                    cycles = loaded_cycles
                                    batch_mode = True
                                    all_samples = {name: loaded_samples[name] for name in selected_samples}
                                    # Set first sample as preview
                                    fluorescence = loaded_samples[selected_samples[0]]
                                    st.session_state.upload_ready_batch = True
                                    st.session_state.upload_batch_samples = all_samples
                                    st.session_state.upload_preview_sample = selected_samples[0]
                                else:
                                    # Single mode
                                    cycles = loaded_cycles
                                    fluorescence = loaded_samples[selected_samples[0]]
                                    batch_mode = False
                                    st.session_state.upload_ready_single = True
                                
                                st.session_state.upload_cycles = cycles
                                st.session_state.upload_fluorescence = fluorescence
                                st.session_state.upload_batch_mode = batch_mode
                                st.rerun()
            
            # If already ready to fit, load from session state
            if st.session_state.get('upload_ready_batch', False):
                cycles = st.session_state.upload_cycles
                all_samples = st.session_state.upload_batch_samples
                batch_mode = True
                preview_sample_name = st.session_state.get('upload_preview_sample')
                fluorescence = all_samples[preview_sample_name]

                # Show info in sidebar
                st.sidebar.info(f"🎯 Ready to fit {len(all_samples)} samples (batch mode)")
                # Allow changing preview sample
                preview_sample_name = st.sidebar.selectbox(
                    "Preview sample:",
                    list(all_samples.keys()),
                    index=list(all_samples.keys()).index(preview_sample_name) if preview_sample_name in all_samples else 0
                )
                fluorescence = all_samples[preview_sample_name]
                st.session_state.upload_preview_sample = preview_sample_name

                # Button to change selection/mode
                if st.sidebar.button("↩️ Change Selection", key="change_selection_batch"):
                    for key in ['upload_ready_batch', 'upload_ready_single', 'upload_batch_samples',
                               'upload_preview_sample', 'upload_cycles', 'upload_fluorescence',
                               'upload_batch_mode', 'fitted_params', 'optimizer', 'batch_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            elif st.session_state.get('upload_ready_single', False):
                cycles = st.session_state.upload_cycles
                fluorescence = st.session_state.upload_fluorescence
                batch_mode = False

                st.sidebar.success("🎯 Ready to fit single sample")

                # Button to change selection/mode
                if st.sidebar.button("↩️ Change Selection", key="change_selection_single"):
                    for key in ['upload_ready_batch', 'upload_ready_single', 'upload_batch_samples',
                               'upload_preview_sample', 'upload_cycles', 'upload_fluorescence',
                               'upload_batch_mode', 'fitted_params', 'optimizer', 'batch_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

elif data_source == "Manual Entry":
    manual_input = st.sidebar.text_area(
        "Enter cycle,fluorescence pairs (one per line):",
        "1,0.1\n2,0.15\n3,0.25\n4,0.45\n5,0.85",
        key="manual_input_text"
    )
    if manual_input:
        try:
            lines = [line.strip() for line in manual_input.split('\n') if line.strip()]
            data_list = [list(map(float, line.split(','))) for line in lines]
            cycles = np.array([d[0] for d in data_list])
            fluorescence = np.array([d[1] for d in data_list])
            st.sidebar.success(f"Parsed {len(cycles)} data points")
            
            # Clear fitted results if input changed
            if 'last_manual_input' not in st.session_state:
                st.session_state['last_manual_input'] = manual_input
            elif st.session_state['last_manual_input'] != manual_input:
                st.session_state['last_manual_input'] = manual_input
                if 'fitted_params' in st.session_state:
                    del st.session_state['fitted_params']
                if 'optimizer' in st.session_state:
                    del st.session_state['optimizer']
                if 'bootstrap_results' in st.session_state:
                    del st.session_state['bootstrap_results']
        except:
            st.sidebar.error("Invalid format. Use: cycle,fluorescence")

# Main content
if cycles is not None and fluorescence is not None:
    
    # Show D0 estimation details
    with st.expander("📊 D₀ Bounds Estimation (from Exponential Fits)", expanded=False):
        from mak2_model import estimate_D0_bounds, estimate_MAK2_params_from_exponential

        # Get bounds and estimates with fit info (D0 is now in fluorescence units)
        D0_lower, D0_upper, F_bg_estimate, fit_info = estimate_D0_bounds(
            cycles, fluorescence
        )

        # Display results - top row for D0
        col1, col2, col3 = st.columns(3)
        col1.metric("D₀ Lower Bound", f"{D0_lower:.2e}", help="From perfect doubling fit (2^n) - in fluorescence units")
        col2.metric("D₀ Upper Bound", f"{D0_upper:.2e}",
                   help=f"From efficiency fit (E^n, E={fit_info.get('efficiency', 1.8):.2f}) - in fluorescence units")
        col3.metric("Background Est.", f"{F_bg_estimate:.4f}", help="Estimated from exponential fits")

        # Also get analytical estimates for k and P0
        try:
            analytical_estimates, analytical_bounds = estimate_MAK2_params_from_exponential(
                cycles, fluorescence, P0_assumed=1.0, verbose=False
            )
            k_estimate = analytical_estimates['k']
            P0_estimate = analytical_estimates['P0']

            # Display k and P0 initial guesses in a second row
            st.markdown("**Initial Parameter Guesses:**")
            col4, col5 = st.columns(2)
            col4.metric("k Initial Guess", f"{k_estimate:.4f}", help="Analytical estimate of primer depletion rate from exponential growth")
            col5.metric("P₀ Initial Guess", f"{P0_estimate:.4f}", help="Estimated initial primer concentration (= F_max)")
        except Exception as e:
            st.warning(f"Could not calculate analytical k and P0 estimates: {str(e)}")
        
        # Show exponential phase region and fits
        if fit_info:
            exp_cycles_lower = fit_info['exp_cycles_lower']
            exp_fluor_lower = fit_info['exp_fluorescence_lower']
            fit_lower = fit_info['fit_lower']
            fit_lower_extended = fit_info['fit_lower_extended']  # Extended to efficiency range
            
            exp_cycles_upper = fit_info['exp_cycles_upper']
            exp_fluor_upper = fit_info['exp_fluorescence_upper']
            fit_upper = fit_info['fit_upper']
            
            threshold_cycle = fit_info.get('threshold_cycle', exp_cycles_lower[-1])
            r2_lower = fit_info.get('r2_lower', 0.0)
            r2_upper = fit_info.get('r2_upper', 0.0)
            
            st.info(f"🔬 **Perfect doubling fit**: cycle {exp_cycles_lower[0]:.0f} to {exp_cycles_lower[-1]:.0f}, R² = {r2_lower:.4f}  \n"
                   f"🔬 **Efficiency fit**: cycle {exp_cycles_upper[0]:.0f} to {exp_cycles_upper[-1]:.0f}, R² = {r2_upper:.4f}, E = {fit_info.get('efficiency', 1.8):.2f}")
            
            # Create visualization of exponential fits
            fig_exp = go.Figure()
            
            # Plot all data
            fig_exp.add_trace(go.Scatter(
                x=cycles, y=fluorescence,
                mode='markers',
                name='All Data',
                marker=dict(color='lightgray', size=5, opacity=0.5)
            ))
            
            # Highlight perfect doubling region data points
            fig_exp.add_trace(go.Scatter(
                x=exp_cycles_lower, 
                y=exp_fluor_lower,
                mode='markers',
                name='Perfect Doubling Region',
                marker=dict(color='blue', size=8, symbol='circle')
            ))
            
            # Highlight efficiency region data points
            fig_exp.add_trace(go.Scatter(
                x=exp_cycles_upper, 
                y=exp_fluor_upper,
                mode='markers',
                name='Efficiency Region',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
            
            # Add perfect doubling fitted line (extended to overlay efficiency region)
            fig_exp.add_trace(go.Scatter(
                x=exp_cycles_upper,  # Extended to efficiency range
                y=fit_lower_extended,
                mode='lines',
                name='Perfect Doubling Fit (2^n)',
                line=dict(color='blue', width=3)
            ))
            
            # Add efficiency fitted line
            fig_exp.add_trace(go.Scatter(
                x=exp_cycles_upper,
                y=fit_upper,
                mode='lines',
                name=f'Efficiency Fit (E^n, E={fit_info["efficiency"]:.2f})',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            # Add vertical line at baseline end
            fig_exp.add_vline(
                x=threshold_cycle,
                line_dash="dash",
                line_color="blue",
                annotation_text="Baseline End",
                annotation_position="top right"
            )
            
            # Add background estimate line (sloped if slope is present)
            if 'bg_slope' in fit_info and 'bg_intercept' in fit_info:
                # Plot sloped background line across the data range
                bg_intercept = fit_info['bg_intercept']
                bg_slope = fit_info['bg_slope']
                bg_line_x = np.array([cycles.min(), cycles.max()])
                bg_line_y = bg_intercept + bg_slope * bg_line_x
                
                fig_exp.add_trace(go.Scatter(
                    x=bg_line_x,
                    y=bg_line_y,
                    mode='lines',
                    line=dict(color='purple', dash='dot', width=2),
                    name='Background Est.',
                    showlegend=True
                ))
            else:
                # Fallback to horizontal line if slope not available
                fig_exp.add_hline(
                    y=F_bg_estimate, 
                    line_dash="dot", 
                    line_color="purple",
                    annotation_text="Background Est."
                )
            
            # Add threshold line
            if 'threshold' in fit_info:
                fig_exp.add_hline(
                    y=fit_info['threshold'],
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Baseline End Level"
                )
            
            fig_exp.update_layout(
                title="Exponential Phase Detection and Fits for D₀ Estimation",
                xaxis_title="Cycle",
                yaxis_title="Fluorescence",
                yaxis_range=[0, fluorescence.max() * 1.05],  # Limit y-axis to data max + 5%
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_exp, use_container_width=True)
            
            st.caption(f"📈 **Blue** (cycles {exp_cycles_lower[0]:.0f}-{exp_cycles_lower[-1]:.0f}): "
                      f"Perfect doubling fit (2^n) → lower D₀ bound. "
                      f"**Red** (cycles {exp_cycles_upper[0]:.0f}-{exp_cycles_upper[-1]:.0f}): "
                      f"Efficiency fit (E={fit_info['efficiency']:.2f}^n) → upper D₀ bound. "
                      f"Both start at cycle 1, but end at different points based on data characteristics.")
    
    # Fitting options
    st.sidebar.header("Fitting Options")
    
    st.sidebar.info(
        "💡 **Smart Truncation**\n\n"
        "Avoids deep plateau effects (enzyme degradation, dNTP depletion) "
        "not modeled by primer depletion alone."
    )
    
    cycles_after_max = st.sidebar.slider(
        "Cycles after max slope",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Cutoff = cycle at maximum slope + this many cycles. "
             "Default: 3 cycles captures plateau onset."
    )
    auto_truncate = True
    truncate_cycle = None
    custom_bounds_dict = None

    # Replicate Analysis Options (only show for batch mode)
    if batch_mode and all_samples:
        with st.sidebar.expander("📊 Replicate Analysis", expanded=False):
            st.markdown("**Group samples into replicates**")

            enable_replicate_analysis = st.checkbox(
                "Enable replicate analysis",
                value=False,
                help="Automatically group samples and calculate statistics"
            )

            if enable_replicate_analysis:
                grouping_pattern = st.radio(
                    "Group samples by:",
                    ["Dot - last (F1.1, F1.2 → F1)",
                     "Dot - first (X1.R1.1, X1.R2.1 → X1)",
                     "Underscore (Sample_A_1, Sample_A_2 → Sample_A)",
                     "Manual grouping (define your own)",
                     "No grouping (show all samples)"],
                    index=0,
                    help="How to parse sample names into replicate groups"
                )

                # Preview groups with current samples
                if "Dot - last" in grouping_pattern:
                    pattern_key = 'dot'
                elif "Dot - first" in grouping_pattern:
                    pattern_key = 'first_part'
                elif grouping_pattern == "Underscore (Sample_A_1, Sample_A_2 → Sample_A)":
                    pattern_key = 'underscore'
                elif grouping_pattern == "Manual grouping (define your own)":
                    pattern_key = 'manual'
                else:
                    pattern_key = None

                # Show preview of groups that will be created
                if pattern_key == 'manual':
                    # Manual grouping interface
                    st.markdown("**Define replicate groups manually:**")
                    st.caption("Enter one group per line. Format: GroupName: Sample1, Sample2, Sample3")

                    # Show available samples
                    with st.expander("📋 Available samples", expanded=False):
                        sample_list = list(all_samples.keys())
                        for i, sample in enumerate(sample_list, 1):
                            st.write(f"{i}. {sample}")

                    # Text area for manual grouping
                    manual_groups_text = st.text_area(
                        "Define groups (one per line):",
                        value=st.session_state.get('manual_groups_text', ''),
                        height=200,
                        placeholder="Example:\nGroup1: Sample_1, Sample_2, Sample_3\nGroup2: Sample_4, Sample_5, Sample_6",
                        help="Each line defines one group. Use format 'GroupName: sample1, sample2, sample3'"
                    )

                    st.session_state['manual_groups_text'] = manual_groups_text

                    # Parse manual groups
                    preview_groups = {}
                    if manual_groups_text.strip():
                        for line in manual_groups_text.strip().split('\n'):
                            if ':' in line:
                                group_name, samples_str = line.split(':', 1)
                                group_name = group_name.strip()
                                samples = [s.strip() for s in samples_str.split(',')]
                                # Validate that samples exist
                                valid_samples = [s for s in samples if s in all_samples]
                                if valid_samples:
                                    preview_groups[group_name] = valid_samples
                                else:
                                    st.warning(f"⚠️ Group '{group_name}' has no valid samples")
                else:
                    preview_groups = parse_sample_groups(list(all_samples.keys()), pattern=pattern_key)

                if len(preview_groups) > 0:
                    st.success(f"✅ Found {len(preview_groups)} replicate groups")
                    with st.expander("Preview groups", expanded=False):
                        for group, samples in list(preview_groups.items())[:5]:  # Show first 5
                            st.write(f"**{group}:** {', '.join(samples)}")
                        if len(preview_groups) > 5:
                            st.write(f"... and {len(preview_groups) - 5} more groups")
                else:
                    st.warning("⚠️ No replicate groups found with this pattern")

                # Dilution series options
                analyze_as_dilution = False
                dilution_info = None

                if len(preview_groups) >= 3:
                    analyze_as_dilution = st.checkbox(
                        "Analyze as dilution series",
                        value=False,
                        help="Check linearity and efficiency for dilution series"
                    )

                    if analyze_as_dilution:
                        dilution_info = st.selectbox(
                            "Dilution type:",
                            ["2-fold serial dilutions",
                             "10-fold serial dilutions",
                             "5-fold serial dilutions",
                             "Custom dilution factors"],
                            help="How are your samples diluted?"
                        )

                        # Determine dilution factor
                        if "2-fold" in dilution_info:
                            dilution_factor = 2
                            custom_dilution_factors = None
                        elif "10-fold" in dilution_info:
                            dilution_factor = 10
                            custom_dilution_factors = None
                        elif "5-fold" in dilution_info:
                            dilution_factor = 5
                            custom_dilution_factors = None
                        else:
                            # Custom dilution factors
                            dilution_factor = None
                            st.markdown("**Define dilution factors for each group:**")
                            st.caption("Enter one line per group. Format: GroupName: dilution_factor")
                            st.caption("Example: if Group1 is undiluted (1×), Group2 is 10× diluted, Group3 is 100× diluted")

                            custom_dilution_text = st.text_area(
                                "Define dilution factors:",
                                value=st.session_state.get('custom_dilution_text', ''),
                                height=150,
                                placeholder="Example:\nGroup1: 1\nGroup2: 10\nGroup3: 100\nGroup4: 1000",
                                help="Dilution factor = how many times diluted (1 = undiluted, 10 = 10× diluted, etc.)"
                            )

                            st.session_state['custom_dilution_text'] = custom_dilution_text

                            # Parse custom dilution factors
                            custom_dilution_factors = {}
                            if custom_dilution_text.strip():
                                for line in custom_dilution_text.strip().split('\n'):
                                    if ':' in line:
                                        group_name, factor_str = line.split(':', 1)
                                        group_name = group_name.strip()
                                        try:
                                            factor = float(factor_str.strip())
                                            if group_name in preview_groups:
                                                custom_dilution_factors[group_name] = factor
                                            else:
                                                st.warning(f"⚠️ Group '{group_name}' not found in replicate groups")
                                        except ValueError:
                                            st.warning(f"⚠️ Invalid dilution factor for '{group_name}': {factor_str}")

                        st.session_state['dilution_factor'] = dilution_factor
                        st.session_state['custom_dilution_factors'] = custom_dilution_factors if dilution_factor is None else None

                        # Option to exclude groups from dilution series
                        st.markdown("**Exclude groups from dilution series (optional):**")
                        st.caption("Select groups to exclude (e.g., most diluted samples with poor amplification)")

                        groups_to_exclude = st.multiselect(
                            "Exclude these groups:",
                            options=list(preview_groups.keys()),
                            default=[],
                            help="These groups will be excluded from dilution series analysis"
                        )

                        st.session_state['exclude_from_dilution'] = groups_to_exclude

                # Store settings in session state
                st.session_state['replicate_analysis_enabled'] = True
                st.session_state['grouping_pattern'] = pattern_key
                st.session_state['preview_groups'] = preview_groups
                st.session_state['analyze_as_dilution'] = analyze_as_dilution
                st.session_state['dilution_info'] = dilution_info
            else:
                st.session_state['replicate_analysis_enabled'] = False

    # Fit button(s)
    if batch_mode and all_samples:
        # Batch mode - fit all samples
        if st.sidebar.button("🔬 Batch Fit All Samples", type="primary"):
            # Clear any previous manual fit results
            if 'fitted_params' in st.session_state:
                del st.session_state['fitted_params']
            if 'optimizer' in st.session_state:
                del st.session_state['optimizer']

            st.subheader("🔄 Batch Fitting Results")

            # === SIGNAL DETECTION: Pre-filter samples before fitting ===
            from data_processing import detect_no_signal_samples

            with st.status("🔍 Detecting samples with no signal...", expanded=True) as status:
                st.write(f"Analyzing {len(all_samples)} samples...")

                valid_samples, no_signal_samples, plate_stats = detect_no_signal_samples(
                    cycles,
                    all_samples,
                    min_range_pct=2.0,  # 2% of max plate signal
                    min_r2=0.80,        # 80% R² for borderline samples
                    verbose=False       # Don't print to console in app
                )

                st.write(f"✅ Found {len(valid_samples)} samples with valid signal")
                if no_signal_samples:
                    st.write(f"⚠️ Flagged {len(no_signal_samples)} samples with no detectable signal")

                status.update(label="✅ Signal detection complete", state="complete")

            # Show flagged samples if any
            if no_signal_samples:
                with st.expander(f"⚠️ {len(no_signal_samples)} samples flagged (will be skipped)", expanded=True):
                    st.info("These samples have insufficient signal for MAK2+ fitting (likely NTC, failed reactions, or no template).")

                    no_signal_df = pd.DataFrame([
                        {
                            'Sample': name,
                            'Reason': info['reason'],
                            'Range': f"{info['F_range']:.4f}",
                            '% of Max': f"{info['F_range_pct']:.1f}%"
                        }
                        for name, info in no_signal_samples.items()
                    ])
                    st.dataframe(no_signal_df, use_container_width=True)

            # Update all_samples to only include valid samples
            all_samples_to_fit = valid_samples

            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Calculate global threshold from all samples for consistent Ct calculation
            # This mimics how real qPCR instruments calculate threshold for a plate
            status_text.text("Calculating global threshold for Ct analysis...")
            global_threshold = None

            # Collect fluorescence from all samples to calculate a plate-wide threshold
            all_fluorescence = []
            for fluor_data in all_samples_to_fit.values():
                all_fluorescence.append(fluor_data)

            if len(all_fluorescence) > 0:
                # Calculate global baseline statistics from early cycles of all samples
                baseline_end = max(3, int(len(cycles) * 0.25))
                all_baseline_values = []

                for fluor_data in all_fluorescence:
                    early_fluor = fluor_data[0:baseline_end]
                    all_baseline_values.extend(early_fluor)

                # Check if data is baseline-subtracted
                mean_early = np.mean(all_baseline_values)
                already_baseline_subtracted = (mean_early < 0.1) or (np.any(np.array(all_baseline_values) < 0))

                if already_baseline_subtracted:
                    # For baseline-subtracted data, use SD of early cycles across all samples
                    global_baseline_sd = np.std(all_baseline_values)
                    global_threshold = 10 * global_baseline_sd if global_baseline_sd > 0 else 0.01
                else:
                    # For raw data, calculate baseline mean and SD
                    global_baseline_mean = np.mean(all_baseline_values)
                    global_baseline_sd = np.std(all_baseline_values)
                    global_threshold = 10 * global_baseline_sd

                st.info(f"📊 Using global threshold: {global_threshold:.6f} (calculated from all {len(all_samples_to_fit)} samples)")

            # Pass 1: Fit all samples normally
            for i, (sample_name, fluor_data) in enumerate(all_samples_to_fit.items()):
                status_text.text(f"Pass 1: Fitting {sample_name}... ({i+1}/{len(all_samples_to_fit)})")
                
                try:
                    model_batch = MAK2Model()
                    optimizer_batch = MAK2Optimizer(model_batch)
                    
                    params_batch = optimizer_batch.fit(
                        cycles,
                        fluor_data,
                        cycles_after_max=cycles_after_max,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=custom_bounds_dict,  # None for automatic, or custom dict
                        verbose=False  # Suppress output in batch mode
                    )
                    
                    metrics_batch = optimizer_batch.calculate_fit_metrics()

                    # Calculate Ct (threshold cycle) for comparison with traditional methods
                    # Use global threshold for consistency across all samples
                    try:
                        ct_results = optimizer_batch.calculate_ct(
                            method='threshold',
                            threshold=global_threshold
                        )
                        ct_value = ct_results['ct']
                    except:
                        ct_value = np.nan

                    # Determine which tier was used
                    if params_batch.get('de_used', False):
                        tier = 'T3-DE'
                    elif params_batch.get('fallback_succeeded', False):
                        tier = 'T2-LHS'
                    elif params_batch.get('used_fixed_background', False):
                        tier = 'T1-Fixed'
                    else:
                        tier = 'T1-Full'

                    results_list.append({
                        'Sample': sample_name,
                        'Ct': ct_value,
                        'D0': params_batch['D0'],
                        'k': params_batch['k'],
                        'P0': params_batch['P0'],
                        'F_bg_intercept': params_batch['F_bg_intercept'],
                        'F_bg_slope': params_batch['F_bg_slope'],
                        'R2': metrics_batch['r_squared'],
                        'RMSE': metrics_batch['rmse'],
                        'NRMSE': metrics_batch['nrmse'] * 100,
                        'SSR': metrics_batch['ssr'],
                        'Tier': tier,
                        'Success': '✓',
                        'FixedBG': '✓' if params_batch.get('used_fixed_background', False) else '',
                        'Fallback': '✓' if params_batch.get('fallback_attempted', False) else '',
                        'FallbackOK': '✓' if params_batch.get('fallback_succeeded', False) else '',
                        'fluor_data': fluor_data  # Store for potential retry
                    })
                except Exception as e:
                    results_list.append({
                        'Sample': sample_name,
                        'D0': None,
                        'k': None,
                        'P0': None,
                        'F_bg_intercept': None,
                        'F_bg_slope': None,
                        'R2': None,
                        'SSR': None,
                        'RMSE': None,
                        'NRMSE': None,
                        'Success': f'✗ Error: {str(e)[:30]}',
                        'fluor_data': fluor_data
                    })
                
                progress_bar.progress((i + 1) / len(all_samples_to_fit))
            
            # Pass 2: Identify high SSR samples and retry with informed bounds
            # Calculate SSR threshold for each sample based on its fluorescence range
            high_ssr_indices = []
            for i, result in enumerate(results_list):
                if result['SSR'] is not None and result['fluor_data'] is not None:
                    F_range = np.max(result['fluor_data']) - np.min(result['fluor_data'])
                    ssr_threshold = 0.01 * (F_range ** 2)
                    if result['SSR'] > ssr_threshold:
                        high_ssr_indices.append(i)
            
            if high_ssr_indices:
                status_text.text(f"Pass 2: Refitting {len(high_ssr_indices)} samples with high SSR...")
                
                # Calculate mean k and P0 from good fits
                good_k_values = [r['k'] for r in results_list if r['k'] is not None and results_list.index(r) not in high_ssr_indices]
                good_P0_values = [r['P0'] for r in results_list if r['P0'] is not None and results_list.index(r) not in high_ssr_indices]
                
                if good_k_values and good_P0_values:
                    mean_k = np.mean(good_k_values)
                    mean_P0 = np.mean(good_P0_values)
                    
                    for idx in high_ssr_indices:
                        result = results_list[idx]
                        sample_name = result['Sample']
                        fluor_data = result['fluor_data']
                        
                        status_text.text(f"Pass 2: Refitting {sample_name} (using mean k={mean_k:.4f}, P0={mean_P0:.2f})...")
                        
                        try:
                            model_retry = MAK2Model()
                            optimizer_retry = MAK2Optimizer(model_retry)
                            
                            # Create informed bounds around mean values from good fits
                            F_max = np.max(fluor_data)
                            informed_bounds = {
                                'k': (mean_k * 0.5, mean_k * 2),  # 4× range around mean
                                'P0': (mean_P0 * 0.5, mean_P0 * 2),  # 4× range around mean
                                'D0': (1e-15, F_max * 10),  # Wide D0 range - allow very low values
                                'F_bg_intercept': (0, F_max),
                                'F_bg_slope': (-0.1, 0.1)
                            }
                            
                            # Override analytical estimates with informed values
                            optimizer_retry.analytical_estimates = {
                                'k': mean_k,
                                'P0': mean_P0,
                                'D0': result['D0'] if result['D0'] else 1e-6,
                                'F_bg_intercept': result['F_bg_intercept'] if result['F_bg_intercept'] else 0.2,
                                'F_bg_slope': result['F_bg_slope'] if result['F_bg_slope'] else 0.0
                            }
                            
                            params_retry = optimizer_retry.fit(
                                cycles,
                                fluor_data,
                                cycles_after_max=cycles_after_max,
                                auto_truncate=auto_truncate,
                                truncate_cycle=truncate_cycle,
                                bounds=informed_bounds,
                                verbose=False
                            )
                            
                            metrics_retry = optimizer_retry.calculate_fit_metrics()
                            
                            # Compare SSR - only use retry if it's better
                            if metrics_retry['ssr'] < result['SSR']:
                                results_list[idx] = {
                                    'Sample': sample_name,
                                    'D0': params_retry['D0'],
                                    'k': params_retry['k'],
                                    'P0': params_retry['P0'],
                                    'F_bg_intercept': params_retry['F_bg_intercept'],
                                    'F_bg_slope': params_retry['F_bg_slope'],
                                    'R2': metrics_retry['r_squared'],
                                    'RMSE': metrics_retry['rmse'],
                                    'NRMSE': metrics_retry['nrmse'] * 100,
                                    'SSR': metrics_retry['ssr'],
                                    'Success': '✓ (retry)',
                                    'fluor_data': fluor_data
                                }
                            else:
                                # Keep original but mark as suspect
                                results_list[idx]['Success'] = '⚠️ High SSR'
                                
                        except Exception as e:
                            # Keep original fit
                            results_list[idx]['Success'] = f'⚠️ Retry failed'
            
            status_text.text("✅ Batch fitting complete!")
            
            # Create results dataframe (remove fluor_data before display)
            display_results = [{k: v for k, v in r.items() if k != 'fluor_data'} for r in results_list]
            results_df = pd.DataFrame(display_results)
            
            # Store in session state for persistence
            st.session_state['batch_results'] = results_df
            st.session_state['batch_results_list'] = results_list
            st.session_state['batch_all_samples'] = all_samples
            st.session_state['batch_no_signal_samples'] = no_signal_samples  # Store flagged samples
            st.session_state['batch_cycles'] = cycles
            st.session_state['batch_settings'] = {
                'cycles_after_max': cycles_after_max,
                'auto_truncate': auto_truncate,
                'truncate_cycle': truncate_cycle,
                'custom_bounds_dict': custom_bounds_dict
            }
        
        # Display batch results (outside button block, always visible if results exist)
        if 'batch_results' in st.session_state:
            st.subheader("🔄 Batch Fitting Results")
            results_df = st.session_state['batch_results']
            results_list = st.session_state['batch_results_list']
            
            # Format numeric columns
            format_dict = {
                'D0': '{:.2e}',
                'k': '{:.6f}',
                'P0': '{:.2e}',
                'F_bg_intercept': '{:.6f}',
                'F_bg_slope': '{:.6f}',
                'R2': '{:.6f}',
                'RMSE': '{:.4f}',
                'NRMSE': '{:.2f}',
                'SSR': '{:.6f}'
            }
            
            st.dataframe(results_df.style.format(format_dict, na_rep='-'), use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            successful = results_df['Success'].str.contains('✓').sum()
            col1.metric("Successful Fits", f"{successful}/{len(results_list)}")
            if successful > 0:
                col2.metric("Mean R²", f"{results_df['R2'].mean():.4f}")
                col3.metric("Median R²", f"{results_df['R2'].median():.4f}")
            
            # Show no-signal samples if any
            if 'batch_no_signal_samples' in st.session_state and st.session_state['batch_no_signal_samples']:
                no_signal_samples = st.session_state['batch_no_signal_samples']
                with st.expander(f"⚠️ {len(no_signal_samples)} samples skipped (no signal detected)"):
                    no_signal_df = pd.DataFrame([
                        {
                            'Sample': name,
                            'Reason': info['reason'],
                            'Fluorescence Range': f"{info['F_range']:.4f}",
                            '% of Max on Plate': f"{info['F_range_pct']:.1f}%"
                        }
                        for name, info in no_signal_samples.items()
                    ])
                    st.dataframe(no_signal_df, use_container_width=True)

            # Export button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Results (CSV)",
                csv,
                "batch_fit_results.csv",
                "text/csv",
                key="batch_download"
            )

            # ============================================================================
            # REPLICATE ANALYSIS SECTION
            # ============================================================================
            if st.session_state.get('replicate_analysis_enabled', False):
                st.markdown("---")
                st.subheader("📊 Replicate Analysis")

                # Get grouping pattern from session state
                pattern = st.session_state.get('grouping_pattern', 'dot')

                # Parse sample groups
                if pattern == 'manual':
                    # Use manually defined groups from session state
                    sample_groups = {}
                    manual_groups_text = st.session_state.get('manual_groups_text', '')
                    if manual_groups_text.strip():
                        for line in manual_groups_text.strip().split('\n'):
                            if ':' in line:
                                group_name, samples_str = line.split(':', 1)
                                group_name = group_name.strip()
                                samples = [s.strip() for s in samples_str.split(',')]
                                # Validate that samples exist in results
                                valid_samples = [s for s in samples if s in results_df['Sample'].tolist()]
                                if valid_samples:
                                    sample_groups[group_name] = valid_samples
                else:
                    sample_groups = parse_sample_groups(
                        results_df['Sample'].tolist(),
                        pattern=pattern
                    )

                if len(sample_groups) == 0:
                    st.warning("⚠️ No replicate groups found with the selected pattern")
                else:
                    # Add Group column to results
                    results_with_groups = results_df.copy()
                    group_mapping = {}
                    for group, samples in sample_groups.items():
                        for sample in samples:
                            group_mapping[sample] = group

                    results_with_groups['Group'] = results_with_groups['Sample'].map(
                        lambda x: group_mapping.get(x, x)
                    )

                    # Filter to only samples that are in groups
                    results_with_groups = results_with_groups[
                        results_with_groups['Group'] != results_with_groups['Sample']
                    ]

                    if len(results_with_groups) == 0:
                        st.warning("⚠️ No samples matched the grouping pattern")
                    else:
                        st.success(f"✅ Analyzed {len(sample_groups)} replicate groups ({len(results_with_groups)} samples total)")

                        # Calculate replicate statistics
                        replicate_stats = calculate_replicate_stats(
                            results_with_groups,
                            group_column='Group',
                            metrics=['Ct', 'D0']
                        )

                        # Precision comparison (use default efficiency unless dilution series provides better estimate)
                        # This will be updated if dilution series is analyzed
                        precision_comparison = compare_precision(replicate_stats, efficiency=0.95)

                        # Display tabs for different analyses
                        tab1, tab2, tab3 = st.tabs([
                            "📈 Replicate Statistics",
                            "🎯 Precision Comparison",
                            "📉 Dilution Series"
                        ])

                        with tab1:
                            st.markdown("**Replicate Statistics (Mean ± SD)**")
                            st.markdown("*Coefficient of Variation (CV%) = (SD / Mean) × 100*")

                            # Format for display
                            display_stats = replicate_stats.copy()
                            format_stats_dict = {
                                'Ct_Mean': '{:.2f}',
                                'Ct_SD': '{:.3f}',
                                'Ct_CV': '{:.2f}%',
                                'D0_Mean': '{:.2e}',
                                'D0_SD': '{:.2e}',
                                'D0_CV': '{:.2f}%'
                            }

                            st.dataframe(
                                display_stats.style.format(format_stats_dict, na_rep='-'),
                                use_container_width=True
                            )

                            # Download button for stats
                            stats_csv = replicate_stats.to_csv(index=False)
                            st.download_button(
                                "📥 Download Replicate Statistics (CSV)",
                                stats_csv,
                                "replicate_statistics.csv",
                                "text/csv",
                                key="replicate_stats_download"
                            )

                        with tab2:
                            st.markdown("**Precision Comparison: Ct vs D0**")
                            st.markdown("*Lower CV% = Better precision*")

                            # Show which efficiency is being used
                            if st.session_state.get('analyze_as_dilution', False) and len(sample_groups) >= 3:
                                st.info("ℹ️ **Note:** Ct values are logarithmic, so Ct CV% is converted to concentration CV% using efficiency from dilution series.")
                            else:
                                st.info("ℹ️ **Note:** Ct values are logarithmic, so Ct CV% is converted to concentration CV% assuming 95% efficiency. Enable dilution series analysis for actual efficiency.")

                            # Add color highlighting for better method
                            def highlight_better_precision(row):
                                if pd.isna(row['Ct_Conc_CV']) or pd.isna(row['D0_CV']):
                                    return [''] * len(row)

                                colors = [''] * len(row)
                                ct_conc_cv_idx = precision_comparison.columns.get_loc('Ct_Conc_CV')
                                d0_cv_idx = precision_comparison.columns.get_loc('D0_CV')

                                if row['Better_Precision'] == 'Ct':
                                    colors[ct_conc_cv_idx] = 'background-color: lightgreen'
                                else:
                                    colors[d0_cv_idx] = 'background-color: lightgreen'

                                return colors

                            format_precision_dict = {
                                'Ct_Mean': '{:.2f}',
                                'Ct_SD': '{:.3f}',
                                'Ct_CV': '{:.2f}%',
                                'Ct_Conc_CV': '{:.2f}%',
                                'D0_Mean': '{:.2e}',
                                'D0_CV': '{:.2f}%',
                                'CV_Difference': '{:.2f}%'
                            }

                            st.dataframe(
                                precision_comparison.style.format(format_precision_dict, na_rep='-').apply(
                                    highlight_better_precision, axis=1
                                ),
                                use_container_width=True
                            )

                            # Summary statistics
                            ct_wins = (precision_comparison['Better_Precision'] == 'Ct').sum()
                            d0_wins = (precision_comparison['Better_Precision'] == 'D0').sum()

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Ct Better", f"{ct_wins}/{len(precision_comparison)}")
                            col2.metric("D0 Better", f"{d0_wins}/{len(precision_comparison)}")
                            avg_ct_conc_cv = precision_comparison['Ct_Conc_CV'].mean()
                            avg_d0_cv = precision_comparison['D0_CV'].mean()
                            col3.metric("Avg CV Diff", f"{abs(avg_ct_conc_cv - avg_d0_cv):.2f}%")

                            st.markdown("---")
                            st.markdown("**Interpretation:**")
                            if d0_wins > ct_wins:
                                st.success(f"✅ **D0 quantification** shows better precision in {d0_wins}/{len(precision_comparison)} groups")
                            elif ct_wins > d0_wins:
                                st.info(f"ℹ️ **Ct method** shows better precision in {ct_wins}/{len(precision_comparison)} groups")
                            else:
                                st.info("ℹ️ Both methods show similar precision across groups")

                            # Download button
                            precision_csv = precision_comparison.to_csv(index=False)
                            st.download_button(
                                "📥 Download Precision Comparison (CSV)",
                                precision_csv,
                                "precision_comparison.csv",
                                "text/csv",
                                key="precision_comparison_download"
                            )

                        with tab3:
                            # Check if dilution series analysis is enabled
                            if st.session_state.get('analyze_as_dilution', False):
                                st.markdown("**Dilution Series Analysis**")

                                if len(sample_groups) < 3:
                                    st.warning("⚠️ Need at least 3 dilution levels for analysis")
                                else:
                                    # Get dilution factor from session state
                                    dilution_factor = st.session_state.get('dilution_factor', 2)
                                    custom_dilution_factors = st.session_state.get('custom_dilution_factors', None)
                                    groups_to_exclude = st.session_state.get('exclude_from_dilution', [])

                                    # Filter out excluded groups before analysis
                                    if groups_to_exclude:
                                        results_filtered = results_with_groups[
                                            ~results_with_groups['Group'].isin(groups_to_exclude)
                                        ].copy()
                                        st.info(f"📝 Excluding {len(groups_to_exclude)} group(s) from dilution analysis: {', '.join(groups_to_exclude)}")
                                    else:
                                        results_filtered = results_with_groups

                                    # Run dilution series analysis
                                    dilution_analysis = analyze_dilution_series(
                                        results_filtered,
                                        dilution_factors=custom_dilution_factors,  # Use custom if provided, else auto-generate
                                        dilution_factor=dilution_factor if custom_dilution_factors is None else 2,
                                        group_column='Group'
                                    )

                                    if 'error' in dilution_analysis:
                                        st.error(dilution_analysis['error'])
                                    else:
                                        # Use efficiency from dilution series to recalculate precision comparison
                                        ct_efficiency = dilution_analysis['ct_analysis']['efficiency'] / 100  # Convert % to fraction
                                        precision_comparison = compare_precision(replicate_stats, efficiency=ct_efficiency)

                                        # Display results
                                        col1, col2 = st.columns(2)

                                        with col1:
                                            st.markdown("**Ct Analysis**")
                                            ct_analysis = dilution_analysis['ct_analysis']
                                            st.metric("R²", f"{ct_analysis['r2']:.4f}")
                                            st.metric("Efficiency", f"{ct_analysis['efficiency']:.1f}%")
                                            st.metric("Slope", f"{ct_analysis['slope']:.4f}")

                                        with col2:
                                            st.markdown("**D0 Analysis**")
                                            d0_analysis = dilution_analysis['d0_analysis']
                                            st.metric("R²", f"{d0_analysis['r2']:.4f}")
                                            st.metric("Slope", f"{d0_analysis['slope']:.4f}")
                                            st.caption(f"Expected: {d0_analysis['expected_slope']:.4f}")

                                        # Plot comparison
                                        st.markdown("---")
                                        st.markdown("**Linearity Comparison**")

                                        # Debug: Show error bar values
                                        with st.expander("🔍 Debug: Error Bar Values"):
                                            data_debug = dilution_analysis['data']
                                            st.write("**Ct Standard Deviations:**")
                                            st.write(data_debug[['Group', 'Ct_Mean', 'Ct_SD']].to_string())
                                            st.write("\n**D0 Standard Deviations:**")
                                            st.write(data_debug[['Group', 'D0_Mean', 'D0_SD']].to_string())

                                        dilution_plot = plot_dilution_series_comparison(dilution_analysis)
                                        st.plotly_chart(dilution_plot, use_container_width=True)

                                        # Summary
                                        comparison = dilution_analysis['comparison']
                                        if comparison['better_linearity'] == 'D0':
                                            st.success(f"✅ **D0 shows better linearity** (R² = {comparison['d0_r2']:.4f} vs {comparison['ct_r2']:.4f})")
                                        else:
                                            st.info(f"ℹ️ **Ct shows better linearity** (R² = {comparison['ct_r2']:.4f} vs {comparison['d0_r2']:.4f})")
                            else:
                                st.info("💡 Enable 'Analyze as dilution series' in the sidebar to see linearity analysis")

        # Batch visualization section (outside button block, always visible if results exist)
        if 'batch_results_list' in st.session_state and 'batch_all_samples' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Visualize Individual Fits")
            
            results_list = st.session_state['batch_results_list']
            all_samples = st.session_state['batch_all_samples']
            cycles = st.session_state['batch_cycles']
            batch_settings = st.session_state['batch_settings']
            
            # Sample selector
            sample_names = [r['Sample'] for r in results_list]
            selected_sample = st.selectbox(
                "Select sample to visualize:",
                sample_names,
                key="batch_viz_selector"
            )
            
            # Find the selected sample's index
            selected_idx = sample_names.index(selected_sample)
            selected_result = results_list[selected_idx]
            
            # Check if fit was successful
            if (selected_result['Success'] and 
                (selected_result['Success'].startswith('✓') or 
                 selected_result['Success'] == '⚠️ k@bound' or
                 selected_result['Success'] == '⚠️ High SSR')):
                # Get the corresponding fluorescence data
                sample_fluor = all_samples[selected_sample]
                
                # Use the stored parameters from batch fit to generate prediction
                model_viz = MAK2Model()
                
                # Generate curve using stored parameters (use actual cycle array)
                try:
                    F_pred = model_viz.simulate_to_cycle(
                        D0=selected_result['D0'],
                        k=selected_result['k'],
                        P0=selected_result['P0'],
                        cycles=cycles,
                        F_bg_intercept=selected_result['F_bg_intercept'],
                        F_bg_slope=selected_result['F_bg_slope']
                    )
                    
                    fig_batch = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f"MAK2 Fit: {selected_sample}", "Residuals"),
                        vertical_spacing=0.12,
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Plot data and fit
                    fig_batch.add_trace(
                        go.Scatter(
                            x=cycles, y=sample_fluor,
                            mode='markers',
                            name='Data',
                            marker=dict(size=8, color='blue', opacity=0.6)
                        ),
                        row=1, col=1
                    )
                    
                    fig_batch.add_trace(
                        go.Scatter(
                            x=cycles, y=F_pred,
                            mode='lines',
                            name='MAK2 Fit',
                            line=dict(color='red', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Residuals
                    residuals = sample_fluor - F_pred
                    fig_batch.add_trace(
                        go.Scatter(
                            x=cycles, y=residuals,
                            mode='markers',
                            name='Residuals',
                            marker=dict(size=6, color='green')
                        ),
                        row=2, col=1
                    )
                    fig_batch.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                    
                    fig_batch.update_xaxes(title_text="Cycle", row=2, col=1)
                    fig_batch.update_xaxes(title_text="Cycle", row=1, col=1)
                    fig_batch.update_yaxes(title_text="Fluorescence", row=1, col=1)
                    fig_batch.update_yaxes(title_text="Residual", row=2, col=1)
                    fig_batch.update_layout(height=600, showlegend=True)
                    
                    st.plotly_chart(fig_batch, use_container_width=True)
                    
                    # Show parameters
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("D₀", f"{selected_result['D0']:.2e}")
                    col2.metric("k", f"{selected_result['k']:.6f}")
                    col3.metric("P₀", f"{selected_result['P0']:.2e}")
                    col4.metric("R²", f"{selected_result['R2']:.4f}")
                    
                except Exception as e:
                    st.error(f"Could not visualize fit: {str(e)}")
            else:
                st.warning(f"Fitting failed for {selected_sample}. No visualization available.")
    
    else:
        # Single sample mode
        if st.sidebar.button("🔬 Fit Model", type="primary"):
            with st.spinner("Fitting MAK2 model..."):
                model = MAK2Model()
                optimizer = MAK2Optimizer(model)

                # Capture stdout to display debug output
                import io
                import sys
                captured_output = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = captured_output

                try:
                    fitted_params = optimizer.fit(
                        cycles,
                        fluorescence,
                        cycles_after_max=cycles_after_max,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=custom_bounds_dict,  # None for automatic, or custom dict
                        verbose=True  # Enable progress output
                    )

                    st.session_state['fitted_params'] = fitted_params
                    st.session_state['optimizer'] = optimizer

                    # Restore stdout and save captured output
                    sys.stdout = old_stdout
                    debug_output = captured_output.getvalue()
                    st.session_state['fit_debug_output'] = debug_output
                    # Store hash of data to validate fit matches current data
                    import hashlib
                    data_hash = hashlib.md5(f"{cycles.tobytes()}{fluorescence.tobytes()}".encode()).hexdigest()
                    st.session_state['fitted_data_hash'] = data_hash
                    st.success("✅ Fitting complete!")

                except Exception as e:
                    # Restore stdout even if there was an error
                    sys.stdout = old_stdout
                    debug_output = captured_output.getvalue()
                    if debug_output:
                        st.session_state['fit_debug_output'] = debug_output
                    st.error(f"Fitting failed: {str(e)}")
                    st.stop()
    
    # Display fitted results if available
    if 'fitted_params' in st.session_state:
        fitted_params = st.session_state['fitted_params']
        optimizer = st.session_state['optimizer']

        # Show debug output if available
        if 'fit_debug_output' in st.session_state and st.session_state['fit_debug_output']:
            with st.expander("🐛 Debug Output (Optimization Details)", expanded=True):
                st.code(st.session_state['fit_debug_output'], language='text')

        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Fit Visualization", "📈 Parameters & Metrics", "🔬 Bootstrap CI", "💾 Export"])
        
        with tab1:
                # Predict fitted curve
                F_pred = optimizer.predict(cycles)
                
                # Calculate truncation point for visualization
                from mak2_model import find_slope_threshold_cycle
                threshold_idx = find_slope_threshold_cycle(
                    fluorescence,
                    cycles_after_max=cycles_after_max
                )
                threshold_label = f"Max slope + {cycles_after_max} cycles"

                threshold_cycle_num = cycles[min(threshold_idx, len(cycles)-1)]

                # Create figure
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        f"qPCR Curve Fit (Truncated at Max Slope + {cycles_after_max})",
                        "Residuals"
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.7, 0.3]
                )
                
                # Plot data and fit
                fig.add_trace(
                    go.Scatter(
                        x=cycles, y=fluorescence,
                        mode='markers',
                        name='Data',
                        marker=dict(size=8, color='blue', opacity=0.6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=cycles, y=F_pred,
                        mode='lines',
                        name='MAK2 Fit',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add threshold line (where truncation occurs)
                if threshold_cycle_num < cycles[-1]:
                    fig.add_vline(
                        x=threshold_cycle_num,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=threshold_label,
                        row=1, col=1
                    )
                
                # Plot residuals
                residuals = fluorescence - F_pred
                fig.add_trace(
                    go.Scatter(
                        x=cycles, y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(size=6, color='purple')
                    ),
                    row=2, col=1
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                # Update layout
                fig.update_xaxes(title_text="Cycle", row=2, col=1)
                fig.update_yaxes(title_text="Fluorescence", row=1, col=1)
                fig.update_yaxes(title_text="Residual", row=2, col=1)
                fig.update_layout(height=700, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comprehensive goodness-of-fit metrics (calculated on fitted data only)
                st.subheader("Goodness of Fit Metrics")
                st.caption("⚠️ Metrics calculated only on the fitted region (after truncation), not the full dataset")
                
                metrics = optimizer.calculate_fit_metrics()
                
                # Display main metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R²", f"{metrics['r_squared']:.6f}", 
                           help="Coefficient of determination. Can be negative for poor nonlinear fits!")
                col2.metric("RMSE", f"{metrics['rmse']:.4f}", 
                           help="Root Mean Squared Error (fluorescence units)")
                col3.metric("NRMSE", f"{metrics['nrmse']*100:.2f}%",
                           help="Normalized RMSE (% of signal range)")
                # Show quality indicator based on R²
                quality = "✅ Excellent" if metrics['r_squared'] >= 0.999 else "⚠️ Check fit"
                col4.metric("Fit Quality", quality,
                           help="Based on R² threshold (≥0.999 = excellent)")
                
                # Additional metrics in expander
                with st.expander("📊 Additional Fit Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MAE", f"{metrics['mae']:.4f}", 
                                 help="Mean Absolute Error")
                        st.metric("MAPE", f"{metrics['mape']:.2f}%",
                                 help="Mean Absolute Percentage Error")
                        st.metric("SSR", f"{metrics['ssr']:.6f}",
                                 help="Sum of Squared Residuals")
                    with col2:
                        st.metric("AIC", f"{metrics['aic']:.2f}",
                                 help="Akaike Information Criterion (lower is better)")
                        st.metric("BIC", f"{metrics['bic']:.2f}",
                                 help="Bayesian Information Criterion (lower is better)")
                        st.metric("Reduced χ²", f"{metrics['reduced_chi_sq']:.4f}",
                                 help="Chi-squared per degree of freedom")
                    
                    st.caption(f"**Data points:** {metrics['n_points']} (after truncation) | "
                              f"**Parameters:** {metrics['n_params']} | "
                              f"**Degrees of freedom:** {metrics['dof']}")
            
        with tab2:
            st.subheader("Fitted Parameters")
                
            param_df = pd.DataFrame({
                'Parameter': ['D₀ (Initial DNA)', 'k (PCR constant)', 'P₀ (Initial primer)',
                             'F_bg (intercept)', 'F_bg_slope'],
                'Value': [
                    f"{fitted_params['D0']:.2e}",  # D0 in fluorescence units - scientific notation
                    f"{fitted_params['k']:.6f}",
                    f"{fitted_params['P0']:.4e}",
                    f"{fitted_params['F_bg_intercept']:.6f}",
                    f"{fitted_params['F_bg_slope']:.6f}"
                ],
                'Description': [
                    'Initial template fluorescence (fluorescence units)',
                    'Ratio of primer binding to reannealing rate',
                    'Initial primer concentration',
                    'Background fluorescence (constant)',
                    'Background fluorescence (linear slope)'
                ]
            })
                
            st.dataframe(param_df, use_container_width=True)
                
            # Amplification efficiency plot
            st.subheader("Amplification Efficiency Over Cycles")
                
            # Simulate full model to get DNA concentrations
            _, D, _ = optimizer.model.simulate_cycles(
                D0=fitted_params['D0'],
                k=fitted_params['k'],
                P0=fitted_params['P0'],
                n_cycles=len(cycles),
                F_bg_intercept=fitted_params['F_bg_intercept'],
                F_bg_slope=fitted_params['F_bg_slope']
            )
                
            efficiency = calculate_amplification_efficiency(D)
                
            fig_eff = go.Figure()
            fig_eff.add_trace(
                go.Scatter(
                    x=np.arange(1, len(efficiency)+1),
                    y=efficiency,
                    mode='lines+markers',
                    name='Efficiency'
                )
            )
            fig_eff.update_layout(
                xaxis_title="Cycle",
                yaxis_title="Amplification Efficiency",
                height=400
            )
            st.plotly_chart(fig_eff, use_container_width=True)
            
        with tab3:
            st.subheader("🔬 Bootstrap Confidence Intervals")
                
            st.markdown("""
            **Get professional-grade uncertainty estimates for your parameters!**
                
            Bootstrap analysis provides 95% confidence intervals by:
            1. Resampling residuals from your fit 1000 times
            2. Refitting the model to each bootstrap sample
            3. Calculating percentile-based confidence intervals
                
            ⏱️ **Analysis time:** 10-30 minutes  
            📊 **Output:** Confidence intervals + distribution plots
            """)
                
            # Initialize session state for bootstrap
            if 'bootstrap_running' not in st.session_state:
                st.session_state.bootstrap_running = False
            if 'bootstrap_results' not in st.session_state:
                st.session_state.bootstrap_results = None
            if 'bootstrap_sample_name' not in st.session_state:
                st.session_state.bootstrap_sample_name = None
                
            # Determine current sample name
            if batch_mode:
                current_sample_name = preview_sample
            else:
                current_sample_name = "single_sample"
                
            # Clear bootstrap results if sample changed
            if st.session_state.bootstrap_sample_name != current_sample_name:
                st.session_state.bootstrap_results = None
                st.session_state.bootstrap_sample_name = current_sample_name
                
            # Email input
            st.markdown("### Run Bootstrap Analysis")
            if batch_mode:
                st.info(f"📊 **Current sample:** {current_sample_name}\n\nBootstrap will run on this sample only. Change preview sample to bootstrap a different curve.")
            else:
                st.info("⚠️ Your browser will need to stay open during analysis (~10-30 minutes)")
                
            col1, col2 = st.columns([2, 1])
            with col1:
                n_bootstrap = st.number_input("Number of bootstrap iterations", 
                                             min_value=100, max_value=2000, value=1000, step=100,
                                             help="More iterations = better CI estimates but longer runtime",
                                             key="n_bootstrap_input")
            with col2:
                estimated_time = n_bootstrap / 1000 * 15  # ~15 min for 1000
                st.metric("Estimated Time", f"{estimated_time:.0f} min")
                
            # Show existing results first if they exist
            if st.session_state.bootstrap_results is not None:
                st.success("✅ Bootstrap analysis complete!")
                # Display results here (we'll move the display code up)
                
            # Run button (disabled if already running or have results)
            run_bootstrap = st.button(
                "▶️ Run Bootstrap Analysis", 
                type="primary", 
                use_container_width=True,
                disabled=st.session_state.bootstrap_running or st.session_state.bootstrap_results is not None,
                key="run_bootstrap_btn"
            )
                
            if run_bootstrap:
                # Create progress placeholder
                progress_bar = st.progress(0)
                status_text = st.empty()
                    
                try:
                    status_text.text("Starting bootstrap analysis...")
                        
                    # Run bootstrap with progress updates
                    import time
                    start_time = time.time()
                        
                    # We'll run in smaller chunks to update progress
                    status_text.text(f"Running {n_bootstrap} bootstrap iterations...")
                    progress_bar.progress(0.1)
                        
                    # Get the fitted region (what was actually used in the fit)
                    cycles_fit = optimizer.cycles_fit
                    fluorescence_fit = optimizer.fluorescence_fit
                        
                    # Ensure arrays
                    cycles_fit = np.asarray(cycles_fit)
                    fluorescence_fit = np.asarray(fluorescence_fit)
                        
                    # Get predicted values on fitted region (use all params including background)
                    F_pred_fit = optimizer.predict(cycles_fit, fitted_params)
                        
                    bootstrap_results = bootstrap_parameter_uncertainty(
                        cycles=cycles_fit,  # Use fitted cycles, not full data
                        fluorescence=fluorescence_fit,  # Use fitted fluorescence
                        original_params=fitted_params,  # Pass ALL params including background
                        original_fit=F_pred_fit,  # Prediction on fitted region
                        n_bootstrap=n_bootstrap,
                        confidence_level=0.95,
                        random_seed=None,
                        show_progress=False  # We'll use streamlit progress bar
                    )
                        
                    elapsed_time = time.time() - start_time
                        
                    # Store results in session state
                    st.session_state.bootstrap_results = bootstrap_results
                        
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Complete! ({elapsed_time/60:.1f} minutes)")
                        
                    # Store success flag to prevent re-running bootstrap
                    st.session_state.bootstrap_just_completed = True
                        
                    st.success(f"🎉 Bootstrap complete! {bootstrap_results.n_successful}/{bootstrap_results.n_bootstrap} successful fits")
                    st.rerun()  # Rerun to display results
                        
                except Exception as e:
                    st.error(f"❌ Bootstrap failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
            # Display results if available
            if st.session_state.bootstrap_results is not None:
                results = st.session_state.bootstrap_results
                    
                st.markdown("---")
                st.markdown("## 📊 Bootstrap Results")
                    
                # Show diagnostics FIRST
                with st.expander("🔍 Bootstrap Diagnostics", expanded=True):
                    D0_samples = results.D0_samples
                    k_samples = results.k_samples
                    P0_samples = results.P0_samples
                        
                    diag_col1, diag_col2 = st.columns(2)
                    with diag_col1:
                        st.write("**Sample Statistics:**")
                        st.write(f"- D₀: min={D0_samples.min():.2e}, max={D0_samples.max():.2e}, std={np.std(D0_samples):.2e}")
                        st.write(f"- k: min={k_samples.min():.4f}, max={k_samples.max():.4f}, std={np.std(k_samples):.4f}")
                        st.write(f"- P₀: min={P0_samples.min():.2e}, max={P0_samples.max():.2e}, std={np.std(P0_samples):.2e}")
                        
                    with diag_col2:
                        st.write("**Variation Check:**")
                        st.write(f"- Unique D₀ values: {len(np.unique(D0_samples))}/{len(D0_samples)}")
                        st.write(f"- Unique k values: {len(np.unique(k_samples))}/{len(k_samples)}")
                        st.write(f"- Unique P₀ values: {len(np.unique(P0_samples))}/{len(P0_samples)}")
                            
                        if len(np.unique(D0_samples)) < 10:
                            st.warning("⚠️ Very few unique D₀ values - bootstrap may not be working correctly!")
                        if len(np.unique(k_samples)) < 10:
                            st.warning("⚠️ Very few unique k values - bootstrap may not be working correctly!")
                        
                    st.write("**Original Fit (for comparison):**")
                    st.write(f"- D₀={results.D0_point:.2e}, k={results.k_point:.4f}, P₀={results.P0_point:.2e}")
                    
                # Display confidence intervals in columns
                st.markdown("### Parameter Estimates with 95% Confidence Intervals")
                col1, col2, col3 = st.columns(3)
                    
                with col1:
                    st.metric(
                        "D₀ (Initial DNA)",
                        f"{results.D0_point:.2e}",
                        help="Point estimate from original fit"
                    )
                    st.caption(f"**95% CI:** [{results.D0_ci[0]:.2e}, {results.D0_ci[1]:.2e}]")
                    st.caption(f"**Std:** {np.std(results.D0_samples):.2e}")
                    
                with col2:
                    st.metric(
                        "k (PCR constant)",
                        f"{results.k_point:.4f}",
                        help="Point estimate from original fit"
                    )
                    st.caption(f"**95% CI:** [{results.k_ci[0]:.4f}, {results.k_ci[1]:.4f}]")
                    st.caption(f"**Std:** {np.std(results.k_samples):.4f}")
                    
                with col3:
                    st.metric(
                        "P₀ (Initial primer)",
                        f"{results.P0_point:.2e}",
                        help="Point estimate from original fit"
                    )
                    st.caption(f"**95% CI:** [{results.P0_ci[0]:.2e}, {results.P0_ci[1]:.2e}]")
                    st.caption(f"**Std:** {np.std(results.P0_samples):.2e}")
                    
                # Efficiency
                st.markdown("### Amplification Efficiency")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Efficiency (E)",
                        f"{results.efficiency_point:.4f}",
                        help="E = 1 + k*P₀"
                    )
                with col2:
                    st.caption(f"**95% CI:** [{results.efficiency_ci[0]:.4f}, {results.efficiency_ci[1]:.4f}]")
                    
                # Distribution plots
                st.markdown("### Parameter Distributions")
                    
                try:
                    # Create figure using bootstrap analyzer
                    analyzer = BootstrapAnalyzer(
                        model=MAK2Model(),
                        optimizer=MAK2Optimizer(MAK2Model())
                    )
                    fig = analyzer.plot_bootstrap_distributions(results, figsize=(15, 5))
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not create distribution plots: {e}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
                # Summary table
                with st.expander("📊 Detailed Bootstrap Summary"):
                    summary = results.summary_dict()
                        
                    summary_df = pd.DataFrame({
                        'Parameter': ['D₀', 'k', 'P₀', 'Efficiency'],
                        'Estimate': [
                            f"{summary['D0']['estimate']:.2e}",
                            f"{summary['k']['estimate']:.4f}",
                            f"{summary['P0']['estimate']:.2e}",
                            f"{summary['efficiency']['estimate']:.4f}"
                        ],
                        'CI Lower': [
                            f"{summary['D0']['ci_lower']:.2e}",
                            f"{summary['k']['ci_lower']:.4f}",
                            f"{summary['P0']['ci_lower']:.2e}",
                            f"{summary['efficiency']['ci_lower']:.4f}"
                        ],
                        'CI Upper': [
                            f"{summary['D0']['ci_upper']:.2e}",
                            f"{summary['k']['ci_upper']:.4f}",
                            f"{summary['P0']['ci_upper']:.2e}",
                            f"{summary['efficiency']['ci_upper']:.4f}"
                        ],
                        'Std Dev': [
                            f"{summary['D0']['std']:.2e}",
                            f"{summary['k']['std']:.4f}",
                            f"{summary['P0']['std']:.2e}",
                            'N/A'
                        ]
                    })
                        
                    st.dataframe(summary_df, use_container_width=True)
                        
                    st.caption(f"**Metadata:** {summary['metadata']['n_successful']}/{summary['metadata']['n_bootstrap']} "
                              f"successful fits ({summary['metadata']['success_rate']:.1%}) | "
                              f"Runtime: {summary['metadata']['runtime_seconds']/60:.1f} minutes | "
                              f"Confidence level: {summary['metadata']['confidence_level']:.0%}")
                    
                # Download bootstrap results
                st.markdown("### Download Bootstrap Results")
                    
                # Create CSV with bootstrap samples
                bootstrap_df = pd.DataFrame({
                    'iteration': range(len(results.D0_samples)),
                    'D0': results.D0_samples,
                    'k': results.k_samples,
                    'P0': results.P0_samples
                })
                    
                csv_bootstrap = bootstrap_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Bootstrap Samples (CSV)",
                    data=csv_bootstrap,
                    file_name="bootstrap_samples.csv",
                    mime="text/csv"
                )
                    
                # Clear button
                if st.button("🗑️ Clear Bootstrap Results", key="clear_bootstrap_btn"):
                    st.session_state.bootstrap_results = None
                    st.rerun()
            
        with tab4:
            st.subheader("Export Results")
                
            # Prepare export data
            export_df = pd.DataFrame({
                'Cycle': cycles,
                'Fluorescence_Data': fluorescence,
                'Fluorescence_Fit': F_pred,
                'Residual': residuals
            })
                
            # CSV download
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="mak2_results.csv",
                mime="text/csv"
            )
                
            # Parameter summary
            st.subheader("Parameter Summary (for copying)")
            summary_text = f"""
    MAK2 Fitting Results
    ====================
    D₀ (Initial DNA): {fitted_params['D0']:.4e}
    k (PCR constant): {fitted_params['k']:.6f}
    P₀ (Initial primer): {fitted_params['P0']:.4e}
    F_bg_intercept: {fitted_params['F_bg_intercept']:.6f}
    F_bg_slope: {fitted_params['F_bg_slope']:.6f}
    
    Goodness of Fit (on fitted region):
    R² = {optimizer.calculate_fit_metrics()['r_squared']:.6f}
    RMSE = {optimizer.calculate_fit_metrics()['rmse']:.4f}
    NRMSE = {optimizer.calculate_fit_metrics()['nrmse']*100:.2f}%
    """
            st.text_area("Summary", summary_text, height=250)

else:
    st.info("👈 Please load data using the sidebar")
    
    # Show instructions
    st.markdown("""
    ### Getting Started
    
    1. **Load your data** using one of three methods:
       - Use provided example datasets
       - Upload a CSV or Excel file (first column: cycles, second: fluorescence)
       - Enter data manually
    
    2. **Configure fitting options** in the sidebar
    
    3. **Click "Fit Model"** to run the optimization
    
    4. **Explore results** in the tabs:
       - Visualize the fit quality
       - Examine fitted parameters
       - Export results
    
    ### About the MAK2 Model
    
    The MAK2 model is a mechanistic model of PCR that accounts for:
    - Competition between primer binding and DNA reannealing
    - Primer depletion over cycles
    - Background fluorescence (with linear drift)
    
    **Reference:** Boggy & Woolf (2010). A Mechanistic Model of PCR for Accurate Quantification 
    of Quantitative PCR Data. PLOS ONE 5(8): e12355.
    
    ### Example Datasets
    
    **Boggy et al.** - Classic dilution series from the original MAK2 paper  
    **Rutledge** - High-throughput screen with 120 wells  
    **Technical Replicates** - Precision study with quad replicates
    
    Each dataset includes detailed metadata and expected results.  
    [View full documentation](https://github.com/gboggy2/MAK2-plus/tree/main/example_data)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
