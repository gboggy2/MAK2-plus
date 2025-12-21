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

from mak2_model import MAK2Model, find_truncation_cycle, calculate_amplification_efficiency
from optimizer import MAK2Optimizer
from data_processing import prepare_example_data, export_results
from bootstrap import bootstrap_parameter_uncertainty, BootstrapAnalyzer

# Page config
st.set_page_config(
    page_title="qPCR MAK2 Analyzer",
    page_icon="🧬",
    layout="wide"
)

# ============================================================================
# PASSWORD PROTECTION
# ============================================================================

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or password not correct
    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "🔒 Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.info("Please enter the password to access MAK2+")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "🔒 Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Run password check before showing app
if not check_password():
    st.stop()  # Don't continue if password check failed

# ============================================================================
# MAIN APP (only runs if password is correct)
# ============================================================================

st.title("🧬 qPCR Model Fitting with MAK2+")
st.markdown("""
This tool fits the MAK2 mechanistic model to qPCR data, including primer depletion effects.

**Two Smart Truncation Methods:**
- **Fluorescence threshold**: Truncate at % of max fluorescence (default: 85%)
- **Slope threshold**: Truncate when growth rate drops to % of max slope (default: 10%)

Both avoid deep plateau artifacts (enzyme degradation, dNTP depletion) not modeled by primer depletion.

Based on Boggy & Woolf (2010) with extensions for primer concentration tracking.
""")

# Sidebar for data input
st.sidebar.header("Data Input")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Example Data", "Upload File", "Manual Entry"]
)

cycles = None
fluorescence = None
all_samples = {}  # For batch processing
batch_mode = False

if data_source == "Example Data":
    examples = prepare_example_data()
    example_name = st.sidebar.selectbox("Select example:", list(examples.keys()))
    data = examples[example_name]
    cycles = data['cycles']
    fluorescence = data['fluorescence']
    
    st.sidebar.success(f"Loaded: {example_name}")
    st.sidebar.write("True parameters:")
    for param, value in data['true_params'].items():
        if isinstance(value, float) and value > 1000:
            st.sidebar.write(f"- {param}: {value:.2e}")
        else:
            st.sidebar.write(f"- {param}: {value}")

elif data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    if uploaded_file is not None:
        # Clear fitted results when new file uploaded (use file_id to detect re-uploads)
        file_id = id(uploaded_file)
        if 'last_uploaded_file_id' not in st.session_state:
            st.session_state['last_uploaded_file_id'] = file_id
        elif st.session_state['last_uploaded_file_id'] != file_id:
            st.session_state['last_uploaded_file_id'] = file_id
            # Clear all previous results
            if 'fitted_params' in st.session_state:
                del st.session_state['fitted_params']
            if 'optimizer' in st.session_state:
                del st.session_state['optimizer']
            if 'bootstrap_results' in st.session_state:
                del st.session_state['bootstrap_results']
            if 'batch_results' in st.session_state:
                del st.session_state['batch_results']
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Check if multiple columns (batch mode)
        if df.shape[1] > 2:
            st.sidebar.info(f"📊 Detected {df.shape[1]-1} samples")
            batch_mode = st.sidebar.checkbox("Batch fit all samples", value=True)
            
            if batch_mode:
                # Store all samples
                cycles = df.iloc[:, 0].values
                for col in df.columns[1:]:
                    all_samples[col] = df[col].values
                st.sidebar.success(f"Loaded {len(all_samples)} samples with {len(cycles)} cycles each")
                
                # Select one to preview
                preview_sample = st.sidebar.selectbox("Preview sample:", list(all_samples.keys()))
                fluorescence = all_samples[preview_sample]
            else:
                # Single sample mode - let user select
                sample_col = st.sidebar.selectbox("Select sample column:", df.columns[1:])
                cycles = df.iloc[:, 0].values
                fluorescence = df[sample_col].values
        else:
            # Single sample file
            cycles = df.iloc[:, 0].values
            fluorescence = df.iloc[:, 1].values
        
        st.sidebar.success(f"Loaded {len(cycles)} data points")

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
        from mak2_model import estimate_D0_bounds
        
        # Get bounds and estimates with fit info (D0 is now in fluorescence units)
        D0_lower, D0_upper, F_bg_estimate, fit_info = estimate_D0_bounds(
            cycles, fluorescence
        )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("D₀ Lower Bound", f"{D0_lower:.2e}", help="From perfect doubling fit (2^n) - in fluorescence units")
        col2.metric("D₀ Upper Bound", f"{D0_upper:.2e}", 
                   help=f"From efficiency fit (E^n, E={fit_info.get('efficiency', 1.8):.2f}) - in fluorescence units")
        col3.metric("Background Est.", f"{F_bg_estimate:.4f}", help="Estimated from exponential fits")
        
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
    
    auto_truncate = st.sidebar.checkbox(
        "Enable auto-truncation", 
        value=True,
        help="Recommended: Truncates before deep plateau artifacts"
    )
    
    if auto_truncate:
        truncation_method = st.sidebar.radio(
            "Truncation method:",
            ["fluorescence", "slope"],
            format_func=lambda x: {
                "fluorescence": "Fluorescence threshold (% of max)",
                "slope": "Slope threshold (% of max slope)"
            }[x],
            help="Fluorescence: truncate at % of max signal\nSlope: truncate when growth rate drops"
        )
        
        if truncation_method == "fluorescence":
            max_fluorescence_pct = st.sidebar.slider(
                "Max fluorescence (%)",
                min_value=50.0,
                max_value=100.0,
                value=85.0,
                step=5.0,
                help="Truncate when fluorescence reaches this % of maximum. "
                     "85% captures early plateau, 100% uses all data."
            )
            max_slope_pct = 10.0  # Default for slope method
        else:  # slope method
            max_slope_pct = st.sidebar.slider(
                "Max slope (%)",
                min_value=5.0,
                max_value=50.0,
                value=10.0,
                step=5.0,
                help="Truncate when slope drops below this % of maximum slope. "
                     "10% is conservative, 25% includes more plateau data."
            )
            max_fluorescence_pct = 85.0  # Default for fluorescence method
    else:
        truncation_method = "fluorescence"
        max_fluorescence_pct = 100.0
        max_slope_pct = 100.0
        st.sidebar.warning("⚠️ Using all data may include non-primer plateau effects")
    
    if not auto_truncate:
        max_cycle_idx = st.sidebar.slider(
            "Manual truncation cycle index",
            0, len(cycles)-1, len(cycles)-1,
            help="Manually select where to truncate the data"
        )
        truncate_cycle = cycles[max_cycle_idx]
    else:
        truncate_cycle = None
    
    # Custom bounds (advanced)
    with st.sidebar.expander("⚙️ Advanced: Override Parameter Bounds", expanded=False):
        st.caption("ℹ️ **By default**, the optimizer uses intelligent data-driven bounds from analytical parameter estimation. Only override if you have specific constraints.")
        
        use_custom_bounds = st.checkbox(
            "Override automatic bounds",
            value=False,
            help="Use custom bounds instead of automatic analytical estimation"
        )
        
        if not use_custom_bounds:
            st.success("✓ Using automatic analytical bounds (recommended)")
            custom_bounds_dict = None
        else:
            st.warning("⚠️ Using custom bounds - optimizer will NOT use analytical estimation!")
            auto_populate_D0 = st.checkbox(
                "Auto-populate D₀ from exponential fits", 
                value=True,
                help="Use data-driven D₀ bounds from exponential fitting",
                key="auto_populate_checkbox"
            )
            
            # Clear the number input session state when toggling auto-populate
            # to force them to update with new values
            if 'last_auto_populate' not in st.session_state:
                st.session_state['last_auto_populate'] = auto_populate_D0
            elif st.session_state['last_auto_populate'] != auto_populate_D0:
                # Toggle detected - clear the number input states
                if 'd0_min' in st.session_state:
                    del st.session_state['d0_min']
                if 'd0_max' in st.session_state:
                    del st.session_state['d0_max']
                st.session_state['last_auto_populate'] = auto_populate_D0
            
            if auto_populate_D0:
                from mak2_model import estimate_D0_bounds
                
                try:
                    if cycles is not None and fluorescence is not None and len(cycles) > 0:
                        result = estimate_D0_bounds(cycles, fluorescence)
                        D0_lower_est, D0_upper_est = result[0], result[1]
                        # Widen the bounds to prevent optimizer getting trapped
                        # Exponential fits give estimates, but MAK2 needs flexibility
                        custom_D0_min = D0_lower_est / 100  # 100x lower
                        custom_D0_max = D0_upper_est * 100  # 100x higher
                        st.caption(f"✓ Auto-populated (widened 100x): {custom_D0_min:.2e} to {custom_D0_max:.2e}")
                        st.caption(f"   (Exponential estimates: {D0_lower_est:.2e} to {D0_upper_est:.2e})")
                    else:
                        custom_D0_min = 1e-4
                        custom_D0_max = 1.0
                        st.caption("⚠️ No data available for auto-populate, using defaults")
                except Exception as e:
                    custom_D0_min = 1e-4
                    custom_D0_max = 1.0
                    st.caption(f"⚠️ Auto-populate failed: {str(e)[:50]}, using defaults")
            else:
                custom_D0_min = 1e-4
                custom_D0_max = 1.0
            
            st.write("**D₀ bounds**")
            if auto_populate_D0:
                st.caption("🔒 Bounds auto-set from data (uncheck to manually edit)")
                # Use the computed values directly, don't use number_input
                D0_min = custom_D0_min
                D0_max = custom_D0_max
                # Display as read-only
                col1, col2 = st.columns(2)
                col1.metric("D0 min", f"{D0_min:.2e}")
                col2.metric("D0 max", f"{D0_max:.2e}")
            else:
                # Manual input mode
                D0_min = st.number_input("D0 min", value=float(custom_D0_min), format="%.2e", key="d0_min")
                D0_max = st.number_input("D0 max", value=float(custom_D0_max), format="%.2e", key="d0_max")
            
            st.write("**Other parameter bounds**")
            k_min = st.number_input("k min", value=1e-4, format="%.3e")
            k_max = st.number_input("k max", value=10.0, format="%.2e")
            P0_min = st.number_input("P0 min", value=0.5, format="%.2e")
            P0_max = st.number_input("P0 max", value=1e2, format="%.2e")
            
            # Create custom bounds dict
            custom_bounds_dict = {
                'D0': (D0_min, D0_max),
                'k': (k_min, k_max),
                'P0': (P0_min, P0_max)
            }
            
            # Show what bounds will be used
            st.info(f"**Custom bounds:** D₀: [{D0_min:.2e}, {D0_max:.2e}], k: [{k_min:.2e}, {k_max:.2e}], P₀: [{P0_min:.2e}, {P0_max:.2e}]")
    
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
            
            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Pass 1: Fit all samples normally
            for i, (sample_name, fluor_data) in enumerate(all_samples.items()):
                status_text.text(f"Pass 1: Fitting {sample_name}... ({i+1}/{len(all_samples)})")
                
                try:
                    model_batch = MAK2Model()
                    optimizer_batch = MAK2Optimizer(model_batch)
                    
                    params_batch = optimizer_batch.fit(
                        cycles,
                        fluor_data,
                        truncation_method=truncation_method,
                        max_fluorescence_pct=max_fluorescence_pct,
                        max_slope_pct=max_slope_pct,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=custom_bounds_dict,  # None for automatic, or custom dict
                        verbose=False  # Suppress output in batch mode
                    )
                    
                    metrics_batch = optimizer_batch.calculate_fit_metrics()
                    
                    results_list.append({
                        'Sample': sample_name,
                        'D0': params_batch['D0'],
                        'k': params_batch['k'],
                        'P0': params_batch['P0'],
                        'F_bg_intercept': params_batch['F_bg_intercept'],
                        'F_bg_slope': params_batch['F_bg_slope'],
                        'R2': metrics_batch['r_squared'],
                        'RMSE': metrics_batch['rmse'],
                        'NRMSE': metrics_batch['nrmse'] * 100,
                        'SSR': metrics_batch['ssr'],
                        'Success': '✓',
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
                
                progress_bar.progress((i + 1) / len(all_samples))
            
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
                                'D0': (1e-10, F_max * 10),  # Wide D0 range
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
                                truncation_method=truncation_method,
                                max_fluorescence_pct=max_fluorescence_pct,
                                max_slope_pct=max_slope_pct,
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
            st.session_state['batch_cycles'] = cycles
            st.session_state['batch_settings'] = {
                'truncation_method': truncation_method,
                'max_fluorescence_pct': max_fluorescence_pct,
                'max_slope_pct': max_slope_pct,
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
            
            # Export button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Results (CSV)",
                csv,
                "batch_fit_results.csv",
                "text/csv",
                key="batch_download"
            )
        
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
                
                try:
                    fitted_params = optimizer.fit(
                        cycles,
                        fluorescence,
                        truncation_method=truncation_method,
                        max_fluorescence_pct=max_fluorescence_pct,
                        max_slope_pct=max_slope_pct,
                        auto_truncate=auto_truncate,
                        truncate_cycle=truncate_cycle,
                        bounds=custom_bounds_dict,  # None for automatic, or custom dict
                        verbose=True  # Enable progress output
                    )
                    
                    st.session_state['fitted_params'] = fitted_params
                    st.session_state['optimizer'] = optimizer
                    # Store hash of data to validate fit matches current data
                    import hashlib
                    data_hash = hashlib.md5(f"{cycles.tobytes()}{fluorescence.tobytes()}".encode()).hexdigest()
                    st.session_state['fitted_data_hash'] = data_hash
                    st.success("✅ Fitting complete!")
                    
                except Exception as e:
                    st.error(f"Fitting failed: {str(e)}")
                    st.stop()
    
    # Display fitted results if available
    if 'fitted_params' in st.session_state:
        fitted_params = st.session_state['fitted_params']
        optimizer = st.session_state['optimizer']
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Fit Visualization", "📈 Parameters & Metrics", "🔬 Bootstrap CI", "💾 Export"])
        
        with tab1:
                # Predict fitted curve
                F_pred = optimizer.predict(cycles)
                
                # Calculate truncation point for visualization
                if truncation_method == 'fluorescence':
                    from mak2_model import find_fluorescence_threshold_cycle
                    threshold_idx = find_fluorescence_threshold_cycle(
                        fluorescence, 
                        threshold_pct=max_fluorescence_pct
                    )
                    threshold_label = f"{max_fluorescence_pct:.0f}% Max F"
                else:  # slope method
                    from mak2_model import find_slope_threshold_cycle
                    threshold_idx = find_slope_threshold_cycle(
                        fluorescence,
                        slope_pct=max_slope_pct
                    )
                    threshold_label = f"{max_slope_pct:.0f}% Max Slope"
                
                threshold_cycle_num = cycles[min(threshold_idx, len(cycles)-1)]
                threshold_F = fluorescence[min(threshold_idx, len(cycles)-1)]
                
                # Create figure
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        f"qPCR Curve Fit (Truncated by {truncation_method.title()} Method)", 
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
                if auto_truncate and threshold_cycle_num < cycles[-1]:
                    fig.add_vline(
                        x=threshold_cycle_num,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=threshold_label,
                        row=1, col=1
                    )
                    # Add horizontal line at threshold fluorescence (for fluorescence method)
                    if truncation_method == 'fluorescence':
                        fig.add_hline(
                            y=threshold_F,
                            line_dash="dot",
                            line_color="green",
                            opacity=0.5,
                            row=1, col=1
                        )
                
                # Add manual truncation line if set
                if truncate_cycle and not auto_truncate:
                    fig.add_vline(
                        x=truncate_cycle,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Manual Truncation",
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
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
