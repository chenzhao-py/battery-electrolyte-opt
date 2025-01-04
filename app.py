import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.doe_utils import generate_doe_plan, plot_doe_distribution
from utils.analysis_utils import create_correlation_plot
from utils.optimization_utils import (
    create_surrogate_model,
    suggest_next_samples,
    plot_pareto_front,
    plot_optimization_history,
    calculate_optimization_metrics,
    plot_radar_chart
)

# Set page config first
st.set_page_config(
    page_title="Battery Electrolyte Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove the default top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Title at the very top
    st.title("Battery Electrolyte Optimization")
    
    # Sidebar for options
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This tool helps optimize battery electrolyte compositions using Design of Experiments (DOE) 
        and machine learning approaches.
        
        Developed by the Battery Research Lab.
        Version 1.0
        """)

    # Main navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home",
        "🔬 DOE Planning",
        "📊 Analysis",
        "🎯 Optimization"
    ])
    
    with tab1:
        show_home()
    
    with tab2:
        show_doe_planning()
    
    with tab3:
        show_analysis()
    
    with tab4:
        show_optimization()

def show_home():
    # Create three columns for a modern layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔬 Design of Experiments")
        st.write("""
        Create optimized experimental designs for your battery electrolyte compositions.
        - Latin Hypercube Sampling
        - Interactive Visualizations
        - Downloadable Plans
        """)
    
    with col2:
        st.subheader("📝 Data Collection")
        st.write("""
        Record and manage your experimental results.
        - Structured Data Input
        - Performance Metrics
        - Real-time Updates
        """)
    
    with col3:
        st.subheader("📊 Analysis & Optimization")
        st.write("""
        Analyze results and optimize compositions.
        - Correlation Analysis
        - Performance Prediction
        - Composition Optimization
        """)
    
    st.divider()
    st.subheader("Getting Started")
    st.write("""
    1. Go to the 'DOE Planning' tab to design your experiments
    2. Use the 'Analysis' tab to input results and analyze your data
    3. Use the 'Optimization' tab to optimize your compositions
    """)

def show_doe_planning():
    st.header("Design of Experiments Planning")
    
    # Create a container for the input parameters
    with st.container():
        st.subheader("🧪 Define Experiment Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            num_components = st.number_input(
                "Number of Components",
                min_value=2,
                max_value=10,
                value=3,
                help="Select the number of components in your electrolyte mixture"
            )
        
        with col2:
            n_experiments = st.number_input(
                "Number of Experiments",
                min_value=5,
                max_value=100,
                value=10,
                help="Select how many experiments you want to run"
            )
    
    # Create a container for component definitions
    with st.container():
        st.subheader("🔍 Define Components")
        
        components = []
        ranges = []
        
        for i in range(num_components):
            with st.expander(f"Component {i+1}", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    component = st.text_input(
                        "Name",
                        key=f"comp_{i}",
                        placeholder="e.g., LiPF6"
                    )
                with col2:
                    min_val = st.number_input(
                        "Minimum (%)",
                        0.0,
                        100.0,
                        0.0,
                        key=f"min_{i}"
                    )
                with col3:
                    max_val = st.number_input(
                        "Maximum (%)",
                        0.0,
                        100.0,
                        100.0,
                        key=f"max_{i}"
                    )
                
                if component:
                    components.append(component)
                    ranges.append((min_val, max_val))
    
    # Generate DOE Plan
    if components:
        col1, col2 = st.columns([2, 1])
        with col1:
            generate_button = st.button(
                "🚀 Generate DOE Plan",
                use_container_width=True,
                type="primary"
            )
        
        if generate_button:
            if len(components) >= 2:
                with st.spinner("Generating DOE plan..."):
                    doe_plan = generate_doe_plan(components, ranges, n_experiments)
                
                # Show results in tabs
                result_tab1, result_tab2 = st.tabs(["📊 Visualization", "📋 Data"])
                
                with result_tab1:
                    fig = plot_doe_distribution(doe_plan, components)
                    st.plotly_chart(fig, use_container_width=True)
                
                with result_tab2:
                    st.dataframe(doe_plan, use_container_width=True)
                    
                    # Download button
                    csv = doe_plan.to_csv(index=False)
                    st.download_button(
                        label="📥 Download DOE Plan",
                        data=csv,
                        file_name="doe_plan.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error("Please define at least 2 components")
    else:
        st.info("👆 Start by defining your components above")

def show_analysis():
    st.subheader("Data Analysis")
    
    # Create two columns for the layout
    col_left, col_right = st.columns([1, 1])
    
    # Left column for data loading and controls
    with col_left:
        uploaded_file = st.file_uploader(
            "Upload your DOE plan CSV file",
            type=['csv'],
            help="Select the CSV file containing your DOE plan"
        )
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Show data preview
            with st.expander("📋 View Original DOE Plan", expanded=False):
                st.dataframe(data, use_container_width=True)
            
            # Add performance metrics
            st.subheader("📝 Add Performance Metrics")
            
            # Default metrics
            default_metrics = [
                "Conductivity (mS/cm)",
                "Viscosity (cP)",
                "Diffusivity (m²/s)"
            ]
            
            # Get existing metrics from session state or use defaults
            if 'custom_metrics' not in st.session_state:
                st.session_state['custom_metrics'] = default_metrics.copy()
            
            # Add new metric input
            new_metric = st.text_input(
                "Add New Metric (press Enter to add)",
                placeholder="e.g., Density (g/mL)",
                key="new_metric"
            )
            
            if new_metric and new_metric not in st.session_state['custom_metrics']:
                st.session_state['custom_metrics'].append(new_metric)
                st.session_state['new_metric'] = ""  # Clear input
                st.rerun()
            
            # Show current metrics with delete buttons
            st.write("Current Metrics:")
            metrics_to_remove = []
            for i, metric in enumerate(st.session_state['custom_metrics']):
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_name = st.text_input(f"Metric {i+1}", value=metric, key=f"metric_{i}")
                    if new_name != metric:
                        st.session_state['custom_metrics'][i] = new_name
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        metrics_to_remove.append(metric)
            
            # Remove deleted metrics
            for metric in metrics_to_remove:
                st.session_state['custom_metrics'].remove(metric)
                st.rerun()
            
            # Create results dataframe with current metrics
            results_data = data.copy()
            for metric in st.session_state['custom_metrics']:
                if metric not in results_data.columns:
                    results_data[metric] = 0.0
            
            edited_df = st.data_editor(
                results_data,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed"
            )
            
            if st.button("💾 Save Results", use_container_width=True, type="primary"):
                st.session_state['experiment_results'] = edited_df
                st.session_state['analysis_data'] = edited_df
                st.success("✅ Results saved successfully!")
            
           
    
    # Right column for visualizations
    with col_right:
        if 'analysis_data' in st.session_state:
            data = st.session_state['analysis_data']
            # Scatter plot
            st.subheader("Scatter Plot")
            x_col = st.selectbox("X-axis", st.session_state['analysis_data'].columns.tolist(), key='scatter_x')
            y_col = st.selectbox("Y-axis", st.session_state['analysis_data'].columns.tolist(), key='scatter_y')
            
            if 'scatter_x' in st.session_state and 'scatter_y' in st.session_state:
                fig_scatter = px.scatter(
                    data,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Correlation heatmap below
            st.subheader("Correlation Analysis")
            fig_corr = create_correlation_plot(data)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Add correlation interpretation
            with st.expander("ℹ️ Understanding Correlations", expanded=False):
                st.write("""
                - **Strong Positive Correlation (close to 1)**: As one variable increases, the other tends to increase
                - **Strong Negative Correlation (close to -1)**: As one variable increases, the other tends to decrease
                - **Weak Correlation (close to 0)**: Little to no relationship between variables
                """)
        else:
            st.info("👈 Start by uploading your data and adding performance metrics")

def show_optimization():
    st.subheader("Bayesian Optimization")
    
    if 'experiment_results' not in st.session_state:
        st.warning("⚠️ Please upload and save your experimental results first!")
        return
    
    results = st.session_state['experiment_results']
    
    # Create two columns for layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Optimization Settings")
        
        # Get all possible objectives (numeric columns that aren't components)
        component_cols = [col for col in results.columns if '%' in col]
        metric_cols = [col for col in results.columns if col not in component_cols]
        
        # Objective selection
        selected_objectives = st.multiselect(
            "Select Objectives",
            metric_cols,
            default=metric_cols[:2] if len(metric_cols) > 1 else metric_cols,
            help="Choose metrics to optimize (2-3 recommended for visualization)"
        )
        
        if not selected_objectives:
            st.warning("Please select at least one objective")
            return
            
        # Create a container for objective settings
        with st.container():
            st.write("#### Objective Settings")
            st.write("For each objective, choose whether to maximize or minimize:")
            
            # Create two columns for better layout
            settings_cols = st.columns(2)
            objectives_info = {}
            
            for i, obj in enumerate(selected_objectives):
                col_idx = i % 2  # Alternate between columns
                with settings_cols[col_idx]:
                    st.write(f"**{obj}**")
                    optimization_type = st.radio(
                        f"Optimization type for {obj}",
                        options=["Maximize", "Minimize"],
                        key=f"opt_type_{obj}",
                        horizontal=True,
                        help=f"Choose whether to maximize or minimize {obj}"
                    )
                    objectives_info[obj] = (optimization_type == "Maximize")
                    
                    # Show current range
                    min_val = results[obj].min()
                    max_val = results[obj].max()
                    st.caption(f"Current range: {min_val:.3f} to {max_val:.3f}")
        
        # Number of suggestions
        st.write("#### Suggestion Settings")
        n_suggestions = st.number_input(
            "Number of Suggestions",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of new experiments to suggest"
        )
        
        # Run optimization button
        if st.button("🎯 Run Optimization", use_container_width=True, type="primary"):
            with st.spinner("Running optimization..."):
                # Prepare data for optimization
                X = results[component_cols].values
                bounds = [(results[col].min(), results[col].max()) for col in component_cols]
                
                # Create and fit surrogate models for each objective
                models = {}
                for obj in selected_objectives:
                    y = results[obj].values
                    if not objectives_info[obj]:  # if minimizing
                        y = -y
                    models[obj] = create_surrogate_model(X, y)
                
                # Get suggestions
                suggested_points = suggest_next_samples(
                    models, bounds, n_suggestions, objectives_info
                )
                
                # Create suggestions dataframe
                suggestions_df = pd.DataFrame(
                    suggested_points,
                    columns=component_cols
                )
                
                # Store suggestions in session state
                st.session_state['optimization_suggestions'] = suggestions_df
                st.session_state['optimization_metrics'] = calculate_optimization_metrics(
                    results, selected_objectives
                )
                
                # Store optimization settings
                st.session_state['objectives_info'] = objectives_info
                st.session_state['selected_objectives'] = selected_objectives
                
                # Force rerun to update the right column
                st.rerun()
    
    with col_right:
        if 'optimization_suggestions' in st.session_state:
            st.subheader("Suggested Experiments")
            st.dataframe(
                st.session_state['optimization_suggestions'].round(2),
                use_container_width=True
            )
            
            # Download suggestions
            csv = st.session_state['optimization_suggestions'].to_csv(index=False)
            st.download_button(
                label="📥 Download Suggestions",
                data=csv,
                file_name="suggested_experiments.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Show optimization metrics
            st.subheader("Optimization Metrics")
            metrics = st.session_state['optimization_metrics']
            for obj in st.session_state['selected_objectives']:
                with st.expander(f"📊 {obj} Metrics ({('Maximize' if st.session_state['objectives_info'][obj] else 'Minimize')})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Value", f"{metrics[obj]['best_value']:.3f}")
                        st.metric("Mean", f"{metrics[obj]['mean']:.3f}")
                    with col2:
                        st.metric("Std Dev", f"{metrics[obj]['std']:.3f}")
                        st.metric("Improvement", f"{metrics[obj]['improvement']:.1f}%")
            
            # Show Pareto visualization
            if len(st.session_state['selected_objectives']) >= 2:
                st.subheader("Pareto Analysis")
                
                # Pareto front visualization (2D or 3D)
                fig_pareto = plot_pareto_front(
                    results,
                    st.session_state['selected_objectives'],
                    st.session_state['objectives_info']
                )
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                # For 4+ objectives, also show radar chart
                if len(st.session_state['selected_objectives']) > 3:
                    st.subheader("Radar Chart Analysis")
                    fig_radar = plot_radar_chart(
                        results,
                        st.session_state['selected_objectives'],
                        st.session_state['objectives_info']
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            # Show optimization history
            st.subheader("Optimization History")
            for obj in st.session_state['selected_objectives']:
                fig_history = plot_optimization_history(results, obj)
                st.plotly_chart(fig_history, use_container_width=True)

if __name__ == "__main__":
    main()
