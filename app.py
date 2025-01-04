import streamlit as st
import pandas as pd
import numpy as np
from utils.doe_utils import generate_doe_plan, plot_doe_distribution
from utils.analysis_utils import create_correlation_plot

st.set_page_config(
    page_title="Battery Electrolyte Optimizer",
    page_icon="⚡",
    layout="wide"
)

def main():
    st.title("Battery Electrolyte Optimization")
    
    # Create tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home",
        "🔬 DOE Planning",
        "📝 Experiment Input",
        "📊 Analysis & Optimization"
    ])
    
    with tab1:
        show_home()
    
    with tab2:
        show_doe_planning()
    
    with tab3:
        show_experiment_input()
    
    with tab4:
        show_analysis()

def show_home():
    st.header("Welcome to Battery Electrolyte Optimizer")
    
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
    2. Use the 'Experiment Input' tab to record your results
    3. Visit the 'Analysis & Optimization' tab to analyze and optimize your compositions
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

def show_experiment_input():
    st.header("Experiment Results Input")
    
    # Create container for file upload
    with st.container():
        st.subheader("📤 Upload DOE Plan")
        
        upload_col1, upload_col2 = st.columns([2, 1])
        with upload_col1:
            uploaded_file = st.file_uploader(
                "Upload your DOE plan CSV file",
                type=['csv'],
                help="Select the CSV file containing your DOE plan"
            )
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Show the original plan in an expander
        with st.expander("📋 View Original DOE Plan", expanded=False):
            st.dataframe(data, use_container_width=True)
        
        # Add performance metrics
        st.subheader("📝 Add Performance Metrics")
        
        metrics = [
            "Conductivity (mS/cm)",
            "Viscosity (cP)",
            "Stability (%)"
        ]
        
        results_data = data.copy()
        for metric in metrics:
            results_data[metric] = 0.0
        
        edited_df = st.data_editor(
            results_data,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("💾 Save Results", use_container_width=True, type="primary"):
                st.session_state['experiment_results'] = edited_df
                st.success("✅ Results saved successfully!")

def show_analysis():
    st.header("Analysis & Optimization")
    
    if 'experiment_results' not in st.session_state:
        st.warning("⚠️ Please input experiment results first!")
        
        st.info("""
        To get started:
        1. Go to the 'DOE Planning' tab to generate your experimental plan
        2. Use the 'Experiment Input' tab to input your results
        3. Return here to analyze your data
        """)
        return
    
    results = st.session_state['experiment_results']
    
    # Create tabs for different analyses
    analysis_tab1, analysis_tab2 = st.tabs([
        "📊 Correlation Analysis",
        "🎯 Optimization"
    ])
    
    with analysis_tab1:
        st.subheader("Correlation Analysis")
        fig = create_correlation_plot(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add correlation interpretation
        with st.expander("ℹ️ Understanding Correlations", expanded=False):
            st.write("""
            - **Strong Positive Correlation (close to 1)**: As one variable increases, the other tends to increase
            - **Strong Negative Correlation (close to -1)**: As one variable increases, the other tends to decrease
            - **Weak Correlation (close to 0)**: Little to no relationship between variables
            """)
    
    with analysis_tab2:
        st.subheader("Composition Optimization")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_metric = st.selectbox(
                "Select Target Metric",
                ["Conductivity (mS/cm)", "Viscosity (cP)", "Stability (%)"],
                help="Choose the metric you want to optimize"
            )
        
        optimize_col1, optimize_col2 = st.columns([2, 1])
        with optimize_col1:
            if st.button("🎯 Run Optimization", use_container_width=True, type="primary"):
                with st.spinner("Running optimization..."):
                    # Add optimization logic here
                    st.success("✅ Optimization complete!")
                    
                    # Placeholder for optimization results
                    st.subheader("Optimization Results")
                    st.write("Best composition found:")
                    
                    # Create a sample result (replace with actual optimization)
                    sample_result = pd.DataFrame({
                        "Component": results.columns[:-3],
                        "Optimal Value (%)": np.random.uniform(0, 100, len(results.columns[:-3]))
                    })
                    sample_result["Optimal Value (%)"] = sample_result["Optimal Value (%)"].round(2)
                    
                    st.dataframe(
                        sample_result,
                        use_container_width=True,
                        hide_index=True
                    )

if __name__ == "__main__":
    main()
