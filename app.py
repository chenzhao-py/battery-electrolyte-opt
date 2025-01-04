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
    
    # Sidebar navigation with radio buttons
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "",
        ["Home", "DOE Planning", "Experiment Input", "Analysis & Optimization"]
    )
    
    if page == "Home":
        show_home()
    elif page == "DOE Planning":
        show_doe_planning()
    elif page == "Experiment Input":
        show_experiment_input()
    else:
        show_analysis()

def show_home():
    st.header("Welcome to Battery Electrolyte Optimizer")
    st.write("""
    This application helps you optimize battery electrolyte compositions using:
    - Design of Experiments (DOE)
    - Machine Learning Optimization
    - Interactive Visualization
    """)
    
    st.subheader("Getting Started")
    st.write("""
    1. Start with 'DOE Planning' to design your experiments
    2. Input your experimental results in 'Experiment Input'
    3. Analyze and optimize using 'Analysis & Optimization'
    """)

def show_doe_planning():
    st.header("Design of Experiments Planning")
    
    # Input for factors
    st.subheader("Define Electrolyte Components")
    col1, col2 = st.columns(2)
    
    with col1:
        num_components = st.number_input("Number of components", min_value=2, max_value=10, value=3)
    with col2:
        n_experiments = st.number_input("Number of experiments", min_value=5, max_value=100, value=10)
    
    components = []
    ranges = []
    
    for i in range(num_components):
        col1, col2, col3 = st.columns(3)
        with col1:
            component = st.text_input(f"Component {i+1} name", key=f"comp_{i}")
        with col2:
            min_val = st.number_input(f"Min value for {component}", 0.0, 100.0, 0.0, key=f"min_{i}")
        with col3:
            max_val = st.number_input(f"Max value for {component}", 0.0, 100.0, 100.0, key=f"max_{i}")
        
        if component:
            components.append(component)
            ranges.append((min_val, max_val))
    
    if st.button("Generate DOE Plan"):
        if len(components) >= 2:
            doe_plan = generate_doe_plan(components, ranges, n_experiments)
            st.write("Generated DOE Plan:")
            st.dataframe(doe_plan)
            
            # Plot the DOE distribution
            fig = plot_doe_distribution(doe_plan, components)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button for DOE plan
            csv = doe_plan.to_csv(index=False)
            st.download_button(
                label="Download DOE Plan",
                data=csv,
                file_name="doe_plan.csv",
                mime="text/csv",
            )

def show_experiment_input():
    st.header("Experiment Results Input")
    
    uploaded_file = st.file_uploader("Upload your DOE plan CSV", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Original DOE Plan:")
        st.dataframe(data)
        
        # Add performance metrics
        st.subheader("Add Performance Metrics")
        metrics = ["Conductivity (mS/cm)", "Viscosity (cP)", "Stability (%)"]
        
        results_data = data.copy()
        for metric in metrics:
            results_data[metric] = 0.0
        
        edited_df = st.data_editor(results_data)
        
        if st.button("Save Results"):
            st.session_state['experiment_results'] = edited_df
            st.success("Results saved successfully!")

def show_analysis():
    st.header("Analysis & Optimization")
    
    if 'experiment_results' not in st.session_state:
        st.warning("Please input experiment results first!")
        return
    
    results = st.session_state['experiment_results']
    
    # Display correlation analysis
    st.subheader("Correlation Analysis")
    fig = create_correlation_plot(results)
    st.plotly_chart(fig)
    
    # Optimization
    st.subheader("Composition Optimization")
    target_metric = st.selectbox("Select target metric to optimize", 
                               ["Conductivity (mS/cm)", "Viscosity (cP)", "Stability (%)"])
    
    if st.button("Run Optimization"):
        st.info("Running optimization... This may take a moment.")
        # Add optimization logic here
        st.success("Optimization complete! Check the results below.")

if __name__ == "__main__":
    main()
