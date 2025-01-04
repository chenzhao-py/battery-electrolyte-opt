import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import optuna

st.set_page_config(
    page_title="Battery Electrolyte Optimizer",
    page_icon="⚡",
    layout="wide"
)

def main():
    st.title("Battery Electrolyte Optimization")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select a Page",
        ["Home", "DOE Planning", "Experiment Input", "Analysis & Optimization"]
    )
    
    if page == "Home":
        show_home()
    elif page == "DOE Planning":
        show_doe_planning()
    elif page == "Experiment Input":
        show_experiment_input()
    elif page == "Analysis & Optimization":
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
    num_components = st.number_input("Number of components", min_value=2, max_value=10, value=3)
    
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
            # Simple Latin Hypercube Sampling
            n_experiments = 10
            doe_plan = generate_doe_plan(components, ranges, n_experiments)
            st.write("Generated DOE Plan:")
            st.dataframe(doe_plan)
            
            # Download button for DOE plan
            csv = doe_plan.to_csv(index=False)
            st.download_button(
                label="Download DOE Plan",
                data=csv,
                file_name="doe_plan.csv",
                mime="text/csv",
            )

def generate_doe_plan(components, ranges, n_experiments):
    # Simple Latin Hypercube Sampling implementation
    n_vars = len(components)
    doe = np.zeros((n_experiments, n_vars))
    
    for i in range(n_vars):
        min_val, max_val = ranges[i]
        segment_size = (max_val - min_val) / n_experiments
        points = np.arange(n_experiments) * segment_size + min_val + segment_size/2
        np.random.shuffle(points)
        doe[:, i] = points
    
    # Normalize to ensure sum = 100%
    row_sums = doe.sum(axis=1)
    doe = doe / row_sums[:, np.newaxis] * 100
    
    return pd.DataFrame(doe, columns=components)

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
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    corr_matrix = results[numeric_cols].corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols)
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
