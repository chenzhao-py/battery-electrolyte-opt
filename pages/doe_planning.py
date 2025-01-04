import streamlit as st
import pandas as pd
import numpy as np
from visualization.doe_plots import plot_doe_distribution

def generate_doe_plan(components, ranges, n_experiments):
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
    
    # Round to 2 decimal places
    doe = np.round(doe, 2)
    
    return pd.DataFrame(doe, columns=components)

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
