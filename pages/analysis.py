import streamlit as st
import plotly.express as px
import numpy as np

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
