import streamlit as st
import pandas as pd

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
