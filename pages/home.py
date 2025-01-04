import streamlit as st

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
