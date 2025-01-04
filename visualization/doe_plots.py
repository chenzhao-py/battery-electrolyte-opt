import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_doe_distribution(doe_data, components):
    n_components = len(components)
    
    if n_components == 3:
        return plot_ternary(doe_data, components)
    elif n_components == 4:
        return plot_quaternary(doe_data, components)
    else:
        return plot_parallel_coordinates(doe_data, components)

def plot_ternary(doe_data, components):
    fig = go.Figure(data=[
        go.Scatterternary(
            a=doe_data[components[0]],
            b=doe_data[components[1]],
            c=doe_data[components[2]],
            mode='markers',
            marker={'size': 10},
            text=[f"{components[0]}: {row[components[0]]:.2f}<br>"
                  f"{components[1]}: {row[components[1]]:.2f}<br>"
                  f"{components[2]}: {row[components[2]]:.2f}"
                  for _, row in doe_data.iterrows()],
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title=f"DOE Distribution - Ternary Plot",
        ternary={
            'aaxis': {'title': components[0]},
            'baxis': {'title': components[1]},
            'caxis': {'title': components[2]}
        }
    )
    return fig

def plot_quaternary(doe_data, components):
    # Create a parallel coordinates plot for 4 components
    fig = px.parallel_coordinates(
        doe_data,
        dimensions=components,
        title="DOE Distribution - Quaternary System"
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='white'
    )
    return fig

def plot_parallel_coordinates(doe_data, components):
    fig = px.parallel_coordinates(
        doe_data,
        dimensions=components,
        title=f"DOE Distribution - {len(components)} Components"
    )
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='white'
    )
    return fig
