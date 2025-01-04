import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def generate_doe_plan(components, ranges, n_experiments):
    """Generate a DOE plan using Latin Hypercube Sampling."""
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

def plot_doe_distribution(doe_data, components):
    """Plot DOE distribution based on number of components."""
    n_components = len(components)
    
    if n_components == 3:
        return plot_ternary(doe_data, components)
    elif n_components == 4:
        return plot_quaternary(doe_data, components)
    else:
        return plot_parallel_coordinates(doe_data, components)

def plot_ternary(doe_data, components):
    """Create a ternary plot for 3 components."""
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
    """Create a parallel coordinates plot for 4 components."""
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
    """Create a parallel coordinates plot for any number of components."""
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
