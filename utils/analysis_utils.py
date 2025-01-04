import numpy as np
import plotly.express as px

def create_correlation_plot(results):
    """Create correlation matrix plot."""
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    corr_matrix = results[numeric_cols].corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=numeric_cols,
                    y=numeric_cols)
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    return fig
