import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict
from scipy.spatial import ConvexHull

def create_surrogate_model(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """Create and fit a Gaussian Process surrogate model."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    model.fit(X, y)
    return model

def expected_improvement(X: np.ndarray, model: GaussianProcessRegressor, y_best: float, 
                        greater_is_better: bool = True, xi: float = 0.01) -> np.ndarray:
    """Calculate expected improvement at points X."""
    mu, sigma = model.predict(X, return_std=True)
    
    if greater_is_better:
        improvement = mu - y_best - xi
    else:
        improvement = y_best - mu - xi
        
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

def suggest_next_samples(models: Dict[str, GaussianProcessRegressor], 
                        bounds: List[Tuple[float, float]], 
                        n_samples: int = 5,
                        objectives_info: Dict[str, bool] = None) -> pd.DataFrame:
    """Suggest next samples using multi-objective Bayesian optimization."""
    
    def objective(x):
        x = x.reshape(1, -1)
        scores = []
        for obj_name, model in models.items():
            pred, std = model.predict(x, return_std=True)
            greater_is_better = objectives_info[obj_name]
            ei = expected_improvement(x, model, 
                                   np.max(model.y_train_) if greater_is_better else np.min(model.y_train_),
                                   greater_is_better)
            scores.append(-ei)  # Minimize negative EI
        return np.mean(scores)  # Simple scalarization
    
    suggested_points = []
    for _ in range(n_samples):
        x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        suggested_points.append(res.x)
    
    return np.array(suggested_points)

def plot_pareto_front(data: pd.DataFrame, objectives: List[str], objectives_info: Dict[str, bool]) -> go.Figure:
    """
    Plot the Pareto front for 2D or 3D objectives.
    For more than 3 objectives, creates a parallel coordinates plot.
    
    Args:
        data: DataFrame with the data
        objectives: List of objective names
        objectives_info: Dictionary mapping objective names to whether they should be maximized
    """
    points = data[objectives].values
    
    # Transform objectives that need to be minimized
    for i, obj in enumerate(objectives):
        if not objectives_info[obj]:  # if minimize
            points[:, i] = -points[:, i]
    
    # Calculate Pareto front
    is_pareto = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = ~np.all(points[is_pareto] >= point, axis=1) | (points[is_pareto] == point).all(axis=1)
    
    pareto_points = points[is_pareto]
    
    # Transform back for plotting
    for i, obj in enumerate(objectives):
        if not objectives_info[obj]:  # if was minimized
            points[:, i] = -points[:, i]
            pareto_points[:, i] = -pareto_points[:, i]
    
    if len(objectives) == 2:
        # 2D Scatter plot
        fig = go.Figure()
        
        # Add all points
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            name='All Points',
            marker=dict(color='blue', size=8)
        ))
        
        # Add Pareto front points
        # Sort points for proper line connection
        if len(pareto_points) > 0:
            idx = np.argsort(pareto_points[:, 0])
            pareto_points = pareto_points[idx]
            
            fig.add_trace(go.Scatter(
                x=pareto_points[:, 0],
                y=pareto_points[:, 1],
                mode='markers+lines',
                name='Pareto Front',
                marker=dict(color='red', size=10),
                line=dict(color='red', dash='dot')
            ))
        
        fig.update_layout(
            title='2D Pareto Front',
            xaxis_title=f"{objectives[0]} ({'maximize' if objectives_info[objectives[0]] else 'minimize'})",
            yaxis_title=f"{objectives[1]} ({'maximize' if objectives_info[objectives[1]] else 'minimize'})",
            showlegend=True
        )
        
    elif len(objectives) == 3:
        # 3D Scatter plot
        fig = go.Figure()
        
        # Add all points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name='All Points',
            marker=dict(
                color='blue',
                size=4,
                opacity=0.6
            )
        ))
        
        # Add Pareto front points
        if len(pareto_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=pareto_points[:, 0],
                y=pareto_points[:, 1],
                z=pareto_points[:, 2],
                mode='markers',
                name='Pareto Front',
                marker=dict(
                    color='red',
                    size=6,
                    opacity=0.8
                )
            ))
            
            # Only attempt to create convex hull if we have enough points
            if len(pareto_points) >= 4:
                try:
                    hull = ConvexHull(pareto_points)
                    # Create triangles for the surface
                    vertices = pareto_points[hull.vertices]
                    # Add surface
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        opacity=0.3,
                        color='red',
                        name='Pareto Surface'
                    ))
                except Exception:
                    # If hull creation fails, skip the surface
                    pass
        
        fig.update_layout(
            title='3D Pareto Front',
            scene=dict(
                xaxis_title=f"{objectives[0]} ({'maximize' if objectives_info[objectives[0]] else 'minimize'})",
                yaxis_title=f"{objectives[1]} ({'maximize' if objectives_info[objectives[1]] else 'minimize'})",
                zaxis_title=f"{objectives[2]} ({'maximize' if objectives_info[objectives[2]] else 'minimize'})"
            ),
            showlegend=True
        )
        
    else:
        # Parallel coordinates plot for higher dimensions
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color='blue',
                    opacity=0.3
                ),
                dimensions=[
                    dict(
                        range=[data[obj].min(), data[obj].max()],
                        label=f"{obj} ({'maximize' if objectives_info[obj] else 'minimize'})",
                        values=data[obj]
                    ) for obj in objectives
                ]
            )
        )
        
        # Highlight Pareto optimal solutions
        if any(is_pareto):
            fig.add_trace(
                go.Parcoords(
                    line=dict(
                        color='red',
                        opacity=0.8
                    ),
                    dimensions=[
                        dict(
                            range=[data[obj].min(), data[obj].max()],
                            label=f"{obj} ({'maximize' if objectives_info[obj] else 'minimize'})",
                            values=data[objectives][is_pareto][obj]
                        ) for obj in objectives
                    ]
                )
            )
        
        fig.update_layout(
            title='Parallel Coordinates Plot of Pareto Solutions',
            showlegend=True
        )
    
    return fig

def plot_radar_chart(data: pd.DataFrame, objectives: List[str], objectives_info: Dict[str, bool]) -> go.Figure:
    """Create a radar chart for comparing multiple objectives."""
    # Normalize the data between 0 and 1 for each objective
    normalized_data = pd.DataFrame()
    for obj in objectives:
        if objectives_info[obj]:  # maximize
            normalized_data[obj] = (data[obj] - data[obj].min()) / (data[obj].max() - data[obj].min())
        else:  # minimize
            normalized_data[obj] = 1 - (data[obj] - data[obj].min()) / (data[obj].max() - data[obj].min())
    
    # Calculate mean and best values
    mean_values = normalized_data.mean()
    best_values = normalized_data.max()
    
    fig = go.Figure()
    
    # Add mean performance
    fig.add_trace(go.Scatterpolar(
        r=mean_values.values,
        theta=objectives,
        fill='toself',
        name='Mean Performance'
    ))
    
    # Add best performance
    fig.add_trace(go.Scatterpolar(
        r=best_values.values,
        theta=objectives,
        fill='toself',
        name='Best Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='Radar Chart of Objective Performance'
    )
    
    return fig

def plot_optimization_history(history: pd.DataFrame, objective: str) -> go.Figure:
    """Plot the optimization history for a single objective."""
    fig = go.Figure()
    
    # Plot all points
    fig.add_trace(go.Scatter(
        x=np.arange(len(history)),
        y=history[objective],
        mode='markers+lines',
        name='Objective Value',
        marker=dict(size=8)
    ))
    
    # Plot best so far
    best_so_far = np.maximum.accumulate(history[objective])
    fig.add_trace(go.Scatter(
        x=np.arange(len(history)),
        y=best_so_far,
        mode='lines',
        name='Best So Far',
        line=dict(color='red', dash='dot')
    ))
    
    fig.update_layout(
        title=f'Optimization History - {objective}',
        xaxis_title='Iteration',
        yaxis_title=objective,
        showlegend=True
    )
    
    return fig

def calculate_optimization_metrics(data: pd.DataFrame, objectives: List[str]) -> Dict:
    """Calculate optimization metrics."""
    metrics = {}
    
    for obj in objectives:
        metrics[obj] = {
            'best_value': data[obj].max(),
            'mean': data[obj].mean(),
            'std': data[obj].std(),
            'improvement': (data[obj].max() - data[obj].min()) / data[obj].min() * 100
        }
    
    return metrics
