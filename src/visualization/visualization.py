"""Functions to visualize point clouds. Heavily based on functions in
https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263 """

import math

from typing import Any

import plotly.graph_objects as go
import pyvista as pv
import numpy as np

def visualize_rotate(data: list[Any], 
                     x_eye: float = 1.25, 
                     y_eye: float = 1.25, 
                     z_eye: float = 0.8, 
                     lower_bnd: float = 0, 
                     upper_bnd: float = 10.26, 
                     step: float = 0.1) -> go.Figure:
    """Visualizes a point cloud in given data as a 3D plotly figure."""

    def rotate_z(x: float, y: float, z: float, theta: float) -> tuple[float, float, float]:
        """Returns a vector rotated about the z axis by angle theta"""
        omega = x + 1j*y
        omega_rot = omega * np.exp(1j * theta)
        return np.real(omega_rot), np.imag(omega_rot), z

    frames = list()

    for theta in np.arange(lower_bnd, upper_bnd, step):
        x_e, y_e, z_e = rotate_z(x_eye, y_eye, z_eye, -theta)
        frames.append(dict(
            layout=dict(
                scene=dict(
                    camera=dict(
                        eye=dict(
                            x=x_e, y=y_e, z=z_e
                        )
                    )
                )
            )
        ))
    
    buttons_list = [dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(
                            duration=50, redraw=True
                            ),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                )]
   
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.8, y=1,
                    xanchor="left", yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=buttons_list
                )
            ]
        ),
        frames=frames
    )
    return fig

def show_pointcloud_notebook(x_s: np.ndarray[Any, Any], 
                             y_s: np.ndarray[Any, Any], 
                             z_s: np.ndarray[Any, Any], 
                             label: str = "") -> None:
    """Plots a point cloud given coordinates, in a notebook environment (e.g. Jupyter notebook)""" 
    data = [go.Scatter3d(
        x=x_s, y=y_s, z=z_s, mode="markers"
    )]

    fig = visualize_rotate(data)   
    fig.update_traces(
        marker=dict(
            size=2,
            line=dict(
                width=2,
                color="DarkSlateGrey"
            )
        ),
        selector=dict(mode="markers")
    )
    fig.update_layout(
            title=dict(text=label)
    )
    return fig

def show_pointcloud(pointcloud: np.ndarray[Any, Any], label: str = "") -> None:
    """Render a pointcloud in a new window."""
    assert pointcloud.shape[1] == 3, "Point cloud shape must be (N, 3)"

    mesh = pv.PolyData(pointcloud)
    mesh.plot(point_size=20, style="points", text=label)

def show_pointcloud_grid(
        points: np.ndarray[Any, Any], 
        labels: np.ndarray[Any, Any],
        label_to_class: dict[int, str],
    ) -> None:
    """Generates a grid of renderings of pointclouds from a batch of pointclouds."""
    
    assert len(points.shape) == 3, "Must supply a batch of pointclouds of shape (B, N, 3)"
    assert points.shape[2] == 3, "Point cloud shape must be (N, 3)"
    assert points.shape[0] == labels.shape[0], "Points and labels must be of same length."
    assert points.shape[0] <= 9, "Can only plot at most 9 pointclouds."
    
    num_rows = math.sqrt(points.shape[0])
    num_rows = int(num_rows) + 1 if (num_rows - int(num_rows) > 0) else int(num_rows)

    plotter = pv.Plotter(shape=(num_rows, num_rows))

    for count_i in range(num_rows):
        for count_j in range(num_rows):
            idx =  num_rows * count_i + count_j
            if idx >= points.shape[0]:
                break

            title = label_to_class[labels[idx]]

            plotter.subplot(count_i, count_j)
            plotter.add_points(points[idx], render_points_as_spheres=True, point_size=10)
            plotter.add_text(title, position="upper_left", font_size=12)
            plotter.show_bounds(all_edges=True)
    
    plotter.show()
    plotter.close()

