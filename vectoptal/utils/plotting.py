from os import PathLike
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

# from vectoptal.order import Order
# from vectoptal.ordering_cone import OrderingCone
# from vectoptal.utils import get_2d_w


def plot_2d_cone(ordering_cone, path: Optional[Union[str, PathLike]]=None):
    """
    Plot the 2D cone with the given cone degree.
    """
    xlim = [-5, 5]
    ylim = [-5, 5]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    W = ordering_cone.W
    cone_degree = ordering_cone.cone_degree if hasattr(ordering_cone, "cone_degree") else 90

    m1 = W[0][0] / -W[0][1] if cone_degree != 90 else 0
    m2 = W[1][0] / -W[1][1]

    # For the x basis
    x_right = np.array([0, xlim[-1]])
    y_right = 0 + m1 * (x_right - 0)
    
    # For the y basis
    if cone_degree > 90:
        x_left = np.array([xlim[0], 0])
    else:
        x_left = np.array([0, xlim[-1]])
    y_left = 0 + m2 * (x_left - 0)

    if cone_degree > 90:
        verts = np.array([
            [0, 0],
            [x_left[0], y_left[0]],
            [xlim[1], ylim[1]],
            [x_right[-1], y_right[-1]],
        ])
    elif cone_degree == 90:
        verts = np.array([
            [0, 0],
            [0, ylim[1]],
            [xlim[1], ylim[1]],
            [xlim[1], 0],
        ])
    else:
        verts = np.array([
            [0, 0],
            [x_left[-1], y_left[-1]],
            [xlim[1], ylim[1]],
            [x_right[-1], y_right[-1]],
        ])

    ax.add_patch(Polygon(verts, color='blue', alpha=0.5))

    if path is not None:
        fig.savefig(path)
    
    return fig

def plot_3d_cone(ordering_cone, path: Optional[Union[str, PathLike]]=None):
    """
    Given a W matrix of shape N-by-3 representing the cone, plot the 3D region of the cone.
    Rows of the matrix represent the normal vectors of the constraints.
    Create a list that includes the corners of the region, than plot the region using Poly3DCollection.
    Use boundaries for the plot limits.
    """

    xlim = [-5, 5]
    ylim = [-5, 5]
    zlim = [-5, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[], zticks=[], zticklabels=[])
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add X, Y, and Z axis lines at the middle of the region
    ax.plot([xlim[0], xlim[1]], [0, 0], [0, 0], color='black')
    ax.plot([0, 0], [ylim[0], ylim[1]], [0, 0], color='black')
    ax.plot([0, 0], [0, 0], [zlim[0], zlim[1]], color='black')

    x_pts = np.linspace(xlim[0], xlim[1], 25)
    y_pts = np.linspace(ylim[0], ylim[1], 25)
    z_pts = np.linspace(zlim[0], zlim[1], 25)

    X, Y, Z = np.meshgrid(x_pts, y_pts, z_pts)
    pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    pts = pts[ordering_cone.is_inside(pts)]
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    ax.scatter(X, Y, Z, alpha=0.3, c='blue', s=8)

    if path is not None:
        fig.savefig(path)

    return fig

def plot_pareto_front(
    order, elements: np.ndarray, path: Optional[Union[str, PathLike]]=None
):
    dim = elements.shape[1]
    assert elements.ndim == 2, "Elements array should be N-by-dim."
    assert dim in [2, 3], "Only 2D and 3D plots are supported."
    fig = plt.figure(figsize=(8, 5))

    pareto_indices = order.get_pareto_set(elements)

    mask = np.ones(len(elements), dtype=np.uint8)
    mask[pareto_indices] = 0
    non_pareto_indices = np.nonzero(mask)
    if dim == 2:
        ax = fig.add_subplot(111)

        ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        ax.spines['bottom'].set_position('center')
        ax.spines['left'].set_position('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.scatter(
            elements[pareto_indices][:, 0],
            elements[pareto_indices][:, 1], c="mediumslateblue", label="Pareto", alpha=0.6
        )
        ax.scatter(
            elements[non_pareto_indices][:, 0],
            elements[non_pareto_indices][:, 1], c="tab:blue", label="Non Pareto", alpha=0.6
        )
    else:
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[], zticks=[], zticklabels=[])
        ax.spines['bottom'].set_position('center')
        ax.spines['left'].set_position('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.scatter(
            elements[pareto_indices][:, 0],
            elements[pareto_indices][:, 1],
            elements[pareto_indices][:, 2], c="mediumslateblue", label="Pareto", alpha=0.6
        )
        ax.scatter(
            elements[non_pareto_indices][:, 0],
            elements[non_pareto_indices][:, 1],
            elements[non_pareto_indices][:, 2], c="tab:blue", label="Non Pareto", alpha=0.6
        )

        # Add X, Y, and Z axis lines at the middle of the region
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], [0, 0], color='black')
        ax.plot([0, 0], [ax.get_ylim()[0], ax.get_ylim()[1]], [0, 0], color='black')
        ax.plot([0, 0], [0, 0], [ax.get_xlim()[0], ax.get_zlim()[1]], color='black')

    ax.legend(loc="lower left")
    fig.tight_layout()
    
    if path is not None:
        fig.savefig(path)

    return fig
