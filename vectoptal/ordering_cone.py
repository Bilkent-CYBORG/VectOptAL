from os import PathLike
from typing import Union, Optional

import numpy as np

from vectoptal.utils import get_2d_w, get_alpha_vec
from vectoptal.utils.plotting import plot_2d_theta_cone, plot_2d_cone, plot_3d_cone


class OrderingCone:
    def __init__(self, W: np.ndarray) -> None:
        """
        Ordering cone in the format C := {x | Wx >= 0}
        """
        self.W = W
        self.dim = W.shape[1]
        self.alpha = get_alpha_vec(W)

    def __eq__(self, other: "OrderingCone") -> bool:
        return np.allclose(self.W, other.W)

    def is_inside(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x @ self.W.T >= 0).all(axis=-1)

    def plot(self, path: Optional[Union[str, PathLike]] = None):
        if self.dim not in [2, 3]:
            raise ValueError("Only 2D and 3D plots are supported.")

        if self.dim == 2:
            fig = plot_2d_cone(self.is_inside, path)
        else:
            fig = plot_3d_cone(self.is_inside, path)

        return fig


class ConeTheta2D(OrderingCone):
    def __init__(self, cone_degree: float) -> None:
        self.cone_degree = cone_degree
        W = get_2d_w(cone_degree)

        super().__init__(W)

    def plot(self, path: Optional[Union[str, PathLike]] = None):
        fig = plot_2d_theta_cone(self.cone_degree, path)
        return fig

    @property
    def beta(self) -> float:
        cone_rad = (self.cone_degree / 180) * np.pi
        if cone_rad < np.pi / 2:
            return 1 / np.sin(cone_rad)
        else:
            return 1.0
