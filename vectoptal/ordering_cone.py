from abc import ABC, abstractmethod

import numpy as np

from vectoptal.utils import get_2d_w, get_alpha_vec


class OrderingCone(ABC):
    def __init__(self, W: np.ndarray) -> None:
        """
        Ordering cone in the format C := {x | Wx >= 0}
        """
        self.W = W
        self.alpha = get_alpha_vec(W)

    def __eq__(self, other):
        return np.allclose(self.W, other.W)

    def is_inside(self, x: np.ndarray) -> bool:
        return (self.W @ x >= 0).all()

    @property
    @abstractmethod
    def beta(self) -> float:
        """Complexity of cone"""
        return NotImplementedError

class ConeTheta2D(OrderingCone):
    def __init__(self, cone_degree: float) -> None:
        self.cone_degree = cone_degree
        W = get_2d_w(cone_degree)

        super().__init__(W)
    
    @property
    def beta(self) -> float:
        cone_rad = (self.cone_degree / 180) * np.pi
        if cone_rad < np.pi/2:
            return 1/np.sin(cone_rad)
        else:
            return 1.

    def plot(self):
        pass


if __name__ == "__main__":
    cone = ConeTheta2D(90)
    print(cone.beta)
