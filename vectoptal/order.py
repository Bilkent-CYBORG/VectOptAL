from os import PathLike
from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from vectoptal.ordering_cone import OrderingCone, ConeTheta2D


class Order(ABC):
    def __init__(self, ordering_cone: OrderingCone) -> None:
        self.ordering_cone = ordering_cone

    def dominates(self, a, b):
        """Does a dominate b?"""
        return self.ordering_cone.is_inside(a-b)
    
    def get_pareto_set(self, elements: np.ndarray):
        assert elements.ndim == 2, "Elements array should be N-by-dim."
        is_pareto = np.arange(len(elements))

        # Next index in the is_pareto array to search for
        next_point_index = 0
        while next_point_index < len(elements):
            nondominated_point_mask = np.zeros(len(elements), dtype=bool)
            vj = elements[next_point_index].reshape(-1, 1)

            for i in range(len(elements)):
                vi = elements[i].reshape(-1, 1)
                nondominated_point_mask[i] = not self.dominates(vj, vi)

            nondominated_point_mask[next_point_index] = True

            # Remove dominated points
            is_pareto = is_pareto[nondominated_point_mask]
            elements = elements[nondominated_point_mask]

            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        return is_pareto

    def get_pareto_set_naive(self, elements):
        pareto_indices = []
        for el_i, el in enumerate(elements):
            for other_el in elements:
                if np.allclose(el, other_el):
                    continue

                if self.dominates(other_el, el):
                    break
            else:
                pareto_indices.append(el_i)
        
        return pareto_indices

class ComponentwiseOrder(Order):
    def __init__(self, dim: int) -> None:
        W = np.eye(dim)
        ordering_cone = OrderingCone(W)
        super().__init__(ordering_cone)

class ConeTheta2DOrder(Order):
    def __init__(self, cone_degree) -> None:
        self.cone_degree = cone_degree
        ordering_cone = ConeTheta2D(cone_degree)
    
        super().__init__(ordering_cone)
    
    def plot(self, path: Optional[Union[str, PathLike]]=None):
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

        W_plt = self.ordering_cone.W

        m1 = W_plt[0][0] / -W_plt[0][1]
        m2 = W_plt[1][0] / -W_plt[1][1]

        # For the x basis
        x_right = np.array([0, xlim[-1]])
        y_right = 0 + m1 * (x_right - 0)
        
        # For the y basis
        if self.cone_degree > 90:
            x_left = np.array([xlim[0], 0])
        else:
            x_left = np.array([0, xlim[-1]])
        y_left = 0 + m2 * (x_left - 0)

        if self.cone_degree > 90:
            verts = np.array([
                [0, 0],
                [x_left[0], y_left[0]],
                [xlim[1], ylim[1]],
                [x_right[-1], y_right[-1]],
            ])
        elif self.cone_degree == 90:
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
        polygon = Polygon(
            verts, closed=True, alpha=0.35, color="mediumslateblue",
            label=rf"Cone $\theta={self.cone_degree}^\circ$"
        )
        ax.add_patch(polygon)
        ax.legend(loc="lower left")
        
        if path:
            plt.tight_layout()
            plt.savefig(path)

        return fig

    def plot_pareto_set(self, elements: np.ndarray, path: Optional[Union[str, PathLike]]=None):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        pareto_indices = self.get_pareto_set(elements)

        mask = np.ones(len(elements), dtype=np.uint8)
        mask[pareto_indices] = 0
        non_pareto_indices = np.nonzero(mask)
        ax.scatter(
            elements[pareto_indices][:, 0],
            elements[pareto_indices][:, 1], c="mediumslateblue", label="Pareto", alpha=0.6
        )
        ax.scatter(
            elements[non_pareto_indices][:, 0],
            elements[non_pareto_indices][:, 1], c="tab:blue", label="Non Pareto", alpha=0.6
        )

        ax.set(xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        ax.spines['bottom'].set_position('center')
        ax.spines['left'].set_position('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc="lower left")
        
        if path:
            plt.tight_layout()
            plt.savefig(path)

        return fig

class ConeOrder3D(Order):
    def __init__(self, cone_type: str) -> None:
        """
        :param cone_type: one of ['acute', 'right', 'obtuse']
        """
        if cone_type == 'acute':
            W = np.array([
                [1.0, -2, 4],
                [4, 1.0, -2],
                [-2, 4, 1.0],
            ])
            norm = np.linalg.norm(W[0])
            W /= norm
        elif cone_type == 'right':
            W = np.eye(3)
        elif cone_type == 'obtuse':
            W = np.array([
                [1, 0.4, 1.6],
                [1.6, 1, 0.4],
                [0.4, 1.6, 1],
            ])
            norm = np.linalg.norm(W[0])
            W /= norm
        ordering_cone = OrderingCone(W)
    
        super().__init__(ordering_cone)

class ConeOrder3DIceCream(Order):
    def __init__(self, cone_degree, num_halfspace) -> None:
        """
        :param cone_type: one of ['acute', 'right', 'obtuse']
        """
        
        W = self.compute_ice_cream_cone(num_halfspace, cone_degree)
        ordering_cone = OrderingCone(W)
    
        super().__init__(ordering_cone)

    def compute_ice_cream_cone(self, K, theta):
        delta_angle = 2 * np.pi / K
        theta_rad = np.pi/2 - np.radians(theta)

        radius = np.tan(theta_rad)
        W = []
        for i in range(K):
            angle = i * delta_angle
            rotated_ny = radius * np.sin(angle)
            rotated_nx = radius * np.cos(angle)
            W.append([rotated_nx, rotated_ny, 1])
        W = np.array(W)
        
        rot_axis = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        rot_rad = np.pi/4
        C = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0],
        ])

        r = np.eye(3) + C*np.sin(rot_rad) + (C @ C)*(1 - np.cos(rot_rad))
        W = (r @ W.T).T

        # Normalize half plane normal vectors
        for i in range(K):
            W[i] = W[i] / np.linalg.norm(W[i])

        return W

