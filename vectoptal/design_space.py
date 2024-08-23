import os
from typing import Optional
from itertools import product
from abc import ABC, abstractmethod

import numpy as np

from vectoptal.confidence_region import (
    RectangularConfidenceRegion, EllipsoidalConfidenceRegion,
)
from vectoptal.models import Model, GPModel


class DesignSpace(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, model: Model):
        pass

class DiscreteDesignSpace(DesignSpace):
    def __init__(self, points, objective_dim, confidence_type='hyperrectangle') -> None:
        super().__init__()

        if confidence_type == 'hyperrectangle':
            confidence_cls = RectangularConfidenceRegion
        elif confidence_type == 'hyperellipsoid':
            confidence_cls = EllipsoidalConfidenceRegion
        else:
            raise NotImplementedError

        self.cardinality = len(points)

        self.points = points
        self.confidence_regions = []
        for _ in range(len(points)):
            self.confidence_regions.append(confidence_cls(objective_dim))

    def update(self, model: Model, scale: np.ndarray, indices_to_update: Optional[list]=None):
        if indices_to_update is None:
            indices_to_update = list(range(self.cardinality))

        mus, covs = model.predict(self.points[indices_to_update])
        for pt_i, mu, cov in zip(indices_to_update, mus, covs):
            self.confidence_regions[pt_i].update(mu, cov, scale)

class AdaptivelyDiscretizedDesignSpace(DesignSpace):
    def __init__(
        self, domain_dim, objective_dim, delta, max_depth, confidence_type='hyperrectangle'
    ) -> None:
        super().__init__()

        if confidence_type == 'hyperrectangle':
            confidence_cls = RectangularConfidenceRegion
        else:
            raise NotImplementedError(
                "Only hyperrectangular confidence region is supported for adaptive discretization."
            )

        self.domain_dim = domain_dim
        self.objective_dim = objective_dim
        self.max_depth = max_depth
        self.delta = delta
        self.confidence_cls = confidence_cls

        # TODO: Consider creating a Design class.
        # If parent design becomes necessary, a reference can be held as an attribute.
        self.points = np.array([[0.5] * domain_dim])
        self.point_depths = [1]
        self.cells = [[[0, 1] for _ in range(domain_dim)]]
        self.confidence_regions = [confidence_cls(objective_dim)]

    def update(self, model: GPModel, scale: np.ndarray, indices_to_update: Optional[list]=None):
        if indices_to_update is None:
            indices_to_update = list(range(len(self.points)))
        
        mus, covs = model.predict(self.points[indices_to_update])
        for pt_i, mu, cov in zip(indices_to_update, mus, covs):
            self.confidence_regions[pt_i].update(mu, cov, scale)

    def refine_design(self, index_to_refine: int) -> list:
        return self.get_child_designs(index_to_refine)

    def get_child_designs(self, design_index: int) -> list:
        options = []
        for dim_i in range(self.domain_dim):
            options.append([
                [
                    self.cells[design_index][dim_i][0],
                    (self.cells[design_index][dim_i][0] + self.cells[design_index][dim_i][1])/2
                ],
                [
                    (self.cells[design_index][dim_i][0] + self.cells[design_index][dim_i][1])/2,
                    self.cells[design_index][dim_i][1]
                ]
            ])
        new_bounds = list(map(list, product(*options)))

        list_children = []
        for bound in new_bounds:
            list_children.append(len(self.points))
            
            x = np.array(bound, dtype=float).mean(axis=1)
            self.points = np.append(self.points, [x], axis=0)
            self.point_depths.append(self.point_depths[design_index]+1)
            self.cells.append(bound)
            self.confidence_regions.append(
                # TODO: Create a copy constructor for ConfidenceRegion
                self.confidence_cls(
                    self.objective_dim,
                    self.confidence_regions[design_index].lower,
                    self.confidence_regions[design_index].upper,
                )
            )

        return list_children

    def should_refine_design(self, model: GPModel, design_index: int, scale: np.ndarray) -> bool:
        vh = self.calculate_design_vh(model, design_index)
        if self.point_depths[design_index] >= self.max_depth:
            return False
        
        mu, cov = model.predict(self.points[[design_index]])
        std = np.sqrt(np.diag(cov.squeeze()))

        return np.all(scale * np.linalg.norm(std) <= np.linalg.norm(vh))

    def calculate_design_vh(
        self, model: GPModel, design_index: int, depth_offset: int = 0
    ) -> np.ndarray:
        # TODO: magic number
        rho = 0.5
        alpha = 1
        N = 4
        
        lengthscales, variances = model.get_lengthscale_and_var()
        Vh = np.zeros([self.objective_dim, 1])
        depth = self.point_depths[design_index] + depth_offset
        diam_x = np.sqrt(self.domain_dim)
        v1 = 0.5 * np.sqrt(self.domain_dim)

        for i in range(self.objective_dim):
            if model.get_kernel_type() == "RBF":
                Cki = np.sqrt(variances[i]) / lengthscales[i]
            else:
                raise ValueError("Kernel type undefined.")

            C1 = np.power((diam_x + 1) * diam_x / 2, self.domain_dim) * np.power(Cki, 1 / alpha)
            C2 = 2 * np.log(2 * np.power(C1 , 2) * np.power(np.pi, 2) / 6)
            C3 = 1. + 2.7 * np.sqrt(2 * self.domain_dim * alpha * np.log(2))

            term1 = Cki * np.power(v1 * np.power(rho, depth), alpha)
            term2 = np.log(
                2 * np.power(depth + 1, 2) * np.power(np.pi, 2)
                * self.objective_dim / (6 * self.delta)
            )
            term3 = depth * np.log(N)
            term4 = np.maximum(0, -4 * self.domain_dim / alpha * np.log(term1))

            Vh[i] = 4 * term1 * (np.sqrt(C2 + 2 * term2 + term3 + term4) + C3)
        return Vh
