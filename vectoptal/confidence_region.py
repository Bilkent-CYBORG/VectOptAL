from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import cvxpy as cp
import scipy as sp

from vectoptal.order import Order
from vectoptal.utils import (
    hyperrectangle_check_intersection, hyperrectangle_get_vertices, is_pt_in_extended_polytope,
    hyperrectangle_get_region_matrix
)


class ConfidenceRegion(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self):
        pass

class RectangularConfidenceRegion(ConfidenceRegion):
    def __init__(
        self, dim: int, lower: Optional[np.ndarray]=None, upper: Optional[np.ndarray]=None,
        intersect_iteratively: Optional[bool]=True
    ) -> None:
        super().__init__()

        self.intersect_iteratively = intersect_iteratively

        if lower is not None and upper is not None:
            self.lower = lower
            self.upper = upper
        else:
            self.lower = np.array([-1e12] * dim)  # TODO: Magic large number.
            self.upper = np.array([ 1e12] * dim)

    def diagonal(self):
        """Returns the euclidean norm of the diagonal of the hyperrectangle"""
        return np.linalg.norm(self.upper - self.lower)

    def update(self, mean: np.ndarray, covariance: np.ndarray, scale: np.ndarray=np.array(1.)):
        assert covariance.shape[-1] == covariance.shape[-2], "Covariance matrix must be square."
        std = np.sqrt(np.diag(covariance.squeeze()))

        L = mean - std * scale
        U = mean + std * scale

        if self.intersect_iteratively:
            self.intersect(L, U)
        else:
            self.lower = L
            self.upper = U

    @property
    def center(self):
        return (self.lower + self.upper) / 2

    def intersect(self, lower: np.ndarray, upper: np.ndarray):
        # if the two rectangles overlap
        if hyperrectangle_check_intersection(self.lower, self.upper, lower, upper):
            self.lower = np.maximum(self.lower, lower)
            self.upper = np.minimum(self.upper, upper)
        else:
        # if there is no intersection, then use the new hyperrectangle
            self.lower = lower
            self.upper = upper

    @classmethod
    def is_dominated(cls, order: Order, obj1, obj2, slackness: np.ndarray):
        verts1 = hyperrectangle_get_vertices(obj1.lower, obj1.upper)
        verts2 = hyperrectangle_get_vertices(obj2.lower, obj2.upper)

        for vert1 in verts1:
            for vert2 in verts2:
                if not order.dominates(vert2 + slackness, vert1):
                    return False
        return True

    @classmethod
    def check_dominates(cls, order: Order, obj1, obj2, slackness: np.ndarray=np.array(0.)):
        cone_matrix = order.ordering_cone.W

        verts1 = hyperrectangle_get_vertices(obj1.lower, obj1.upper) @ cone_matrix.transpose()
        verts2 = hyperrectangle_get_vertices(obj2.lower, obj2.upper) @ cone_matrix.transpose()

        # For every vertex of x', check if element of x+C. Return False if any vertex is not.
        for ref_point in verts1:
            if is_pt_in_extended_polytope(ref_point, verts2) is False:
                return False

        return True

    @classmethod
    def is_covered(cls, order, obj1, obj2, slackness):
        cone_matrix = order.ordering_cone.W
        n = cone_matrix.shape[1]

        z_point = cp.Variable(n)
        z_point2 = cp.Variable(n)

        # Represent rectangular confidence regions as matrices
        obj1_matrix, obj1_boundary = hyperrectangle_get_region_matrix(obj1.lower, obj1.upper)
        obj2_matrix, obj2_boundary = hyperrectangle_get_region_matrix(obj2.lower, obj2.upper)

        constraints = [
            obj1_matrix @ z_point >= obj1_boundary,
            obj2_matrix @ z_point2 >= obj2_boundary,
            cone_matrix @ (z_point2 - z_point - slackness) >= 0
        ]
        
        prob = cp.Problem(cp.Minimize(0), constraints=constraints)

        try:
            prob.solve(solver="OSQP", max_iter=10000)
        except:
            prob.solve(solver = "ECOS")

        if prob.status == None:
            return True

        condition = prob.status == 'optimal'  
        return condition

class EllipsoidalConfidenceRegion(ConfidenceRegion):
    def __init__(
        self, dim, center: Optional[np.ndarray]=None, sigma: Optional[np.ndarray]=None,
        alpha: Optional[float]=None
    ) -> None:
        super().__init__()

        if center is not None and sigma is not None and alpha is not None:
            self.center = center
            self.sigma = sigma
            self.alpha = alpha
        else:
            self.center = np.zeros(dim)
            self.sigma = np.eye(dim)
            self.alpha = 1.0

    def update(self, mean: np.ndarray, covariance: np.ndarray, scale: np.ndarray=np.array(1.)):
        assert covariance.shape[-1] == covariance.shape[-2], "Covariance matrix must be square."

        self.center = mean
        self.sigma = covariance
        self.alpha = scale

    @classmethod
    def is_dominated(cls, order: Order, obj1, obj2, slackness: np.ndarray):
        output_dim = len(obj1.center)
        cone_matrix = order.ordering_cone.W

        mux = cp.Variable(output_dim)
        muy = cp.Variable(output_dim)

        # # quad_form( A * x - b, Q ) <= 1
        # cons1 = cp.quad_form((mux - mx).T, np.linalg.inv(sigma_x)) <= alpha
        # cons2 = cp.quad_form((muy - my).T, np.linalg.inv(sigma_y)) <= alpha
        # # norm( Qsqrt * ( A * x - b ) ) <= 1
        cons1 = cp.norm(
            sp.linalg.sqrtm(np.linalg.inv(obj1.sigma)) @ (mux - obj1.center).T
        ) <= obj1.alpha
        cons2 = cp.norm(
            sp.linalg.sqrtm(np.linalg.inv(obj2.sigma)) @ (muy - obj2.center).T
        ) <= obj2.alpha
        
        constraints = [cons1, cons2]

        for n in range(cone_matrix.shape[0]):
            objective = cp.Minimize(cone_matrix[n] @ (muy - mux))
            
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver="ECOS")
            except:
                prob.solve(solver="MOSEK")

            if prob.value < -slackness:
                return False

        return True

    @classmethod
    def check_dominates(cls, order: Order, obj1, obj2, slackness: np.ndarray=np.array(0.)):
        raise NotImplementedError

    @classmethod
    def is_covered(cls, order, obj1, obj2, slackness):
        cone_matrix = order.ordering_cone.W
        output_dim = cone_matrix.shape[1]
        mux = cp.Variable(output_dim)
        muy = cp.Variable(output_dim)

        # # norm( Qsqrt * ( A * x - b ) ) <= 1
        cons1 = cp.norm(
            sp.linalg.sqrtm(np.linalg.inv(obj1.sigma)) @ (mux - obj1.center).T
        ) <= obj1.alpha
        cons2 = cp.norm(
            sp.linalg.sqrtm(np.linalg.inv(obj2.sigma)) @ (muy - obj2.center).T
        ) <= obj2.alpha
        cons3 = cone_matrix @ (muy - mux) >= slackness

        constraints = [cons1, cons2, cons3]

        objective = cp.Minimize(0)

        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver="ECOS")
        except:
            prob.solve(solver="MOSEK")

        if "infeasible" in prob.status:
            return False
        else:
            return True

def confidence_region_is_dominated(order, region1, region2, slackness) -> bool:
    if isinstance(region1, RectangularConfidenceRegion):
        return RectangularConfidenceRegion.is_dominated(order, region1, region2, slackness)
    elif isinstance(region1, EllipsoidalConfidenceRegion):
        return EllipsoidalConfidenceRegion.is_dominated(order, region1, region2, slackness)
    else:
        raise NotImplementedError

def confidence_region_check_dominates(order, region1, region2) -> bool:
    if isinstance(region1, RectangularConfidenceRegion):
        return RectangularConfidenceRegion.check_dominates(order, region1, region2)
    elif isinstance(region1, EllipsoidalConfidenceRegion):
        return EllipsoidalConfidenceRegion.check_dominates(order, region1, region2)
    else:
        raise NotImplementedError

def confidence_region_is_covered(order, region1, region2, slackness) -> bool:
    # TODO: is_covered may be a bad name. Maybe is_not_dominated?
    if isinstance(region1, RectangularConfidenceRegion):
        return RectangularConfidenceRegion.is_covered(order, region1, region2, slackness)
    elif isinstance(region1, EllipsoidalConfidenceRegion):
        return EllipsoidalConfidenceRegion.is_covered(order, region1, region2, slackness)
    else:
        raise NotImplementedError
