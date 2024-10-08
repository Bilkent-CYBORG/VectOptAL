import copy
import logging

import numpy as np

from vectoptal.order import Order
from vectoptal.datasets import get_dataset
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset
from vectoptal.acquisition import SumVarianceAcquisition, optimize_acqf_discrete
from vectoptal.models import EmpiricalMeanVarModel
from vectoptal.confidence_region import (
    confidence_region_is_dominated,
    confidence_region_check_dominates,
    confidence_region_is_covered
)


class PaVeBa(PALAlgorithm):
    """
    Implement the Pareto Vector Bandits (PaVeBa) algorithm.

    :param float epsilon: Determines the accuracy of the PAC-learning framework.
    :param float delta: Determines the success probability of the PAC-learning framework.
    :param str dataset_name: Name of the dataset to be used.
    :param order: Order to be used.
    :param float noise_var: Variance of the Gaussian sampling noise.
    :param float conf_contraction: Contraction coefficient to shrink the confidence regions empirically.

    The algorithm sequentially samples design rewards with a multivariate white Gaussian noise whose diagonal
    entries are specified by the user.

    Returns None.

    Example:
        >>> from vectoptal.order import ComponentwiseOrder
        >>> from vectoptal.algorithms import PaVeBa
        >>>
        >>> epsilon, delta, noise_var = 0.01, 0.01, 0.01
        >>> dataset_name = "DiskBrake"
        >>> order_acute = ComponentwiseOrder(2)
        >>>
        >>> PaVeBa = PaVeBa(epsilon, delta, dataset_name, order_acute, noise_var)
        >>>
        >>> while True:
        >>>     is_done = PaVeBa.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_indices = PaVeBa.P

    Reference: "Learning the Pareto Set Under Incomplete Preferences: Pure Exploration in Vector Bandits,"
            Karagözlü, Yıldırım, Ararat, Tekin, AISTATS, '24
            https://proceedings.mlr.press/v238/karagozlu24a.html
    """

    def __init__(
        self, epsilon, delta,
        dataset_name, order: Order,
        noise_var,
        conf_contraction=32,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.noise_var = noise_var
        self.conf_contraction = conf_contraction

        dataset = get_dataset(dataset_name)

        self.m = dataset.out_dim

        # Trick to keep indices alongside points. This is for predictions from the model.
        in_data = np.hstack((dataset.in_data, np.arange(len(dataset.in_data))[:, None]))
        self.design_space = FixedPointsDesignSpace(
            in_data, dataset.out_dim, confidence_type='hyperellipsoid'
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = EmpiricalMeanVarModel(
            dataset.in_dim, self.m, noise_var, self.design_space.cardinality, track_variances=False
        )

        self.cone_alpha = self.order.ordering_cone.alpha.flatten()
        self.cone_alpha_eps = self.cone_alpha * self.epsilon

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.U = set()
        self.round = 0
        self.sample_count = 0

    def modeling(self):
        """
        Construct the confidence regions of all active designs given all past observations.
        """
        # All active designs have the same radius. We provide it as scale parameter.
        # Model does not track variances, so scale*var = scale.
        self.r_t = self.compute_radius()
        A = self.S.union(self.U)
        self.design_space.update(self.model, self.r_t, list(A))

    def discarding(self):
        """
        Discard the designs that are highly likely to be suboptimal using the confidence regions.
        """
        A = self.S.union(self.U)

        to_be_discarded = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in A:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_dominated(self.order, pt_conf, pt_p_conf, 0):
                    to_be_discarded.append(pt)
                    break
        
        for pt in to_be_discarded:
            self.S.remove(pt)

    def pareto_updating(self):
        """
        Identify the designs that are highly likely to be `epsilon`-optimal using the confidence regions.
        """
        A = self.S.union(self.U)

        new_pareto_pts = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in A:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(
                    self.order, pt_conf, pt_p_conf, self.cone_alpha_eps
                ):
                    break
            else:
                new_pareto_pts.append(pt)

        for pt in new_pareto_pts:
            self.S.remove(pt)
            self.P.add(pt)
        logging.debug(f"Pareto: {str(self.P)}")

    def useful_updating(self):
        """
        Identify the useful designs.
        """
        self.U = set()
        for pt in self.P:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in self.S:
                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(
                    self.order, pt_p_conf, pt_conf, self.cone_alpha_eps
                ):
                    self.U.add(pt)
                    break
        logging.debug(f"Useful: {str(self.U)}")

    def evaluating(self):
        """
        Observe the active designs via sampling.
        """
        A = self.S.union(self.U)
        active_pts = self.design_space.points[list(A)]

        observations = self.problem.evaluate(active_pts[:, :-1])

        self.sample_count += len(A)
        self.model.add_sample(A, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        """
        Run one step of the algorithm.

        Returns True if the algorithm is over.
        """
        self.round += 1
        print(f"Round {self.round}")

        print(f"Round {self.round}:Evaluating")
        self.evaluating()

        print(f"Round {self.round}:Modeling")
        self.modeling()

        print(f"Round {self.round}:Discarding")
        self.discarding()

        print(f"Round {self.round}:Pareto update")
        self.pareto_updating()

        print(f"Round {self.round}:Useful update")
        self.useful_updating()

        print(
            f"There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        print(f"Round {self.round}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_radius(self):
        """
        Compute the radii of the confidence regions to be used in modeling.
        """
        t1 = (8 * self.noise_var / self.round)
        t2 = np.log(  # ni**2 is equal to t**2 since only active arms are sampled
            (np.pi**2 * (self.m + 1) * self.design_space.cardinality * self.round**2)
            / (6 * self.delta)
        )
        r = np.sqrt(t1 * t2)

        # TODO: Do we need to scale because of norm-subgaussianity?
        return (r / self.conf_contraction) * np.ones(self.m, )
