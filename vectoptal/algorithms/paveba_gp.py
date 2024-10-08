import copy
import logging
from typing import Literal

import numpy as np

from vectoptal.order import Order
from vectoptal.datasets import get_dataset
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset
from vectoptal.acquisition import SumVarianceAcquisition, optimize_acqf_discrete
from vectoptal.confidence_region import (
    confidence_region_is_dominated,
    confidence_region_is_covered
)
from vectoptal.models import (
    CorrelatedExactGPyTorchModel,
    IndependentExactGPyTorchModel,
    get_gpytorch_model_w_known_hyperparams
)


class PaVeBaGP(PALAlgorithm):
    """
    Implements the GP-based Pareto Vector Bandits (PaVeBa) algorithm.

    :param float epsilon: Determines the accuracy of the PAC-learning framework.
    :param float delta: Determines the success probability of the PAC-learning framework.
    :param str dataset_name: Name of the dataset to be used.
    :param order: Order to be used.
    :param float noise_var: Variance of the Gaussian sampling noise.
    :param float conf_contraction: Contraction coefficient to shrink the confidence regions empirically.
    :param str type: Specifies if the algorithm uses dependent hyperellipsoidal or independent hyperrectangular confidence regions.
    :param int batch_size: Number of samples taken in each round.

    The algorithm sequentially samples design rewards with a multivariate white Gaussian noise whose diagonal
    entries are specified by the user.

    Returns None.

    Example:
        >>> from vectoptal.order import ConeTheta2DOrder
        >>> from vectoptal.algorithms import PaVeBaGP
        >>>
        >>> epsilon, delta, noise_var = 0.01, 0.01, 0.01
        >>> dataset_name = "DiskBrake"
        >>> order_acute = ConeTheta2DOrder(cone_degree = 45)
        >>>
        >>> PaVeBaGP = PaVeBaGP(epsilon, delta, dataset_name, order_acute, noise_var)
        >>>
        >>> while True:
        >>>     is_done = PaVeBaGP.run_one_step()
        >>>
        >>>     if is_done:
        >>>          break
        >>>
        >>> pareto_indices = PaVeBaGP.P

    Reference: "Learning the Pareto Set Under Incomplete Preferences: Pure Exploration in Vector Bandits,"
            Karagözlü, Yıldırım, Ararat, Tekin, AISTATS, '24
            https://proceedings.mlr.press/v238/karagozlu24a.html
    """

    def __init__(
        self, epsilon, delta,
        dataset_name, order: Order,
        noise_var,
        conf_contraction=32,
        type: Literal["IH", "DE"]="IH",
        batch_size=1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction

        dataset = get_dataset(dataset_name)

        self.m = dataset.out_dim

        if type == "IH":
            design_confidence_type = 'hyperrectangle'
            model_class = IndependentExactGPyTorchModel
        elif type == "DE":
            design_confidence_type = 'hyperellipsoid'
            model_class = CorrelatedExactGPyTorchModel

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type=design_confidence_type
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = get_gpytorch_model_w_known_hyperparams(
            model_class, self.problem, noise_var, initial_sample_cnt=1,
            X=dataset.in_data, Y=dataset.out_data
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
        Constructs the confidence regions of all active designs given all past observations.
        """
        self.alpha_t = self.compute_alpha()
        A = self.S.union(self.U)
        self.design_space.update(self.model, self.alpha_t, list(A))

    def discarding(self):
        """
        Discards the designs that are highly likely to be suboptimal using the confidence regions.
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
        Identifies the designs that are highly likely to be `epsilon`-optimal using the confidence regions.
        """
        A = self.S.union(self.U)

        is_index_pareto = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in A:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(
                    self.order, pt_conf, pt_p_conf, self.cone_alpha_eps
                ):
                    is_index_pareto.append(False)
                    break
            else:
                is_index_pareto.append(True)

        tmp_S = copy.deepcopy(self.S)
        for is_pareto, pt in zip(is_index_pareto, tmp_S):
            if is_pareto:
                self.S.remove(pt)
                self.P.add(pt)
        logging.debug(f"Pareto: {str(self.P)}")

    def useful_updating(self):
        """
        Identifies the useful designs.
        """
        self.U = set()
        for pt in self.P:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in self.S:
                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(
                    self.order, pt_conf, pt_p_conf, self.cone_alpha_eps
                ):
                    self.U.add(pt)
                    break
        logging.debug(f"Useful: {str(self.U)}")

    def evaluating(self):
        """
        Observes the active designs via sampling.
        """
        A = self.S.union(self.U)
        acq = SumVarianceAcquisition(self.model)
        active_pts = self.design_space.points[list(A)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        observations = self.problem.evaluate(candidate_list)

        self.sample_count += len(candidate_list)
        self.model.add_sample(candidate_list, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        """
        Runs one step of the algorithm.

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

    def compute_alpha(self):
        """
        Computes the radii of the confidence regions to be used in modeling.
        """
        alpha = (
            8*self.m*np.log(6) + 4*np.log(
                (np.pi**2 * self.round**2 * self.design_space.cardinality)/(6*self.delta)
            )
        )

        return (alpha / self.conf_contraction) * np.ones(self.m, )
