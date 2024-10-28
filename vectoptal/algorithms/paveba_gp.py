import copy
import logging
from typing import Literal

import numpy as np

from vectoptal.order import Order
from vectoptal.datasets import get_dataset_instance
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset
from vectoptal.acquisition import SumVarianceAcquisition, optimize_acqf_discrete
from vectoptal.confidence_region import confidence_region_is_dominated, confidence_region_is_covered
from vectoptal.models import (
    CorrelatedExactGPyTorchModel,
    IndependentExactGPyTorchModel,
    get_gpytorch_model_w_known_hyperparams,
)


class PaVeBaGP(PALAlgorithm):
    def __init__(
        self,
        epsilon,
        delta,
        dataset_name,
        order: Order,
        noise_var,
        conf_contraction=32,
        type: Literal["IH", "DE"] = "IH",
        batch_size=1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim

        if type == "IH":
            design_confidence_type = "hyperrectangle"
            model_class = IndependentExactGPyTorchModel
        elif type == "DE":
            design_confidence_type = "hyperellipsoid"
            model_class = CorrelatedExactGPyTorchModel

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type=design_confidence_type
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = get_gpytorch_model_w_known_hyperparams(
            model_class,
            self.problem,
            noise_var,
            initial_sample_cnt=1,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.cone_alpha = self.order.ordering_cone.alpha.flatten()
        self.cone_alpha_eps = self.cone_alpha * self.epsilon

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.U = set()
        self.round = 0
        self.sample_count = 0

    def modeling(self):
        self.alpha_t = self.compute_alpha()
        A = self.S.union(self.U)
        self.design_space.update(self.model, self.alpha_t, list(A))

    def discarding(self):
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
        A = self.S.union(self.U)
        acq = SumVarianceAcquisition(self.model)
        active_pts = self.design_space.points[list(A)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        observations = self.problem.evaluate(candidate_list)

        self.sample_count += len(candidate_list)
        self.model.add_sample(candidate_list, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        if len(self.S) == 0:
            return True

        self.round += 1

        round_str = f"Round {self.round}"

        logging.info(f"{round_str}:Evaluating")
        self.evaluating()

        logging.info(f"{round_str}:Modeling")
        self.modeling()

        logging.info(f"{round_str}:Discarding")
        self.discarding()

        logging.info(f"{round_str}:Pareto update")
        self.pareto_updating()

        logging.info(f"{round_str}:Useful update")
        self.useful_updating()

        logging.info(
            f"{round_str}:There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        logging.info(f"{round_str}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_alpha(self):
        alpha = 8 * self.m * np.log(6) + 4 * np.log(
            (np.pi**2 * self.round**2 * self.design_space.cardinality) / (6 * self.delta)
        )

        return (alpha / self.conf_contraction) * np.ones(
            self.m,
        )
