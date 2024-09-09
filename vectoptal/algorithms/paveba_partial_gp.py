import copy
import logging
from typing import Literal, Optional

import numpy as np

from vectoptal.order import Order
from vectoptal.datasets import get_dataset
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset, DecoupledEvaluationProblem
from vectoptal.acquisition import (
    MaxVarianceDecoupledAcquisition,
    optimize_decoupled_acqf_discrete
)
from vectoptal.confidence_region import (
    confidence_region_is_dominated,
    confidence_region_is_covered
)
from vectoptal.models import (
    GPyTorchModelListExactModel,
    get_gpytorch_modellist_w_known_hyperparams
)


class PaVeBaPartialGP(PALAlgorithm):
    def __init__(
        self, epsilon, delta,
        dataset_name, order: Order,
        noise_var,
        conf_contraction=32,
        costs: Optional[list] = None,
        cost_budget: Optional[float] = None,
        confidence_type: Literal["hyperrectangle", "hyperellipsoid"]="hyperrectangle",
        batch_size=1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction
        self.costs = np.array(costs) if costs is not None else costs
        self.cost_budget = cost_budget if cost_budget is not None else np.inf

        dataset = get_dataset(dataset_name)

        self.m = dataset.out_dim

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type=confidence_type
        )
        self.problem = DecoupledEvaluationProblem(ProblemFromDataset(dataset, noise_var))

        self.model: GPyTorchModelListExactModel = get_gpytorch_modellist_w_known_hyperparams(
            self.problem, noise_var, initial_sample_cnt=1,
            X=dataset.in_data, Y=dataset.out_data
        )

        self.cone_alpha = self.order.ordering_cone.alpha.flatten()
        self.cone_alpha_eps = self.cone_alpha * self.epsilon

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.U = set()
        self.round = 0
        # TODO: Initial samples are managed in model preparation, they're not taken into account.
        self.sample_count = 0
        self.total_cost = 0.0

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
        acq = MaxVarianceDecoupledAcquisition(self.model, costs=self.costs)
        active_pts = self.design_space.points[list(A)]
        candidate_list, acq_values, eval_indices = optimize_decoupled_acqf_discrete(
            acq, self.batch_size, choices=active_pts
        )

        print("Points:", candidate_list)
        print("Acq. values:", acq_values)
        print("Obj. indices:", eval_indices)

        observations = self.problem.evaluate(candidate_list, eval_indices)

        self.sample_count += len(candidate_list)
        if self.costs is not None:
            self.total_cost += np.sum(self.costs[eval_indices])
        self.model.add_sample(candidate_list, observations, eval_indices)
        self.model.update()

    def run_one_step(self) -> bool:
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

        return len(self.S) == 0 or self.total_cost >= self.cost_budget

    def compute_alpha(self):
        alpha = (
            2*np.log(
                (np.pi**2 * self.round**2 * self.design_space.cardinality)/(3*self.delta)
            )
        )

        return (alpha / self.conf_contraction) * np.ones(self.m, )
