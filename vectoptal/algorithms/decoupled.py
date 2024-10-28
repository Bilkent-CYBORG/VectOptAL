import logging
from typing import Optional

import numpy as np

from vectoptal.order import Order
from vectoptal.datasets import get_dataset_instance
from vectoptal.algorithms.algorithm import Algorithm
from vectoptal.maximization_problem import ProblemFromDataset, DecoupledEvaluationProblem
from vectoptal.acquisition import (
    ThompsonEntropyDecoupledAcquisition,
    optimize_decoupled_acqf_discrete,
)
from vectoptal.models import GPyTorchModelListExactModel, get_gpytorch_modellist_w_known_hyperparams


class DecoupledGP(Algorithm):
    def __init__(
        self,
        dataset_name,
        order: Order,
        noise_var,
        costs: Optional[list],
        cost_budget: Optional[float],
        batch_size=1,
    ) -> None:
        super().__init__()

        self.order = order
        self.batch_size = batch_size
        self.costs = np.array(costs)
        self.cost_budget = cost_budget

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim

        self.points = dataset.in_data
        self.problem = DecoupledEvaluationProblem(ProblemFromDataset(dataset, noise_var))

        self.model: GPyTorchModelListExactModel = get_gpytorch_modellist_w_known_hyperparams(
            self.problem, noise_var, initial_sample_cnt=1, X=dataset.in_data, Y=dataset.out_data
        )

        self.P = set()
        self.round = 0
        # TODO: Initial samples are managed in model preparation, they're not taken into account.
        self.sample_count = 0
        self.total_cost = 0.0

    def pareto_updating(self):
        m, v = self.model.predict(self.points)
        self.P = self.order.get_pareto_set(m)
        logging.debug(f"Pareto: {str(self.P)}")

    def evaluating(self):
        acq = ThompsonEntropyDecoupledAcquisition(self.model, order=self.order, costs=self.costs)
        candidate_list, acq_values, eval_indices = optimize_decoupled_acqf_discrete(
            acq, self.batch_size, choices=self.points
        )

        observations = self.problem.evaluate(candidate_list, eval_indices)

        self.sample_count += len(candidate_list)
        if self.costs is not None:
            self.total_cost += np.sum(self.costs[eval_indices])
        self.model.add_sample(candidate_list, observations, eval_indices)
        self.model.update()

    def run_one_step(self) -> bool:
        if len(self.P) == 0:
            return True

        self.round += 1

        round_str = f"Round {self.round}"

        logging.info(f"{round_str}:Evaluating")
        self.evaluating()

        logging.info(f"{round_str}:Pareto update")
        self.pareto_updating()

        logging.info(f"{round_str}:There are {len(self.P)} designs in set P.")

        logging.info(
            f"{round_str}:" f"Sample count {self.sample_count}, Cost {self.total_cost:.2f}"
        )

        return self.total_cost >= self.cost_budget
