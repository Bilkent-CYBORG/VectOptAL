import logging
from typing import Optional

import numpy as np

from vectoptal.order import Order, ConeTheta2DOrder
from vectoptal.datasets import get_dataset
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset


class NaiveElimination(PALAlgorithm):
    def __init__(
        self, epsilon, delta,
        dataset_name, order: Order,
        noise_var,
        L: Optional[int]=None
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order

        self.dataset = get_dataset(dataset_name)
        self.m = self.dataset.out_dim

        self.K = len(self.dataset.in_data)
        if L is None:  # Use theoretical sampling count if not given.
            assert hasattr(order.ordering_cone, "beta"), "Ordering complexity needs to be defined."
            ordering_complexity = order.ordering_cone.beta

            c = 1 + np.sqrt(2)  # Any c>0 should suffice according to Lemma B.12.
            self.L = np.ceil(
                4
                * ((c*noise_var*ordering_complexity/self.epsilon)**2)
                * np.log(4*self.m /(2*self.delta/(self.K*(self.K-1))))
            ).astype(int)
        else:
            self.L = L

        self.problem = ProblemFromDataset(self.dataset, noise_var)

        self.samples = np.empty((self.K, 0, self.m))

        self.round = 0
        self.sample_count = 0

    def run_one_step(self) -> bool:
        self.round += 1

        new_samples = self.problem.evaluate(self.dataset.in_data)
        self.samples = np.concatenate(
            [self.samples, np.expand_dims(new_samples, axis=-2)],
            axis=-2
        )

        self.sample_count += self.K

        if self.round % 100 == 0:
            print(f"Round {self.round}")
            print(f"Round {self.round}:Sample count {self.sample_count}")

        return self.round == self.L

    @property
    def P(self):
        return self.order.get_pareto_set(self.samples.mean(axis=-2))
