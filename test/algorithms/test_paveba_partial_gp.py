from unittest import TestCase

import numpy as np

from vectoptal.utils.seed import SEED
from vectoptal.order import ComponentwiseOrder
from vectoptal.algorithms import PaVeBaPartialGP
from vectoptal.datasets import get_dataset_instance
from vectoptal.utils.evaluate import calculate_epsilonF1_score


class TestPaVeBaPartialGP(TestCase):
    """Test the PaVeBa class."""

    def setUp(self):
        # A basic setup for the model.
        np.random.seed(SEED)

        self.epsilon = 0.1
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.noise_var = 0.00001
        self.conf_contraction = 1
        self.costs = [1.0, 1.5]
        self.cost_budget = 64
        self.algo = PaVeBaPartialGP(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
            costs=self.costs,
            cost_budget=self.cost_budget,
        )

    def test_modeling(self):
        """Test the modeling method."""

    def test_discarding(self):
        """Test the discarding method."""

    def test_pareto_updating(self):
        """Test the pareto_updating method."""

    def test_useful_updating(self):
        """Test the useful_updating method."""

    def test_evaluating(self):
        """Test the evaluating method."""

    def test_whole_class(self):
        while True:
            is_done = self.algo.run_one_step()
            if is_done:
                break

        pareto_indices = self.algo.P
        dataset = get_dataset_instance(self.dataset_name)
        eps_f1 = calculate_epsilonF1_score(
            dataset,
            self.order,
            self.order.get_pareto_set(dataset.out_data),
            list(pareto_indices),
            self.epsilon,
        )
        self.assertTrue(eps_f1 > 0.9)
        self.assertLess(self.algo.total_cost, self.cost_budget + max(self.costs))
        self.assertLessEqual(self.algo.total_cost, self.algo.round * max(self.costs))
        self.assertGreaterEqual(self.algo.total_cost, self.algo.round * min(self.costs))

    def test_run_one_step(self):
        """Test the run_one_step method."""
        for i in range(42):
            self.algo.run_one_step()
            if i == 3:
                S3 = self.algo.S
                P3 = self.algo.P
        self.assertTrue(42 >= self.algo.round)
        S = self.algo.S
        P = self.algo.P
        self.assertTrue(len(S3) >= len(S))
        self.assertTrue(len(P) >= len(P3))

    def test_compute_alpha(self):
        """Test the compute_alpha method."""
        self.algo.run_one_step()
        alpha = 2 * np.log((np.pi**2 * self.dataset_cardinality) / (3 * self.delta))
        r1 = alpha
        r2 = self.algo.compute_alpha()
        self.assertTrue((np.array([r1, r1]) / self.conf_contraction == r2).all())

        self.algo.run_one_step()
        r3 = self.algo.compute_alpha()
        self.assertTrue((r3 > r2).all())
