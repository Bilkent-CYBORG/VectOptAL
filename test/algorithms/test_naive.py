from unittest import TestCase

import numpy as np

from vectoptal.utils.seed import SEED
from vectoptal.order import ConeTheta2DOrder
from vectoptal.algorithms import NaiveElimination
from vectoptal.datasets import get_dataset_instance
from vectoptal.utils.evaluate import calculate_epsilonF1_score


class TestNaiveElimination(TestCase):
    """Test the NaiveElimination class."""

    def setUp(self):
        """A basic setup for the model."""
        np.random.seed(SEED)

        self.epsilon = 0.1
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ConeTheta2DOrder(cone_degree=90)
        self.noise_var = 0.00001
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.algo = NaiveElimination(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
        )

    def test_whole_class(self):
        """Test the whole class by running it until the end end checking its score."""
        while True:
            is_done = self.algo.run_one_step()
            if is_done:
                break

        self.assertTrue(self.algo.run_one_step())

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

    def test_run_one_step(self):
        """Test the run_one_step method."""
        num_rounds = 10
        alg_done = False
        for i in range(num_rounds):  # Run for 10 rounds, it should be enough.
            if not alg_done and i <= 3:  # Save the state at round 3 at the latest.
                sample_test = self.algo.sample_count
            alg_done = self.algo.run_one_step()

        sample = self.algo.sample_count

        self.assertTrue(num_rounds >= self.algo.round)
        self.assertTrue(sample_test < sample)
