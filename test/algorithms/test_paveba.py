from unittest import TestCase

import numpy as np

from vectoptal.utils.seed import SEED
from vectoptal.algorithms import PaVeBa
from vectoptal.order import ComponentwiseOrder
from vectoptal.datasets import get_dataset_instance
from vectoptal.utils.evaluate import calculate_epsilonF1_score


class TestPaVeBa(TestCase):
    """Test the PaVeBa class."""

    def setUp(self):
        """A basic setup for the model."""
        np.random.seed(SEED)

        self.epsilon = 0.1
        self.delta = 0.1
        self.dataset_name = "Test"
        self.order = ComponentwiseOrder(2)
        self.noise_var = 0.00001
        self.dataset_cardinality = get_dataset_instance(self.dataset_name)._cardinality
        self.conf_contraction = 1
        self.algo = PaVeBa(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
        )

    def test_evaluating(self):
        """Test the evaluating method."""
        sample_test = self.algo.sample_count
        self.algo.evaluating()
        self.assertTrue(self.algo.sample_count > sample_test)

    def test_whole_class(self):
        """Test the whole class by running it until the end end checking its score."""
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

    def test_run_one_step(self):
        """Test the run_one_step method."""
        num_rounds = 10
        alg_done = False
        for i in range(num_rounds):  # Run for 10 rounds, it should be enough.
            if not alg_done and i <= 3:  # Save the state at round 3 at the latest.
                S_test = self.algo.S
                P_test = self.algo.P
                sample_test = self.algo.sample_count
            alg_done = self.algo.run_one_step()

        S = self.algo.S
        P = self.algo.P
        sample = self.algo.sample_count

        self.assertTrue(num_rounds >= self.algo.round)
        self.assertTrue(len(S_test) >= len(S))
        self.assertTrue(len(P) >= len(P_test))
        self.assertTrue(sample_test < sample)

    def test_compute_radius(self):
        """Test the compute_radius method."""
        self.algo.run_one_step()
        t1 = 8 * self.noise_var
        t2 = np.log((np.pi**2 * (3) * self.dataset_cardinality) / (6 * 0.1))
        r1 = np.sqrt(t1 * t2)
        r2 = self.algo.compute_radius()
        self.assertTrue((np.array([r1, r1]) == r2).all())

        self.algo.run_one_step()
        r3 = self.algo.compute_radius()
        self.assertTrue((r3 <= r2).all())
