from unittest import mock, TestCase

import numpy as np

from vectoptal.algorithms import PaVeBa
from vectoptal.order import ComponentwiseOrder
from vectoptal.datasets import Dataset


class TestPaVeBa(TestCase):
    """Test the PaVeBa class."""

    def setUp(self):
        # A basic setup for the model.
        self.epsilon = 0.1
        self.delta = 0.1
        self.dataset_name = "DiskBrake"
        self.order = ComponentwiseOrder(2)
        self.noise_var = 0
        self.conf_contraction = 1
        self.algo = PaVeBa(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
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
        dataset = globals()[self.dataset_name]()
        eps_f1 = calculate_epsilonF1_score(dataset, self.order, self.order.get_pareto_set(dataset.out_data), list(pareto_indices), self.epsilon)
        self.assertTrue(eps_f1 > 0.9)

    def test_run_one_step(self):
        """Test the run_one_step method."""
        for i in range(42):
            self.algo.run_one_step()
            if i == 3:
                S3 = algo.S
                P3 = algo.P
        self.assertEqual(42, algo.round)
        S = algo.S
        P = algo.P
        self.assertTrue(len(S3) >= len(S))
        self.assertTrue(len(P) >= len(P3))

    def test_compute_radius(self):
        """Test the compute_radius method."""
        self.algo.run_one_step()
        t1 = 8 * self.noise_var
        t2 = np.log(
            (np.pi**2 * (3) * 128)
            / (6 * 0.1)
        )
        r1 = np.sqrt(t1 * t2)
        r2 = algo.compute_radius()
        self.assertEqual(np.array([r1, r1]), r2)

        self.algo.run_one_step()
        r3 = algo.compute_radius()
        self.assertTrue((r3 < r2).all())