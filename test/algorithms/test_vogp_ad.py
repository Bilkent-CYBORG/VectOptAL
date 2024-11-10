import unittest
import numpy as np
from vectoptal.utils.seed import SEED
from vectoptal.order import ConeTheta2DOrder
from vectoptal.algorithms.vogp_ad import VOGP_AD
from vectoptal.utils import set_seed
from vectoptal.maximization_problem import ContinuousProblem, get_continuous_problem
from vectoptal.utils.evaluate import calculate_hypervolume_discrepancy_for_model


class TestVOGP_AD(unittest.TestCase):
    """Test the VOGP_AD algorithm class."""

    def setUp(self):
        """
        Basic setup for the algorithm.
        """
        # Set random seed for reproducibility
        set_seed(SEED)

        # Parameters for VOGP instance
        self.epsilon = 0.1
        self.delta = 0.1
        self.noise_var = self.epsilon
        self.problem_name = "BraninCurrin"
        self.problem: ContinuousProblem = get_continuous_problem(self.problem_name, self.noise_var)
        self.order = ConeTheta2DOrder(cone_degree=90)
        self.iter_count = 1
        self.conf_contraction = 32
        self.hv_values = []

        # Create the VOGP instance
        self.algorithm = VOGP_AD(
            epsilon=self.epsilon,
            delta=self.delta,
            problem=self.problem,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=self.conf_contraction,
        )

    def test_ustar(self):
        """
        Test the calculation of u_star. Ensures that the computed u* is inside the cone and has
        a norm of 1.
        """
        self.assertAlmostEqual(np.linalg.norm(self.algorithm.u_star), 1.0)
        self.assertTrue(self.order.ordering_cone.is_inside(self.algorithm.u_star))

    def test_evaluating(self):
        """
        Test the evaluating function.
        """
        self.algorithm.run_one_step()
        initial_sample_count = self.algorithm.sample_count
        initial_S_size = len(self.algorithm.S)
        self.algorithm.evaluate_refine()
        self.assertTrue(
            (self.algorithm.sample_count > initial_sample_count)
            or (len(self.algorithm.S) > initial_S_size)
        )

    def test_vogp_ad_run(self):
        """
        This test performs a full run of the VOGP_AD algorithm and calculates the log. hypervolume
        discrepancy from the resulting predictive model.
        """

        while True:
            is_done = self.algorithm.run_one_step()
            if is_done:
                break

        log_hv_discrepancy = calculate_hypervolume_discrepancy_for_model(
            self.order, self.problem, self.algorithm.model
        )

        self.assertLessEqual(
            log_hv_discrepancy, -3.5, "Log. hypervolume discrepancy should be reasonably low."
        )
