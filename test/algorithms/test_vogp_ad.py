import unittest
import numpy as np
from vectoptal.utils.seed import SEED
from vectoptal.order import ConeTheta2DOrder
from vectoptal.algorithms.vogp_ad import VOGP_AD
from vectoptal.utils import set_seed
from vectoptal.maximization_problem import ContinuousProblem, get_continuous_problem
from vectoptal.utils.evaluate import (
    calculate_hypervolume_discrepancy_for_model
)


class TestVOGP_AD(unittest.TestCase):
    def setUp(self):
        """This method is run before each test to initialize
        parameters for the VOGP instance and prepare for
        testing both individual phases and full runs."""
        # Parameters for VOGP instance
        self.epsilon = 0.01
        self.delta = 0.05
        self.noise_var = self.epsilon
        self.problem_name = "BraninCurrin"
        self.problem: ContinuousProblem = get_continuous_problem(name=self.problem_name, noise_var=self.noise_var)
        self.order = ConeTheta2DOrder(cone_degree=90)
        self.iter_count = 1
        self.hv_values = []
        # Set random seed for reproducibility
        set_seed(SEED)

        # Create the VOGP instance
        self.algorithm = VOGP_AD(
            epsilon=self.epsilon,
            delta=self.delta,
            problem=self.problem,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=32,
        )
        self.beta = self.algorithm.compute_beta()

        print("setUp: Parameters and VOGP instance initialized.")

    def tearDown(self):
        """
        This method is run after each test
        to clean up resources or reset states.
        """
        self.algorithm = None
        print("tearDown: VOGP instance destroyed.")

    def test_ustar(self):
        """Test the calculation of u_star. Ensures that the computed u*
        is correctly calculated to be inside the cone."""
        in_cone = np.all(
            self.algorithm.order.ordering_cone.W @ self.algorithm.u_star >= 0
        )
        msg = "u* is not inside the cone. Check u* calculation."
        self.assertTrue(in_cone, msg=msg)

    def test_discarding(self):
        """Test the discarding phase of VOGP_AD by ensuring
        that the design count isn't increased."""
        self.algorithm.beta = self.algorithm.compute_beta()
        self.algorithm.modeling()  # Ensure model is initialized
        initial_S_size = len(self.algorithm.S)
        self.algorithm.discarding()
        self.assertLessEqual(
            len(self.algorithm.S),
            initial_S_size,
            "S should shrink or remain the same after discarding.",
        )
        print("test_discarding passed.")

    def test_epsilon_covering(self):
        """Test the epsilon covering phase of VOGP_AD by ensuring
        that the design count isn't increased."""
        self.algorithm.beta = self.algorithm.compute_beta()
        self.algorithm.modeling()
        initial_S_size = len(self.algorithm.S)
        self.algorithm.epsiloncovering()
        self.assertLessEqual(
            len(self.algorithm.S),
            initial_S_size,
            "S should shrink or remain the same after epsilon covering.",
        )
        print("test_epsilon_covering passed.")

    def test_vogp_ad_run(self):
        """This test performs a full run of the VOGP_AD algorithm
        on the BraninCurrin dataset and calculates the
        hypervolume score over multiple iterations."""
        for iter_i in range(self.iter_count):
            set_seed(SEED + iter_i + 1)

            algorithm = VOGP_AD(
                epsilon=self.epsilon,
                delta=self.delta,
                problem=self.problem,
                order=self.order,
                noise_var=self.noise_var,
                conf_contraction=32,
            )

            while True:

                initial_S_size = len(self.algorithm.S)
                initial_P_size = len(self.algorithm.P)
                is_done = algorithm.run_one_step()
                self.assertLessEqual(
                    len(self.algorithm.S),
                    initial_S_size,
                    msg="S should shrink or remain the same.",
                )
                self.assertGreaterEqual(
                    len(self.algorithm.P),
                    initial_P_size,
                    msg="P should get bigger or remain the same.",
                )

                ready_for_covering = True
                for design_i in self.algorithm.S:
                    ready_for_covering = (
                        ready_for_covering
                        and self.algorithm.design_space.point_depths[design_i]
                        != self.algorithm.max_discretization_depth
                    )
                if not ready_for_covering:
                    self.assertEqual(self.algorithm.P, {})
                if is_done:
                    break

            log_hv_discrepancy = calculate_hypervolume_discrepancy_for_model(
                self.order, self.problem, algorithm.model
            )
            print(f"Logarithm HV discrepancy is: {log_hv_discrepancy:.2f}")
            self.hv_values.append(log_hv_discrepancy)
        avg_hv = sum(self.hv_values) / len(self.hv_values)
        print(
            f"Avg. hypervolume score: {avg_hv:.2f}"
        )
        self.assertLessEqual(
            avg_hv, -0.5, "Avg. hypervolume score should be reasonably high."
        )


if __name__ == "__main__":
    unittest.main()
