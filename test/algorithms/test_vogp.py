import unittest
import numpy as np
from vectoptal.utils.seed import SEED
from vectoptal.order import ConeTheta2DOrder
from vectoptal.algorithms.vogp import VOGP
from vectoptal.utils import set_seed
from vectoptal.utils.evaluate import calculate_epsilonF1_score
from vectoptal.design_space import FixedPointsDesignSpace


class TestVOGP(unittest.TestCase):

    def setUp(self):
        """
        This method is run before each test to initialize parameters
        for the VOGP instance and prepare for testing both individual
        phases and full runs.
        """
        # Parameters for VOGP instance
        self.epsilon = 0.01
        self.delta = 0.05
        self.noise_var = self.epsilon
        self.dataset_name = "DiskBrake"
        self.order = ConeTheta2DOrder(cone_degree=90)
        self.iter_count = 10
        self.eps_f1_values = []
        # Set random seed for reproducibility
        set_seed(SEED)

        # Create the VOGP instance
        self.algorithm = VOGP(
            epsilon=self.epsilon,
            delta=self.delta,
            dataset_name=self.dataset_name,
            order=self.order,
            noise_var=self.noise_var,
            conf_contraction=32,
        )
        self.beta = self.algorithm.compute_beta()

    def tearDown(self):
        """
        This method is run after each test to
        clean up resources or reset states.
        """
        self.algorithm = None

    def test_ustar(self):
        """
        Test the calculation of u_star
        Ensures that the computed u* is correctly
        calculated to be inside the cone.
        """
        in_cone = np.all(self.algorithm.order.ordering_cone.W @ self.algorithm.u_star >= 0)
        msg = "u* is not inside the cone. Check u* calculation."
        self.assertTrue(in_cone, msg=msg)

    def test_discarding(self):
        """
        Main test function that iteratively calls the discarding_subtest.
        """
        for _ in range(self.iter_count):
            self.test_discarding_subtest()

    def test_discarding_subtest(self):
        """
        Tests the discarding phase of VOGP. Compares randomly generated
        confidence regions where one is supposed to be discarded by
        the other.
        """

        self.manual_in_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.manual_out_data = np.array([[1.0, 2.0], [1.5, 2.5]])
        # Redefine dataset
        self.algorithm.problem.dataset.in_data = self.manual_in_data
        self.algorithm.problem.dataset.out_data = self.manual_out_data

        # Manually update the confidence regions with means and covariances
        mean_array = np.array([[1.0, 1.0], [1.0, 1.0]])
        random_cov_diagonal_values = np.random.rand(2)
        cov_array = np.array(
            [
                np.diag(
                    [
                        random_cov_diagonal_values[i],
                        1 - random_cov_diagonal_values[i],
                    ]
                )
                for i in range(2)
            ]
        )
        shifted_mu = mean_array[0] + self.algorithm.u_star * (2 * self.algorithm.d1 - self.epsilon)
        mean_array[1] = shifted_mu

        self.algorithm.design_space = FixedPointsDesignSpace(
            self.manual_in_data,
            self.manual_out_data,
            confidence_type="hyperrectangle",
        )

        # Manually setting the confidence regions:
        for pt_i in range(self.algorithm.design_space.cardinality):
            iterative_inter = self.algorithm.design_space.confidence_regions[
                pt_i
            ].intersect_iteratively
            self.algorithm.design_space.confidence_regions[pt_i].intersect_iteratively = False
            self.algorithm.design_space.confidence_regions[pt_i].update(
                mean=mean_array[pt_i],
                covariance=cov_array[pt_i],
                scale=np.ones(2),
            )
            self.algorithm.design_space.confidence_regions[pt_i].intersect_iteratively = (
                iterative_inter
            )

        self.algorithm.S = set(range(self.algorithm.design_space.cardinality))
        self.algorithm.P = set()
        initial_S_size = len(self.algorithm.S)
        self.algorithm.discarding()

        self.assertLess(
            len(self.algorithm.S),
            initial_S_size,
            "A design to be discarded was not discarded.",
        )
        self.assertEqual(self.algorithm.S, {1}, "Wrong design was discarded.")

    def test_epsilon_covering(self):
        """
        Main test function that iteratively calls the epsilon_covering_subtest.
        """
        for _ in range(self.iter_count):
            self.test_epsilon_covering_subtest()

    def test_epsilon_covering_subtest(self):
        """
        Tests the identification phase of VOGP. Compares randomly generated
        confidence regions where one design prevents the other design from
        being added to the Pareto set but not vice versa.
        """
        self.manual_in_data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.manual_out_data = np.array([[1.0, 2.0], [1.5, 2.5]])
        # Redefine dataset
        self.algorithm.problem.dataset.in_data = self.manual_in_data
        self.algorithm.problem.dataset.out_data = self.manual_out_data
        # Manually update the confidence regions
        # with placeholder means and covariances
        mean_array = np.array([[1.0, 1.0], [1.0, 1.0]])
        random_cov_diagonal_values = np.random.rand(2)
        cov_array = np.array(
            [
                np.diag(
                    [
                        random_cov_diagonal_values[i],
                        1 - random_cov_diagonal_values[i],
                    ]
                )
                for i in range(2)
            ]
        )
        shifted_mu = mean_array[0] + self.algorithm.u_star * (2 * self.algorithm.d1 + self.epsilon)
        mean_array[1] = shifted_mu
        self.algorithm.design_space = FixedPointsDesignSpace(
            self.manual_in_data,
            self.manual_out_data,
            confidence_type="hyperrectangle",
        )

        # Manually setting the confidence regions:
        for pt_i in range(self.algorithm.design_space.cardinality):
            iterative_inter = self.algorithm.design_space.confidence_regions[
                pt_i
            ].intersect_iteratively
            self.algorithm.design_space.confidence_regions[pt_i].intersect_iteratively = False
            self.algorithm.design_space.confidence_regions[pt_i].update(
                mean=mean_array[pt_i],
                covariance=cov_array[pt_i],
                scale=np.ones(2),
            )
            self.algorithm.design_space.confidence_regions[pt_i].intersect_iteratively = (
                iterative_inter
            )

        self.algorithm.S = set(range(self.algorithm.design_space.cardinality))
        self.algorithm.P = set()
        initial_S_size = len(self.algorithm.S)
        self.algorithm.epsiloncovering()
        self.assertLess(
            len(self.algorithm.S),
            initial_S_size,
            "Covering rule did not work.",
        )
        self.assertEqual(self.algorithm.P, {1}, "Wrong design was added to Pareto.")

    def test_evaluating(self):
        """
        Test the evaluating function of VOGP. Checks if the sample
        count is increased after the evaluation phase.
        """
        self.algorithm.modeling()
        initial_sample_count = self.algorithm.sample_count
        self.algorithm.evaluating()
        self.assertGreater(
            self.algorithm.sample_count,
            initial_sample_count,
            "Sample count should increase after evaluation.",
        )

    def test_vogp_run(self):
        """
        This test performs a full run of
        the VOGP algorithm on the DiskBrake
        dataset and calculates the epsilon-F1
        score over multiple iterations.
        """
        for iter_i in range(self.iter_count):
            set_seed(SEED + iter_i + 1)

            # Re-initialize VOGP instance for each iteration
            algorithm = VOGP(
                epsilon=self.epsilon,
                delta=self.delta,
                dataset_name=self.dataset_name,
                order=self.order,
                noise_var=self.noise_var,
                conf_contraction=32,
            )

            # Run VOGP until completion
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
                if is_done:
                    break

            # Get Pareto indices and calculate epsilon-F1 score
            pareto_indices = self.order.get_pareto_set(self.algorithm.problem.dataset.out_data)
            eps_f1 = calculate_epsilonF1_score(
                self.algorithm.problem.dataset,
                self.order,
                pareto_indices,
                list(algorithm.P),
                self.epsilon,
            )
            self.eps_f1_values.append(eps_f1)

        # Check the average epsilon-F1 score
        avg_eps_f1 = sum(self.eps_f1_values) / len(self.eps_f1_values)
        self.assertGreaterEqual(avg_eps_f1, 0.5, "Avg. eps-F1 score should be reasonably high.")


if __name__ == "__main__":
    unittest.main()
