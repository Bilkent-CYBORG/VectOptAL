import copy
import logging

import numpy as np
from scipy.optimize import minimize

from vectoptal.order import Order
from vectoptal.datasets import get_dataset_instance
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset
from vectoptal.acquisition import MaxDiagonalAcquisition, optimize_acqf_discrete
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.models import CorrelatedExactGPyTorchModel, get_gpytorch_model_w_known_hyperparams
from vectoptal.confidence_region import (
    confidence_region_is_dominated,
    confidence_region_check_dominates,
    confidence_region_is_covered,
)


class VOGP(PALAlgorithm):
    """
    Implement the Vector Optimization with Gaussian Process (VOGP) algorithm.

    :param epsilon: Accuracy parameter for the PAC algorithm
    :type epsilon: float
    :param delta: Confidence parameter for the PAC algorithm
    :type delta: float
    :param dataset_name: Name of the dataset to be used
    :type dataset_name: str
    :param order: An instance of the Order class representing the ordering cone
    :type order: Order
    :param noise_var: Variance of the noise in observations
    :type noise_var: float
    :param conf_contraction: Factor for contracting the confidence region, defaults to 32
    :type conf_contraction: int, optional
    :param batch_size: Batch size for evaluation, defaults to 1
    :type batch_size: int, optional

    Attributes include:

    - `m`: Dimension of the output space
    - `design_space`: The design space containing fixed points and confidence types (e.g., hyperrectangle)
    - `problem`: An instance of the `ProblemFromDataset` class, constructed from the specified dataset
    - `model`: A GP model to estimate outcomes
    - `S`: The set of uncertain designs at each iteration
    - `P`: The set of Pareto optimal designs identified by the algorithm
    - `round`: Tracks the current round of the algorithm
    - `sample_count`: Tracks the total count of samples evaluated across all rounds

    Example Usage:
        >>> from vectoptal.order import ComponentwiseOrder
        >>> from vectoptal.algorithms import VOGP
        >>>
        >>> vogp = VOGP(epsilon=0.1, delta=0.05, dataset_name='DiskBrake', order=ComponentwiseOrder(2), noise_var=0.01)
        >>> while not vogp.run_one_step():
        >>>     pass
        >>> pareto_set = vogp.P
    """

    def __init__(
        self,
        epsilon,
        delta,
        dataset_name,
        order: Order,
        noise_var,
        conf_contraction=32,
        batch_size=1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.batch_size = batch_size
        self.conf_contraction = conf_contraction

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type="hyperrectangle"
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = get_gpytorch_model_w_known_hyperparams(
            CorrelatedExactGPyTorchModel,
            self.problem,
            noise_var,
            initial_sample_cnt=1,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.u_star, self.d1 = self.compute_u_star()
        self.u_star_eps = self.u_star * epsilon

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.round = 0
        self.sample_count = 0

    def modeling(self):
        """
        Updates confidence regions for all active nodes.
        """
        self.beta = self.compute_beta()
        # Active nodes, union of sets s_t and p_t at the beginning of round t
        W = self.S.union(self.P)
        self.design_space.update(self.model, self.beta, list(W))

    def discarding(self):
        """
        Discards designs that are highly likely to be dominated based on
        current confidence regions.
        """
        pessimistic_set = self.compute_pessimistic_set()
        difference = self.S.difference(pessimistic_set)

        to_be_discarded = []
        for pt in difference:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in pessimistic_set:
                pt_p_conf = self.design_space.confidence_regions[pt_prime]
                # Function to check if ∃z' in R(x') such that R(x) <_C z + u, where u < epsilon
                if confidence_region_is_dominated(self.order, pt_conf, pt_p_conf, self.u_star_eps):
                    to_be_discarded.append(pt)
                    break

        for pt in to_be_discarded:
            self.S.remove(pt)

    def epsiloncovering(self):
        """
        Identifies and removes designs from `S` that are not covered by the confidence region of other designs,
        adding them to `P` as Pareto-optimal.
        """
        W = self.S.union(self.P)

        is_index_pareto = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in W:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(self.order, pt_conf, pt_p_conf, self.u_star_eps):
                    is_index_pareto.append(False)
                    break
            else:
                is_index_pareto.append(True)

        tmp_S = copy.deepcopy(self.S)
        for is_pareto, pt in zip(is_index_pareto, tmp_S):
            if is_pareto:
                self.S.remove(pt)
                self.P.add(pt)

    def evaluating(self):
        """
        Selects designs for evaluation based on an acquisition function and
        updates the model with new observations.
        """
        W = self.S.union(self.P)
        acq = MaxDiagonalAcquisition(self.design_space)
        active_pts = self.design_space.points[list(W)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        observations = self.problem.evaluate(candidate_list)

        self.sample_count += len(candidate_list)
        self.model.add_sample(candidate_list, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        """
        Executes one iteration of the VOGP algorithm, performing modeling, discarding, epsilon-covering,
        and evaluating phases.

        :return: True if the set `S` is empty, indicating termination, False otherwise
        :rtype: bool
        """
        if len(self.S) == 0:
            return True

        round_str = f"Round {self.round}"

        logging.info(f"{round_str}:Modeling")
        self.modeling()

        logging.info(f"{round_str}:Discarding")
        self.discarding()

        logging.info(f"{round_str}:Epsilon-Covering")
        self.epsiloncovering()

        logging.info(f"{round_str}:Evaluating")
        if self.S:  # If S_t is not empty
            self.evaluating()

        logging.info(
            f"{round_str}:There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        self.round += 1

        logging.info(f"{round_str}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self):
        """
        Computes the confidence scaling parameter `beta` for Gaussian Process modeling.

        :return: A vector representing `beta` for each dimension of the output space
        :rtype: np.ndarray
        """
        # This is according to the proofs.
        beta_sqr = 2 * np.log(
            self.m
            * self.design_space.cardinality
            * (np.pi**2)
            * ((self.round + 1) ** 2)
            / (3 * self.delta)
        )
        return np.sqrt(beta_sqr / self.conf_contraction) * np.ones(
            self.m,
        )

    def compute_u_star(self):
        """
        Computes the normalized direction vector `u_star` and the ordering difficulty `d1` of a polyhedral ordering cone
        defined by the matrix `W`.

        :return: A tuple containing `u_star`, the normalized direction vector of the cone, and `d1`, the Euclidean norm of `z`
        :rtype: tuple (numpy.ndarray, float)

        :notes:
            The optimization is performed using the Sequential Least Squares Programming (SLSQP) method from
            `scipy.optimize.minimize`. The initial guess for `z` is a vector of ones. The optimization has a
            high maximum iteration limit and tight function tolerance to ensure convergence to an optimal solution.

        **Example**:
            >>> import numpy as np
            >>>
            >>> W = np.sqrt(21)*np.array([[1, -2, 4], [4, 1, -2], [-2, 4, 1]])
            >>> u_star_optimized,d_1 = compute_ustar_scipy(W)
            >>> print("u_star:", u_star_optimized)
            >>> print("d1:", d1)
        """

        cone_matrix = self.order.ordering_cone.W

        n = cone_matrix.shape[0]

        def objective(z):
            return np.linalg.norm(z)

        def constraint_func(z):
            constraints = []
            constraint = cone_matrix @ (z) - np.ones((n,))
            constraints.extend(constraint)
            return np.array(constraints)

        z_init = np.ones(self.m)  # Initial guess
        cons = [{"type": "ineq", "fun": lambda z: constraint_func(z)}]  # Constraints
        # Solving the problem
        res = minimize(
            objective,
            z_init,
            method="SLSQP",
            constraints=cons,
            options={"maxiter": 1000000, "ftol": 1e-30},
        )
        norm = np.linalg.norm(res.x)
        construe = np.all(constraint_func(res.x) + 1e-14)

        if not construe:
            pass

        return res.x / norm, norm

    def compute_pessimistic_set(self) -> set:
        """
        The pessimistic Pareto set of the set S+P of designs.

        :return: Set of pessimistic Pareto indices.
        :rtype: set
        """
        W = self.S.union(self.P)

        pessimistic_set = set()
        for pt in W:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in W:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                # Check if there is another point j that dominates i, if so,
                # do not include i in the pessimistic set
                if confidence_region_check_dominates(self.order, pt_p_conf, pt_conf):
                    break
            else:
                pessimistic_set.add(pt)

        return pessimistic_set
