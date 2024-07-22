import copy
import logging

import numpy as np
from scipy.optimize import minimize

from vectoptal.order import Order
from vectoptal.datasets import get_dataset
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ContinuousProblem
from vectoptal.acquisition import SumVarianceAcquisition, optimize_acqf_discrete
from vectoptal.design_space import AdaptivelyDiscretizedDesignSpace
from vectoptal.models import CorrelatedExactGPyTorchModel, get_gpytorch_model_w_known_hyperparams
from vectoptal.confidence_region import (
    confidence_region_is_dominated,
    confidence_region_check_dominates,
    confidence_region_is_covered
)


class VOGP_AD(PALAlgorithm):
    def __init__(
        self, epsilon, delta,
        problem: ContinuousProblem, order: Order,
        noise_var,
        conf_contraction=32,
        batch_size=1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.order = order
        self.conf_contraction = conf_contraction
        
        self.batch_size = batch_size
        assert self.batch_size == 1, \
            "Only batch size of 1 is supported with adaptive discretization."

        assert hasattr(problem, "depth_max"), "Problem does not have a max depth defined."
        self.max_discretization_depth = problem.depth_max

        self.m = problem.out_dim

        self.design_space = AdaptivelyDiscretizedDesignSpace(
            problem.in_dim, problem.out_dim, delta=self.delta,
            max_depth=self.max_discretization_depth, confidence_type='hyperrectangle'
        )
        self.problem = problem

        self.model = get_gpytorch_model_w_known_hyperparams(
            CorrelatedExactGPyTorchModel, self.problem, noise_var, initial_sample_cnt=1
        )

        self.u_star, self.d1 = self.compute_u_star()
        self.u_star_eps = self.u_star * epsilon

        self.S = set(range(1))
        self.P = set()
        self.round = 0
        self.sample_count = 0
        self.enable_epsilon_covering = False

    def modeling(self):
        # Active nodes, union of sets s_t and p_t at the beginning of round t
        W = self.S.union(self.P)
        self.design_space.update(self.model, self.beta, list(W))

    def discarding(self):
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
        if not self.enable_epsilon_covering:
            for design_i in self.S:
                if self.design_space.point_depths[design_i] != self.max_discretization_depth:
                    return
            else:
                self.enable_epsilon_covering = True

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

    def evaluate_refine(self):
        W = self.S.union(self.P)
        acq = SumVarianceAcquisition(self.model)
        active_pts = self.design_space.points[list(W)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        candidate_pt = candidate_list[0]
        candidate_i = np.where(np.all(self.design_space.points == candidate_pt, axis=1))[0].item()
        should_refine = self.design_space.should_refine_design(self.model, candidate_i, self.beta)
        if should_refine:
            child_designs = self.design_space.refine_design(candidate_i)
            if candidate_i in self.S:
                self.S.remove(candidate_i)
                self.S = self.S.union(child_designs)
            else:  # candidate_pt in self.P
                self.P.remove(candidate_i)
                self.P = self.P.union(child_designs)
        else:
            observations = self.problem.evaluate(candidate_list)

            self.sample_count += len(candidate_list)
            self.model.add_sample(candidate_list, observations)
            self.model.update()

    def run_one_step(self) -> bool:
        print(f"Round {self.round}")

        self.beta = self.compute_beta()

        print(f"Round {self.round}:Modeling")
        self.modeling()

        print(f"Round {self.round}:Discarding")
        self.discarding()

        print(f"Round {self.round}:Epsilon-Covering")
        self.epsiloncovering()

        print(f"Round {self.round}:Evaluating")
        if self.S:  # If S_t is not empty
            self.evaluate_refine()

        print(
            f"There are {len(self.S)} designs left in set S and"
            f" {len(self.P)} designs in set P."
        )

        self.round += 1

        print(f"Round {self.round}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self):
        Kn = self.model.evaluate_kernel()
        rkhs_bound = 0.1
        beta_sqr = rkhs_bound + np.sqrt(
            self.problem.noise_var * np.log(
                (1 / self.problem.noise_var) * np.linalg.det(Kn + np.eye(len(Kn)))
            ) - 2 * np.log(self.delta)
        )

        beta_sqr = beta_sqr**2  # Instead of dividing with sqrt of conf_contraction.
        return np.sqrt(beta_sqr / self.conf_contraction) * np.ones(self.m, )

    def compute_u_star(self):
        """
        Given a matrix W that corresponds to a polyhedral ordering cone, this function 
        computes the ordering difficulty of the cone and u^* direction of the cone.
        
        The function solves an optimization problem where the objective is to minimize the
        Euclidean norm of `z`, subject to the constraint that W @ z >= 1 for each row
        of W. 
        
        Parameters
        ----------
        W : numpy.ndarray
            A numpy array representing the matrix that defines the half-spaces of the
            polyhedral cone. Each row of W corresponds to a linear constraint on `z`.
        
        Returns
        -------
        u_star of cone : numpy.ndarray
            The normalized optimized vector `z`.
        d(1) of cone : float
            The Euclidean norm of the optimized `z`.

        Prints
        ------

        d(1) of cone : float
            The Euclidean norm of the optimized `z`.
            
        Are constraints obeyed : bool
            A boolean flag indicating whether all cone constraints are satisfied. 
            This corresponds to the unit sphere to be inside of the cone.
            
        If the constraints are not obeyed, it also prints:
        
        Distance to cone hyperplanes : numpy.ndarray
            The distances from the optimized `z` to the hyperplanes defined by W.
        
        Notes
        -----
        The optimization problem is solved using the Sequential Least SQuares Programming (SLSQP)
        method provided by scipy.optimize.minimize. The initial guess is a vector of ones, and
        the optimization runs with a very high maximum iteration limit and tight function tolerance
        to ensure convergence.
        
        Examples
        --------
        >>> W = W = np.sqrt(21)*np.array([[1, -2, 4], [4, 1, -2], [-2, 4, 1]])
        >>> u_star_optimized,d_1 = compute_ustar_scipy(W)
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
        cons = [{'type': 'ineq', 'fun': lambda z: constraint_func(z)}] # Constraints 
        # Solving the problem
        res = minimize(
            objective, z_init, method='SLSQP', constraints=cons,
            options={'maxiter': 1000000, 'ftol': 1e-30}
        )
        norm = np.linalg.norm(res.x)
        construe = np.all(constraint_func(res.x) + 1e-14)
        
        # print(f"Optimized d(1) was found to be {norm}")
        # print(f"Optimized u_star was found to be {res.x/norm}")
        # print(f"Are constraints obeyed: {construe}")
        
        if not construe: 
            # print(f"Distance to cone hyperplanes: {constraint_func(res.x)}")
            pass
        
        return res.x/norm, norm

    def compute_pessimistic_set(self) -> set:
        """
        The pessimistic Pareto set of the set S+P of designs.
        :return: Set of pessimistic Pareto indices.
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
