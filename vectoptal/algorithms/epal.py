import copy

import numpy as np

from vectoptal.order import ComponentwiseOrder
from vectoptal.datasets import get_dataset_instance
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset
from vectoptal.acquisition import MaxDiagonalAcquisition, optimize_acqf_discrete
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.models import IndependentExactGPyTorchModel, get_gpytorch_model_w_known_hyperparams
from vectoptal.confidence_region import (
    confidence_region_is_dominated,
    confidence_region_check_dominates,
    confidence_region_is_covered,
)


class EpsilonPAL(PALAlgorithm):
    def __init__(
        self,
        epsilon,
        delta,
        dataset_name,
        noise_var,
        conf_contraction=9,
        batch_size=1,
    ) -> None:
        super().__init__(epsilon, delta)

        self.batch_size = batch_size
        self.conf_contraction = conf_contraction

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim
        self.order = ComponentwiseOrder(dim=self.m)

        self.design_space = FixedPointsDesignSpace(
            dataset.in_data, dataset.out_dim, confidence_type="hyperrectangle"
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = get_gpytorch_model_w_known_hyperparams(
            IndependentExactGPyTorchModel,
            self.problem,
            noise_var,
            initial_sample_cnt=1,
            X=dataset.in_data,
            Y=dataset.out_data,
        )

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.round = 0
        self.sample_count = 0

    def modeling(self):
        self.beta = self.compute_beta()
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
                if confidence_region_is_dominated(self.order, pt_conf, pt_p_conf, self.epsilon):
                    to_be_discarded.append(pt)
                    break

        for pt in to_be_discarded:
            self.S.remove(pt)

    def epsiloncovering(self):
        W = self.S.union(self.P)

        is_index_pareto = []
        for pt in self.S:
            pt_conf = self.design_space.confidence_regions[pt]
            for pt_prime in W:
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]

                if confidence_region_is_covered(self.order, pt_conf, pt_p_conf, self.epsilon):
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
        W = self.S.union(self.P)
        acq = MaxDiagonalAcquisition(self.design_space)
        active_pts = self.design_space.points[list(W)]
        candidate_list, _ = optimize_acqf_discrete(acq, self.batch_size, choices=active_pts)

        observations = self.problem.evaluate(candidate_list)

        self.sample_count += len(candidate_list)
        self.model.add_sample(candidate_list, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        if len(self.S) == 0:
            return True

        print(f"Round {self.round}")

        print(f"Round {self.round}:Modeling")
        self.modeling()

        print(f"Round {self.round}:Discarding")
        self.discarding()

        print(f"Round {self.round}:Epsilon-Covering")
        self.epsiloncovering()

        print(f"Round {self.round}:Evaluating")
        if self.S:  # If S_t is not empty
            self.evaluating()

        print(
            f"There are {len(self.S)} designs left in set S and" f" {len(self.P)} designs in set P."
        )

        self.round += 1

        print(f"Round {self.round}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self):
        # This is according to the proofs.
        beta_sqr = 2 * np.log(
            self.m
            * self.design_space.cardinality
            * (np.pi**2)
            * ((self.round + 1) ** 2)
            / (6 * self.delta)
        )
        return np.sqrt(beta_sqr / self.conf_contraction) * np.ones(
            self.m,
        )

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
