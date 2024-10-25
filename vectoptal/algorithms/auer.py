import logging

import numpy as np

from vectoptal.order import ComponentwiseOrder
from vectoptal.datasets import get_dataset_instance
from vectoptal.design_space import FixedPointsDesignSpace
from vectoptal.algorithms.algorithm import PALAlgorithm
from vectoptal.maximization_problem import ProblemFromDataset
from vectoptal.models import EmpiricalMeanVarModel


class Auer(PALAlgorithm):
    def __init__(
        self, epsilon, delta, dataset_name, noise_var, conf_contraction=32, use_empirical_beta=False
    ) -> None:
        super().__init__(epsilon, delta)

        self.noise_var = noise_var
        self.conf_contraction = conf_contraction
        self.use_empirical_beta = use_empirical_beta

        dataset = get_dataset_instance(dataset_name)

        self.m = dataset.out_dim
        self.order = ComponentwiseOrder(self.m)

        # Trick to keep indices alongside points. This is for predictions from the model.
        in_data = np.hstack((dataset.in_data, np.arange(len(dataset.in_data))[:, None]))
        self.design_space = FixedPointsDesignSpace(
            in_data, dataset.out_dim, confidence_type="hyperrectangle"
        )
        self.problem = ProblemFromDataset(dataset, noise_var)

        self.model = EmpiricalMeanVarModel(
            dataset.in_dim,
            self.m,
            noise_var,
            self.design_space.cardinality,
            track_variances=self.use_empirical_beta,
        )

        self.S = set(range(self.design_space.cardinality))
        self.P = set()
        self.round = 0
        self.sample_count = 0

    def small_m(self, i, j):
        return max(0, np.min(j - i))

    def big_m(self, i, j):
        return max(0, np.max((i + self.epsilon) - j))

    def modeling(self):
        # All active designs have the same beta value if empirical beta is not used. In that case;
        # model does not track variances. If the empirical beta is used, variances are tracked and
        # used here to provide individual scales for each design but disabled when updating the
        # design space.

        self.beta_t = self.compute_beta()

        self.model.track_variances = self.use_empirical_beta and False  # always False
        self.design_space.update(self.model, self.beta_t, list(self.S))
        self.model.track_variances = self.use_empirical_beta and True

    def discarding(self):
        to_be_discarded = []
        for pt_i, pt in enumerate(self.S):
            pt_conf = self.design_space.confidence_regions[pt]
            pt_beta = self.beta_t[pt_i]
            for pt_prime_i, pt_prime in enumerate(self.S):
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]
                pt_p_beta = self.beta_t[pt_prime_i]

                beta = pt_beta + pt_p_beta
                if np.all(self.small_m(pt_conf.center, pt_p_conf.center) > beta):
                    to_be_discarded.append(pt)
                    break

        for pt in to_be_discarded:
            self.S.remove(pt)

    def pareto_updating(self):
        P1_pt_is = []
        P1_pts = []
        for pt_i, pt in enumerate(self.S):
            pt_conf = self.design_space.confidence_regions[pt]
            pt_beta = self.beta_t[pt_i]
            for pt_prime_i, pt_prime in enumerate(self.S):
                if pt_prime == pt:
                    continue

                pt_p_conf = self.design_space.confidence_regions[pt_prime]
                pt_p_beta = self.beta_t[pt_prime_i]

                beta = pt_beta + pt_p_beta
                if np.all(self.big_m(pt_conf.center, pt_p_conf.center) < beta):
                    break
            else:
                P1_pt_is.append(pt_i)
                P1_pts.append(pt)

        new_pareto_pts = []
        for p1_pt_i, p1_pt in zip(P1_pt_is, P1_pts):
            p1_pt_conf = self.design_space.confidence_regions[p1_pt]
            p1_pt_beta = self.beta_t[p1_pt_i]
            for pt_i, pt in enumerate(self.S):
                if pt in P1_pts:
                    continue

                pt_conf = self.design_space.confidence_regions[pt]
                pt_beta = self.beta_t[pt_i]

                beta = p1_pt_beta + pt_beta
                if np.all(self.big_m(pt_conf.center, p1_pt_conf.center) <= beta):
                    # Leave as useful
                    break
            else:
                # Add to P
                new_pareto_pts.append(p1_pt)

        for pt in new_pareto_pts:
            self.S.remove(pt)
            self.P.add(pt)
        logging.debug(f"Pareto: {str(self.P)}")

    def evaluating(self):
        active_pts = self.design_space.points[list(self.S)]

        observations = self.problem.evaluate(active_pts[:, :-1])

        self.sample_count += len(self.S)
        self.model.add_sample(self.S, observations)
        self.model.update()

    def run_one_step(self) -> bool:
        if len(self.S) == 0:
            return True

        self.round += 1
        print(f"Round {self.round}")

        print(f"Round {self.round}:Evaluating")
        self.evaluating()

        print(f"Round {self.round}:Modeling")
        self.modeling()

        print(f"Round {self.round}:Discarding")
        self.discarding()

        print(f"Round {self.round}:Pareto update")
        self.pareto_updating()

        print(
            f"There are {len(self.S)} designs left in set S and" f" {len(self.P)} designs in set P."
        )

        print(f"Round {self.round}:Sample count {self.sample_count}")

        return len(self.S) == 0

    def compute_beta(self):
        # Original beta
        if not self.use_empirical_beta:
            t1 = np.log((4 * self.design_space.cardinality * self.m * self.round**2) / self.delta)
            t2 = np.ones((len(self.S), self.m))
        else:  # Empirical beta
            # Indices are enough for prediction.
            active_pts = np.array(list(self.S)).reshape(-1, 1)

            v_hat = self.model.predict(active_pts)[1].diagonal(axis1=-2, axis2=-1)
            v_bar = 1
            t1 = np.log((self.design_space.cardinality * self.m * self.round) / self.delta)
            t2 = v_hat + v_bar * np.sqrt((4 * t1) / self.round)

        beta = np.sqrt((2 * t1 * t2) / self.round)

        return beta / self.conf_contraction
