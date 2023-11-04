import os
import math
import logging

from copy import deepcopy

import numpy as np
import cvxpy as cp
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt

from itertools import repeat
from concurrent.futures import ProcessPoolExecutor as Pool

from models.auer import AuerSubgaussianModel
from models.indep_subg import IndependentSubgaussianModel

from utils.utils import get_alpha_vec, get_noisy_evaluations_chol
from utils.order_utils import (
    is_x_dominated, is_x_eps_pareto, x_not_dominated_by_y
)


class AlgorithmUE:
    def __init__(
        self, n, d, m, W, noise_var, delta, kernel, epsilon=0, batch_size=None
    ) -> None:
        self.design_count = n
        self.input_dim = d
        self.output_dim = m
        self.W = W
        self.noise_var = noise_var
        self.delta = delta
        self.epsilon = epsilon

        if not hasattr(self, "conf_contraction"):
            self.conf_contraction = 1
        
        self.batch_size = batch_size

        self.gp_mod = IndependentSubgaussianModel(
            input_dim=d, output_dim=m, noise_var=noise_var, kernel=kernel,
            delta=self.delta, design_count=self.design_count,
            conf_contraction=self.conf_contraction
        )

        noise_covar = np.eye(self.output_dim) * noise_var
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

        self.alpha_vec = get_alpha_vec(W)

        self.round = 0
        self.sample_count = 0
        self.undecided_count = 0

    def pretrain_model(self):
        self.gp_mod.add_sample(self.S_undecided, self.output_space)
        self.gp_mod.update()
        self.gp_mod.train(iter=200, lr=0.01)
        self.gp_mod.clear_data()

    # Update self.A_non_discarded
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (name == 'S_undecided' and hasattr(self, 'U_useful')) or name == 'U_useful':
            self.A_non_discarded = np.vstack((self.U_useful, self.S_undecided))

    def prepare(self, input_space, output_space):
        # Register indices as last column
        self.S_undecided = np.hstack((input_space, np.arange(input_space.shape[0])[:, None]))
        self.output_space = output_space
        
        self.pretrain_model()

        self.means = np.zeros((input_space.shape[0], self.output_dim))
        self.variances = np.zeros((input_space.shape[0],))
        self.P_pareto = np.empty((0, self.input_dim+1))
        self.U_useful = np.empty((0, self.input_dim+1))

        self.undecided_count = len(self.S_undecided)

    def get_samples(self):
        return self.A_non_discarded

    def run_once(self):
        logging.info(
            f"ROUND {self.round}"
            f" - Number of samples: {self.sample_count}"
            f" - Undecided design count: {len(self.S_undecided)}"
        )

        self.undecided_count = len(self.S_undecided)

        samples = self.get_samples()
        sample_means = self.output_space[samples[..., -1].astype(int)]
        self.gp_mod.add_sample(
            samples,
            get_noisy_evaluations_chol(sample_means, self.noise_cholesky)
        )

        self.sample_count += samples.shape[0]

        self.gp_mod.update()
        logging.debug("Model updated")

        # Update GP model
        m, v = self.gp_mod.predict(self.A_non_discarded)
        self.means[self.A_non_discarded[..., -1].astype(int)] = m
        self.variances[self.A_non_discarded[..., -1].astype(int)] = v
        logging.debug("Modelling done")

        # Discard
        logging.debug("Discarding")
        
        discard_indices = self.discard()
        self.S_undecided = np.delete(self.S_undecided, discard_indices, 0)

        # Pareto
        logging.debug("Pareto updating")
        
        pareto_indices = self.pareto()
        self.P_pareto = np.vstack((self.P_pareto, self.S_undecided[pareto_indices]))
        self.S_undecided = np.delete(self.S_undecided, pareto_indices, 0)

        # Useful
        logging.debug("Useful updating")

        useful_indices = self.useful()
        self.U_useful = self.P_pareto[useful_indices]

    def run(self):
        while len(self.S_undecided) != 0:
            self.round += 1
            self.run_once()
        
        return self.P_pareto[..., -1].astype(int), self.sample_count

    def discard(self):
        with Pool(max_workers=min(len(self.S_undecided), 8)) as pool:
            results = pool.map(
                is_x_dominated,
                repeat(self.A_non_discarded),
                self.S_undecided,
                repeat(self.W),
                repeat(self.means),
                repeat(self.variances),
                repeat(self.alpha_vec),
                repeat(self.epsilon),
            )
            results = list(results)

        discard_indices = np.flatnonzero(np.array(results))
        logging.debug("Discard IDs: " + str(self.A_non_discarded[discard_indices][..., -1].astype(int).tolist()))

        return discard_indices

    def pareto(self):
        with Pool(max_workers=max(1, min(len(self.S_undecided), 8))) as pool:
            results = pool.map(
                is_x_eps_pareto,
                repeat(self.A_non_discarded),
                self.S_undecided,
                repeat(self.W),
                repeat(self.means),
                repeat(self.variances),
                repeat(self.alpha_vec),
                repeat(self.epsilon),
            )
            results = list(results)

        pareto_indices = np.flatnonzero(np.array(results))
        logging.debug("Pareto IDs: " + str(self.A_non_discarded[pareto_indices][..., -1].astype(int).tolist()))

        return pareto_indices

    def useful(self):
        useful_indices = list(range(len(self.P_pareto)))
        logging.debug('Useful: ' + str(self.P_pareto[useful_indices][..., -1].astype(int).tolist()))
        return useful_indices


class Algorithm1(AlgorithmUE):
    def __init__(self, n, d, m, W, noise_var, delta, kernel, batch_size=None):
        super().__init__(n, d, m, W, noise_var, delta, kernel, batch_size)
    
class Algorithm2(AlgorithmUE):
    def __init__(self, n, d, m, W, noise_var, delta, kernel, batch_size=None):
        super().__init__(n, d, m, W, noise_var, delta, kernel, batch_size)

    def useful(self):
        useful_indices = list()
        for y_c_i, design_y in enumerate(self.P_pareto):
            y_i = design_y[-1].astype(int)

            for design_x in self.S_undecided:
                x_i = design_x[-1].astype(int)

                if not x_not_dominated_by_y(
                    self.W,
                    self.means[x_i], self.means[y_i], self.variances[x_i], self.variances[y_i],
                    self.alpha_vec, 0
                ):
                    useful_indices.append(y_c_i)
                    break
        
        logging.debug('Useful: ' + str(self.P_pareto[useful_indices][..., -1].astype(int).tolist()))
        return useful_indices

class Algorithm3(AlgorithmUE):
    def __init__(
        self, n, d, m, W, noise_var, delta, kernel, epsilon, num_pt_ellipse,
        conf_contraction=1, batch_size=None
    ):
        self.conf_contraction = conf_contraction
        super().__init__(n, d, m, W, noise_var, delta, kernel, epsilon, batch_size)

    def useful(self):
        useful_indices = list()
        for y_c_i, design_y in enumerate(self.P_pareto):
            y_i = design_y[-1].astype(int)

            for design_x in self.S_undecided:
                x_i = design_x[-1].astype(int)

                if not x_not_dominated_by_y(
                    self.W,
                    self.means[x_i], self.means[y_i], self.variances[x_i], self.variances[y_i],
                    self.alpha_vec, self.epsilon
                ):
                    useful_indices.append(y_c_i)
                    break

        logging.debug('Useful: ' + str(self.P_pareto[useful_indices][..., -1].astype(int).tolist()))
        return useful_indices


class AlgorithmAuer:
    def __init__(
            self, n, d, m, W, noise_var, delta, kernel, epsilon, conf_contraction=1,
            batch_size=None
        ) -> None:
        self.design_count = n
        self.input_dim = d
        self.output_dim = m
        self.W = W
        self.noise_var = noise_var
        self.delta = delta
        self.epsilon = epsilon

        self.conf_contraction = conf_contraction
        self.batch_size = 1 if batch_size is None else batch_size

        self.gp_mod = AuerSubgaussianModel(
            input_dim=d, output_dim=m, noise_var=noise_var, kernel=kernel,
            delta=self.delta, design_count=self.design_count,
            conf_contraction=self.conf_contraction
        )

        noise_covar = np.array([
            [noise_var, 0],
            [0, noise_var]
        ])
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

        self.alpha_vec = get_alpha_vec(W)

        self.round = 0
        self.sample_count = 0
        self.undecided_count = 0

    def m(self, i, j):
        return max(0, np.min(j-i))

    def M(self, i, j, epsilon=0):
        return max(0, np.max((i+epsilon)-j))

    def prepare(self, input_space, output_space):
        # Register indices as last column
        self.S_undecided = np.hstack((input_space, np.arange(input_space.shape[0])[:, None]))
        self.output_space = output_space
        
        self.means = np.zeros((input_space.shape[0], self.output_dim))
        self.variances = np.zeros((input_space.shape[0], ))
        self.P_pareto = np.empty((0, self.input_dim+1))

        self.undecided_count = len(self.S_undecided)

    def get_samples_all(self):
        return self.S_undecided

    def run_once(self):
        logging.info(
            f"ROUND {self.round}"
            f" - Number of samples: {self.sample_count}"
            f" - Undecided design count: {len(self.S_undecided)}"
        )
        
        self.undecided_count = len(self.S_undecided)

        samples = self.get_samples_all()
        sample_means = self.output_space[samples[..., -1].astype(int)]
        self.gp_mod.add_sample(
            samples,
            get_noisy_evaluations_chol(sample_means, self.noise_cholesky)
        )

        self.sample_count += samples.shape[0]

        self.gp_mod.update()
        logging.debug("Model updated")

        # Update GP model
        m, v = self.gp_mod.predict(self.S_undecided)
        self.means[self.S_undecided[..., -1].astype(int)] = m
        self.variances[self.S_undecided[..., -1].astype(int)] = v
        logging.debug("Modelling done")

        # Discard
        logging.debug("Discarding")
        
        discard_indices = self.discard()
        self.S_undecided = np.delete(self.S_undecided, discard_indices, 0)

        # Pareto
        logging.debug("Pareto updating")
        
        pareto_indices = self.pareto()
        self.P_pareto = np.vstack((self.P_pareto, self.S_undecided[pareto_indices]))
        self.S_undecided = np.delete(self.S_undecided, pareto_indices, 0)

    def run(self):
        while len(self.S_undecided) != 0:
            self.round += 1
            self.run_once()
        
        return self.P_pareto[..., -1].astype(int), self.sample_count

    def discard(self):
        results = []
        for design_i in self.S_undecided:
            i_i = design_i[-1].astype(int)

            for design_j in self.S_undecided:
                j_i = design_j[-1].astype(int)
                
                if i_i == j_i:
                    continue

                beta = self.variances[i_i] + self.variances[j_i]
                if np.all(self.m(self.means[i_i], self.means[j_i]) > beta):
                    results.append(True)  # Discard
                    break
            else:
                results.append(False)  # Do not discard

        discard_indices = np.flatnonzero(np.array(results))
        logging.debug(
            "Discard IDs: " + str(self.S_undecided[discard_indices][..., -1].astype(int).tolist())
        )

        return discard_indices

    def pareto(self):
        P1_orig_inds = []
        P1_actv_inds = []
        for actv_ind_i, design_i in enumerate(self.S_undecided):
            i_i = design_i[-1].astype(int)

            for design_j in self.S_undecided:
                j_i = design_j[-1].astype(int)

                if i_i == j_i:
                    continue

                beta = self.variances[i_i] + self.variances[j_i]
                if np.all(self.M(self.means[i_i], self.means[j_i], self.epsilon) < beta):
                    # P1_orig_inds.append(False)  # Not pareto
                    break
            else:  # Pareto
                P1_orig_inds.append(i_i)
                P1_actv_inds.append(actv_ind_i)

        pareto_indices = []
        for j_i, actv_j_i in zip(P1_orig_inds, P1_actv_inds):
            for design_i in self.S_undecided:
                i_i = design_i[-1].astype(int)
                
                if i_i in P1_orig_inds:
                    continue

                beta = self.variances[i_i] + self.variances[j_i]
                if np.all(self.M(self.means[i_i], self.means[j_i], self.epsilon) <= beta):
                    # Leave as useful
                    break
            else:
                # Add to P
                pareto_indices.append(actv_j_i)

        logging.debug(
            "Pareto IDs: " + str(self.S_undecided[pareto_indices][..., -1].astype(int).tolist())
        )

        return pareto_indices
