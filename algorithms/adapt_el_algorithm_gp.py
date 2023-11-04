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

from models.gpyt import GPModel
from models.bnn import BNN
from models.indep_gpyt import IndependentGPModel
# from models.indep_bot import IndependentBotorchModel

from utils.utils import get_alpha_vec, get_noisy_evaluations_chol
from utils.order_utils_gp import (
    is_x_dominated, is_x_eps_pareto, x_not_dominated_by_y,
    is_x_dominated_rect, is_x_eps_pareto_rect,
)


class AlgorithmUE:
    def __init__(
        self, n, d, m, W, noise_var, delta, kernel, independent, epsilon=0,
        batch_size=None, use_ellipsoid=True
    ) -> None:
        self.design_count = n
        self.input_dim = d
        self.output_dim = m
        self.W = W
        self.noise_var = noise_var
        self.delta = delta
        self.use_ellipsoid = use_ellipsoid
        self.epsilon = epsilon

        self.batch_size = batch_size

        GaussianProcessModel = IndependentGPModel if independent else GPModel  # IndependentBotorchModel
        # GaussianProcessModel = BNN
        self.gp_mod = GaussianProcessModel(
            input_dim=d, output_dim=m, noise_var=noise_var, kernel=kernel
        )

        if isinstance(noise_var, list):
            noise_covar = np.array(noise_var)
        else:
            noise_covar = np.eye(self.output_dim) * noise_var
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

        self.alpha_vec = get_alpha_vec(W)

        self.round = 0
        self.sample_count = 0
        self.undecided_count = 0

    def pretrain_model(self):
        self.gp_mod.add_sample(self.S_undecided, self.output_space)
        self.gp_mod.update()
        self.gp_mod.train(iter=200, lr=0.1)
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
        self.variances = np.zeros((input_space.shape[0], self.output_dim, self.output_dim))
        self.P_pareto = np.empty((0, self.input_dim+1))
        self.U_useful = np.empty((0, self.input_dim+1))

        self.undecided_count = len(self.S_undecided)

    def get_samples_sov(self):
        if self.batch_size != 1:
            tmp_mod = deepcopy(self.gp_mod)

        # Map dataset design index to active design index
        idx_map = dict(zip(self.A_non_discarded[..., -1], range(len(self.A_non_discarded))))
        to_pick = self.A_non_discarded[..., -1].astype(int)
        sample_cnt = min(len(to_pick), self.batch_size)
        samples = np.empty((sample_cnt, self.input_dim+1))
        sample_idxs = []
        # Choose maximum sum of variances
        vars = self.variances[to_pick].copy()
        for batch_i in range(sample_cnt):
            traces = np.trace(vars, axis1=1, axis2=2)
            # dets = np.linalg.det(vars)
            
            # Choose from eligible designs (currently active)
            sample_det_idx = np.argmax(traces)
            sample_idx = to_pick[sample_det_idx]
            # Get active index for sample
            active_sample_idx = idx_map[sample_idx]
            sample_idxs.append(sample_idx)
            # Pick sampled active designs to evaluate
            samples[batch_i] = self.A_non_discarded[active_sample_idx]

            if self.batch_size != 1:
                tmp_mod.add_sample(
                    samples[batch_i:batch_i+1][..., :-1], self.means[sample_idx:sample_idx+1]
                )
                tmp_mod.update()
                _, vars = tmp_mod.predict(self.A_non_discarded[..., :-1])

        logging.debug("Sample IDs: " + str(sample_idxs))

        return samples

    def get_samples(self):
        if self.batch_size == None:
            return self.A_non_discarded
        elif self.round == 1:
            initial_design_inds = np.random.choice(
                len(self.A_non_discarded), self.batch_size, replace=False
            )
            return self.A_non_discarded[initial_design_inds]
        else:
            return self.get_samples_sov()

    def run_once(self):
        logging.info(
            f"ROUND {self.round}"
            f" - Number of samples: {self.sample_count}"
            f" - Undecided design count: {len(self.S_undecided)}"
        )

        self.alpha = (1 / self.conf_contraction) * (
            8*self.output_dim*math.log(6) + (4)*math.log(
                (math.pi**2 * self.round**2 * len(self.output_space))/(6*self.delta)
            )
        )
        logging.info(f"alpha_t is currently: {self.alpha:.3f}")

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
        dom_func = is_x_dominated_rect if not self.use_ellipsoid else is_x_dominated
        with Pool(max_workers=min(len(self.S_undecided), 8)) as pool:
            results = pool.map(
                dom_func,
                repeat(self.A_non_discarded),
                self.S_undecided,
                repeat(self.W),
                repeat(self.means),
                repeat(self.variances),
                repeat(self.alpha),
                repeat(self.alpha_vec),
                repeat(self.epsilon),
            )
            results = list(results)

        discard_indices = np.flatnonzero(np.array(results))
        logging.debug("Discard IDs: " + str(self.A_non_discarded[discard_indices][..., -1].astype(int).tolist()))

        return discard_indices

    def pareto(self):
        dom_func = is_x_eps_pareto_rect if not self.use_ellipsoid else is_x_eps_pareto
        with Pool(max_workers=max(1, min(len(self.S_undecided), 8))) as pool:
            results = pool.map(
                dom_func,
                repeat(self.A_non_discarded),
                self.S_undecided,
                repeat(self.W),
                repeat(self.means),
                repeat(self.variances),
                repeat(self.alpha),
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
                    self.alpha, self.alpha_vec, 0
                ):
                    useful_indices.append(y_c_i)
                    break
        
        logging.debug('Useful: ' + str(self.P_pareto[useful_indices][..., -1].astype(int).tolist()))
        return useful_indices

class Algorithm3(AlgorithmUE):
    def __init__(
            self, n, d, m, W, noise_var, delta, kernel, epsilon, num_pt_ellipse,
            independent, conf_contraction=1, batch_size=None, use_ellipsoid=True
        ):
        self.conf_contraction = conf_contraction
        super().__init__(
            n, d, m, W, noise_var, delta, kernel, independent, epsilon, batch_size, use_ellipsoid
        )

    def useful(self):
        useful_indices = list()
        if self.use_ellipsoid:
            for y_c_i, design_y in enumerate(self.P_pareto):
                y_i = design_y[-1].astype(int)

                for design_x in self.S_undecided:
                    x_i = design_x[-1].astype(int)

                    if not x_not_dominated_by_y(
                        self.W,
                        self.means[x_i], self.means[y_i], self.variances[x_i], self.variances[y_i],
                        self.alpha, self.alpha_vec, self.epsilon
                    ):
                        useful_indices.append(y_c_i)
                        break
        else:
            for y_c_i, design_y in enumerate(self.P_pareto):
                y_i = design_y[-1].astype(int)
                by = np.array(self.means[y_i] + self.alpha*np.sqrt(np.diag(self.variances[y_i])))
                
                for design_x in self.S_undecided:
                    x_i = design_x[-1].astype(int)
                    bx = np.array(self.means[x_i] - self.alpha*np.sqrt(np.diag(self.variances[x_i])))

                    if np.min(by - bx) > self.epsilon:
                        useful_indices.append(y_c_i)
                        break
        
        logging.debug('Useful: ' + str(self.P_pareto[useful_indices][..., -1].astype(int).tolist()))
        return useful_indices
