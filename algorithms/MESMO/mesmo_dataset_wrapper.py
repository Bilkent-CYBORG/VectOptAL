import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import utils.dataset
from utils.utils import get_noisy_evaluations_chol


class MESMODatasetWrapper():
    def __init__(self, dataset: utils.dataset.Dataset, noise_var: float):
        self.in_dim = dataset.in_data.shape[1]
        self.out_dim = dataset.out_data.shape[1]
        self.designs = dataset.in_data

        noise_covar = np.eye(self.out_dim) * noise_var
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

        self.dataset = dataset

    def get_design_indices_from_points(self, X):
        distances = euclidean_distances(X, self.dataset.in_data, squared=True)
        x_inds = np.argmin(distances, axis=1)
        return x_inds.astype(int)
    
    def evaluate(self, x):
        x_ind_arr = self.get_design_indices_from_points([x])
        y = self.dataset.out_data[x_ind_arr]

        y = get_noisy_evaluations_chol(y.reshape(1, -1), self.noise_cholesky)

        return y.tolist()[0]

    @property
    def bounds(self):
        return self.in_dim * [[
            np.min(self.dataset.in_data),
            np.max(self.dataset.in_data)
        ]]
