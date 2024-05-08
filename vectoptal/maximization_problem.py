from abc import ABC, abstractmethod

import numpy as np

from vectoptal.datasets import Dataset
from vectoptal.utils import get_closest_indices_from_points, get_noisy_evaluations_chol


class Problem(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, x: np.ndarray):
        pass

class ProblemFromDataset(Problem):
    def __init__(self, dataset: Dataset, noise_var: float) -> None:
        super().__init__()

        self.dataset = dataset
        self.noise_var = noise_var

        noise_covar = np.eye(self.dataset.out_dim) * noise_var
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

    def evaluate(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        indices = get_closest_indices_from_points(x, self.dataset.in_data)
        f = self.dataset.out_data[indices].reshape(len(x), -1)
        y = get_noisy_evaluations_chol(f, self.noise_cholesky)
        return y
