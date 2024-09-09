from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np

from vectoptal.datasets import Dataset
from vectoptal.utils import get_closest_indices_from_points, get_noisy_evaluations_chol


# TODO: Revise evaluate abstract method.
# It should both accomodate evaluations that are noisy in themselves and synthetically noisy ones.
# Maybe distinguish real problems from test problems.
class Problem(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

class ProblemFromDataset(Problem):
    def __init__(self, dataset: Dataset, noise_var: float) -> None:
        super().__init__()

        self.dataset = dataset
        self.noise_var = noise_var

        noise_covar = np.eye(self.dataset.out_dim) * noise_var
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

    def evaluate(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        if x.ndim <= 1:
            x = x.reshape(1, -1)

        indices = get_closest_indices_from_points(x, self.dataset.in_data, squared=True)
        f = self.dataset.out_data[indices].reshape(len(x), -1)
        
        if not noisy:
            return f
        
        y = get_noisy_evaluations_chol(f, self.noise_cholesky)
        return y

class ContinuousProblem(Problem):
    def __init__(self, noise_var: float) -> None:
        super().__init__()
        
        self.noise_var = noise_var

        noise_covar = np.eye(self.out_dim) * noise_var
        self.noise_cholesky = np.linalg.cholesky(noise_covar)

class BraninCurrin(ContinuousProblem):
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    in_dim = len(bounds)
    out_dim = 2
    domain_discretization_each_dim = 33
    depth_max = 5

    def __init__(self, noise_var: float) -> None:
        super().__init__(noise_var)

    def _branin(self, X):
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        X = np.stack([x_0, x_1], axis=1)

        t1 = (
            X[..., 1]
            - 5.1 / (4 * np.pi**2) * X[..., 0] ** 2
            + 5 / np.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(X[..., 0])
        return t1**2 + t2 + 10

    def _currin(self, X):
        x_0 = X[..., 0]
        x_1 = X[..., 1]
        x_1[x_1 == 0] += 1e-9
        factor1 = 1 - np.exp(-1 / (2 * x_1))
        numer = 2300 * np.power(x_0, 3) + 1900 * np.power(x_0, 2) + 2092 * x_0 + 60
        denom = 100 * np.power(x_0, 3) + 500 * np.power(x_0, 2) + 4 * x_0 + 20
        return factor1 * numer / denom

    def evaluate_true(self, x: np.ndarray) -> np.ndarray:
        branin = self._branin(x)
        currin = self._currin(x)

        # Normalize the results
        branin = (branin - 54.3669) / 51.3086
        currin = (currin - 7.5926) / 2.6496
        
        Y = np.stack([-branin, -currin], axis=1)
        return Y

    def evaluate(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        f = self.evaluate_true(x)

        if not noisy:
            return f
        
        y = get_noisy_evaluations_chol(f, self.noise_cholesky)
        return y

class DecoupledEvaluationProblem(Problem):
    def __init__(self, problem: Problem) -> None:
        super().__init__()
        self.problem = problem
    
    def evaluate(
        self, x: np.ndarray, evaluation_index: Optional[Union[int, List[int]]]=None
    ) -> np.ndarray:
        values = self.problem.evaluate(x)

        if evaluation_index is None:
            return values
        
        if isinstance(evaluation_index, int):
            return values[:, evaluation_index]
        
        assert len(x) == len(evaluation_index), \
            "evaluation_index should be the same length as data"

        evaluation_index = np.array(evaluation_index, dtype=np.int32)
        return values[np.arange(len(evaluation_index)), evaluation_index]
