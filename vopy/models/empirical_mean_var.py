from typing import Iterable

import numpy as np

from vopy.models.model import UncertaintyPredictiveModel


class EmpiricalMeanVarModel(UncertaintyPredictiveModel):
    """
    Implements a model that tracks empirical means and variances for each design.

    This class tracks the empirical means and variances for each design in the input space,
    controlled by flags `track_means` and `track_variances`.

    :param input_dim: The dimension of the input space.
    :type input_dim: int
    :param output_dim: The dimension of the output space.
    :type output_dim: int
    :param noise_var: The variance of the noise in the output space.
    :type noise_var: float
    :param design_count: The number of designs to track.
    :type design_count: int
    :param track_means: A flag to enable/disable tracking of means, defaults to True.
    :type track_means: bool
    :param track_variances: A flag to enable/disable tracking of variances, defaults to True.
    :type track_variances: bool
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        noise_var: float,
        design_count: int,
        track_means: bool = True,
        track_variances: bool = True,
    ):
        super().__init__()

        self.noise_var = noise_var
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.design_count = design_count

        self.track_means = track_means
        self.track_variances = track_variances

        # Data containers.
        self.clear_data()

    def add_sample(self, indices: Iterable[int], Y_t: np.ndarray):
        """
        Add new samples for specified design indices.

        :param indices: Represents the indices of designs to which the samples belong.
        :type indices: Iterable[int]
        :param Y_t: A N-by-output_dim array containing the new samples to be added.
        :type Y_t: np.ndarray
        """
        if len(indices) != len(Y_t):
            raise ValueError("Number of samples is ambiguous.")
        if max(indices) >= self.design_count:
            raise ValueError("Design index out of bounds.")

        for idx, y in zip(indices, Y_t):
            self.design_samples[idx] = np.concatenate(
                [self.design_samples[idx], y.reshape(-1, self.output_dim)], axis=0
            )

    def clear_data(self):
        """
        This method generates/clears the sample containers.
        """
        self.design_samples = [np.empty((0, self.output_dim)) for _ in range(self.design_count)]

    def update(self):
        """
        This method calculates and updates the means and variances of the design samples based on
        the current data. If `track_means` is enabled, it updates the `means` attribute with the
        mean of each design sample. If `track_variances` is enabled, it updates the `variances`
        attribute with the variance of each design sample.
        """
        if self.track_means:
            self.means = np.array(
                [
                    np.mean(design, axis=0) if len(design) > 0 else np.zeros(self.output_dim)
                    for design in self.design_samples
                ]
            )
        else:
            self.means = None

        if self.track_variances:
            self.variances = np.array(
                [
                    (
                        np.diag(np.var(design, axis=0))
                        if len(design) > 1
                        else np.eye(self.output_dim) * self.noise_var
                    )
                    for design in self.design_samples
                ]
            )
        else:
            self.variances = None

    def train(self):
        """
        This method is a no-op for this model.
        """
        pass

    def predict(self, test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This method takes test inputs and returns the predicted means and variances based on the
        tracked data. If `track_means` is enabled, it returns the corresponding means for the test
        inputs. If `track_variances` is enabled, it returns the corresponding variances for the
        test inputs.

        :param test_X: The test inputs for which predictions are to be made. The last column of
            `test_X` should contain indices.
        :type test_X: np.ndarray
        :return: A tuple containing two numpy arrays: the predicted means and variances.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if test_X.shape[1] != self.input_dim + 1:
            raise ValueError("Test data needs to have an additional column for indices.")

        indices = test_X[..., -1].astype(int)
        if self.track_means:
            means = self.means[indices]
        else:
            means = np.zeros((len(indices), self.output_dim))

        if self.track_variances:
            variances = self.variances[indices]
        else:
            variances = np.array([np.eye(self.output_dim) for _ in indices])

        return means, variances
