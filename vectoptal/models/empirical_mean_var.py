import numpy as np

from vectoptal.models import Model


class EmpiricalMeanVarModel(Model):
    def __init__(
        self, input_dim, output_dim, noise_var, design_count,
        track_means: bool = True, track_variances: bool = True
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

    def add_sample(self, indices, Y_t):
        # Last column of X_t are sample space indices.
        for idx, y in zip(indices, Y_t):
            self.design_samples[idx] = np.concatenate(
                [self.design_samples[idx], y.reshape(-1, self.output_dim)], axis=0
            )

    def clear_data(self):
        self.design_samples = [np.empty((0, self.output_dim)) for _ in range(self.design_count)]

    def update(self):
        if self.track_means:
            self.means = np.array([
                np.mean(design, axis=0)
                if len(design) > 0 else np.zeros(self.output_dim)
                for design in self.design_samples
            ])
        if self.track_variances:
            self.variances = np.array([
                np.diag(np.var(design, axis=0))
                if len(design) > 1 else np.eye(self.output_dim) * self.noise_var
                for design in self.design_samples
            ])

    def train(self):
        pass

    def predict(self, test_X) -> tuple[np.ndarray, np.ndarray]:
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
