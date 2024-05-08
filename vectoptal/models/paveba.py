import numpy as np

from vectoptal.models import Model


class PaVeBaModel(Model):
    def __init__(self, input_dim, output_dim, noise_var, design_count):
        super().__init__()

        self.noise_var = noise_var
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.design_count = design_count

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
        self.means = np.array([
            np.mean(design, axis=0)
            if len(design) > 0 else np.zeros(self.output_dim)
            for design in self.design_samples
        ])

    def train(self):
        pass

    def predict(self, test_X):
        indices = test_X[..., -1].astype(int)
        means = self.means[indices]
        variances = np.array([np.eye(self.output_dim) for _ in indices])

        return means, variances
