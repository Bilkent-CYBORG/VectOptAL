import numpy as np


class IndependentModel:
    def __init__(self, input_dim, output_dim, noise_var, kernel, design_count=206):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.design_count = design_count

        # Data containers.
        self.clear_data()

        self.noise_var = noise_var

        self.kernel = kernel
        
        self.model = None

    def add_sample(self, X_t, Y_t):
        # Last column of X_t are sample space indices.
        for idx, y in zip(X_t[..., -1].astype(int), Y_t):
            self.design_samples[idx] = np.concatenate(
                [self.design_samples[idx], y.reshape(-1, self.output_dim)], 0
            )

    def clear_data(self):
        self.design_samples = [np.empty((0, self.output_dim)) for _ in range(self.design_count)]

    def update(self):
        # Covariance matrix of sample mean.
        # https://stats.stackexchange.com/questions/166474/
        self.means = [
            np.mean(design, axis=0) if len(design) > 0 else 0 for design in self.design_samples
        ]
        self.variances = []
        for design in self.design_samples:
            if len(design) < 3:
                var = np.eye(self.output_dim) * self.noise_var
            else:
                var = np.cov(design, rowvar=False)
            var /= max(len(design), 1)
            self.variances.append(var)

    def train(self, iter=5, lr=0.01):
        pass

    def predict(self, test_X):
        indices = test_X[..., -1].astype(int)
        means = [self.means[ind] for ind in indices]
        variances = [self.variances[ind] for ind in indices]

        return means, variances
