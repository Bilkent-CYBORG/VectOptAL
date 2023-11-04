import os
import numpy as np

import GPy


class GPyModel:
    def __init__(self, input_dim, output_dim, noise_var, kernel):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Training data containers.
        self.clear_data()

        self.noise_var = noise_var

        # Set up likelihood and model
        self.likelihood = GPy.likelihoods.Gaussian(variance=self.noise_var)
        self.model = None

    def add_sample(self, X_t, Y_t):
        self.X_T = np.concatenate([self.X_T, X_t[..., :self.input_dim]], 0)
        self.Y_T = np.concatenate([self.Y_T, Y_t], 0)

    def clear_data(self):
        self.X_T = np.empty((0, self.input_dim))
        self.Y_T = np.empty((0, self.output_dim))

    def update(self):
        train_X = np.concatenate([
            np.repeat(self.X_T, self.output_dim, axis=0),
            np.arange(self.output_dim).reshape(-1, 1).repeat(len(self.X_T), axis=1).T.reshape(-1, 1)
        ], 1)
        train_Y = self.Y_T.flatten().reshape(-1, 1)

        # kernel = GPy.kern.RBF(self.input_dim)
        # icm = GPy.util.multioutput.ICM(self.input_dim, self.output_dim, kernel, self.output_dim)

        # train_X = self.X_T
        # train_Y = self.Y_T

        kern1 = GPy.kern.RBF(self.input_dim)
        kern2 = GPy.kern.Coregionalize(1, output_dim=self.output_dim, rank=self.output_dim)

        # Create the model with new data and sparsity setting.
        if self.model == None:
            # likelihoods = [self.likelihood for _ in range(self.output_dim)]
            # self.model = GPy.models.GPCoregionalizedRegression(
            #     [train_X for _ in range(self.output_dim)],
            #     [train_Y[..., i:i+1] for i in range(self.output_dim)],
            #     kernel=icm
            # )
            self.model = GPy.models.GPRegression(
                train_X, train_Y,
                kernel=kern1**kern2,
                mean_function=GPy.mappings.Constant(4, 1)
            )
        else:
            # self.model = GPy.models.GPCoregionalizedRegression(
            #     [train_X for _ in range(self.output_dim)],
            #     [train_Y[..., i:i+1] for i in range(self.output_dim)],
            #     kernel=self.model.kern
            # )
            # self.model.set_XY(
            #     [train_X for _ in range(self.output_dim)],
            #     [train_Y[..., i:i+1] for i in range(self.output_dim)]
            # )
            self.model.set_XY(train_X, train_Y)

    def train(self, iter=5, lr=0.01):
        self.model.optimize(max_iters=iter)

    def predict(self, test_X):
        test_X = test_X[..., :-1]
        test_size = len(test_X)

        test_X = np.concatenate([
            np.repeat(test_X, self.output_dim, axis=0),
            np.arange(self.output_dim).reshape(-1, 1).repeat(len(test_X), axis=1).T.reshape(-1, 1)
        ], 1)

        means, var = self.model.predict_noiseless(test_X, full_cov=True)

        means = means.reshape(-1, self.output_dim)
        variances = np.empty((test_size, self.output_dim, self.output_dim))
        for i in range(test_size):
            start_i = i * self.output_dim
            variances[i] = var[
                start_i:start_i+self.output_dim,
                start_i:start_i+self.output_dim
            ]

            # variances[i] = (variances[i] + variances[i].T) / 2

        return means, variances
