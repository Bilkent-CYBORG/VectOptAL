import logging

import numpy as np

import torch
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize


torch.set_default_dtype(torch.float64)


class IndependentBotorchModel:
    def __init__(self, input_dim, output_dim, noise_var, kernel):
        super().__init__()

        self.device = 'cpu'

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Training data containers.
        self.clear_data()
        self.input_scaler = None
        self.output_scaler = None
        self.bounds = None

        self.noise_var = noise_var

        self.kernel = kernel

        self.model = None
        self.params = None

    def to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64).to(self.device)

        return data

    def add_sample(self, X_t, Y_t):
        if self.bounds is None:
            # First time calling, so the data is all designs.
            self.bounds = torch.tensor(self.input_dim * [[np.min(X_t), np.max(X_t)]]).T

        # Last column of X_t are sample space indices.
        X_t = self.to_tensor(X_t[..., :self.input_dim])
        Y_t = self.to_tensor(Y_t)

        X_t = normalize(X_t, self.bounds)

        self.X_T = torch.cat([self.X_T, X_t], 0)
        self.Y_T = torch.cat([self.Y_T, Y_t], 0)

    def clear_data(self):
        self.X_T = torch.empty((0, self.input_dim)).to(self.device)
        self.Y_T = torch.empty((0, self.output_dim)).to(self.device)

    def update(self):
        covar_module = ScaleKernel(
            self.kernel(
                ard_num_dims=self.input_dim,
                batch_shape=torch.Size([self.output_dim]),
                lengthscale_prior=GammaPrior(3.0, 6.0)
            ), batch_shape=torch.Size([self.output_dim]), outputscale_prior=GammaPrior(2.0, 0.15)
        )
        self.model = SingleTaskGP(
            self.X_T, self.Y_T, outcome_transform=Standardize(m=self.output_dim),
            covar_module=covar_module
        )
        if self.params:
            self.model.covar_module.base_kernel.lengthscale = self.params["length_scales"]
            self.model.covar_module.outputscale = self.params["output_scales"] ** 2
            self.model.likelihood.noise = self.params["noise"] ** 2

    def train(self, iter=5, lr=0.01):
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        self.params = {
            "length_scales": self.model.covar_module.base_kernel.lengthscale.detach(),
            "output_scales": torch.sqrt(self.model.covar_module.outputscale).detach(),
            "noise": torch.sqrt(self.noise_var * torch.ones_like(self.model.likelihood.noise)),
        }

    def predict(self, test_X):
        self.model.eval()

        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[..., :self.input_dim])
        test_X = normalize(test_X, self.bounds)

        with torch.no_grad():
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().T.numpy()  # Squeeze the sample dimension
            variances = torch.einsum('ij,ik->kij', torch.eye(self.output_dim), res.variance).cpu().numpy()
        
        return means, variances
