import logging
from abc import ABC, abstractmethod

import torch
import gpytorch
from botorch.fit import fit_gpytorch_model

import numpy as np

from vectoptal.models import Model
from vectoptal.maximization_problem import Problem

torch.set_default_dtype(torch.float64)


class MultitaskExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, input_X, input_Y, likelihood, kernel):
        if not isinstance(input_X, torch.Tensor):
            input_X = torch.tensor(input_X, dtype=torch.float64)
        if not isinstance(input_Y, torch.Tensor):
            input_Y = torch.tensor(input_Y, dtype=torch.float64)

        input_dim = input_X.shape[-1]
        output_dim = input_Y.shape[-1]

        super().__init__(input_X, input_Y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=output_dim
        )

        # self.base_covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=input_dim))
        self.base_covar_module = kernel(
            ard_num_dims=input_dim,
            lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
        )  # Scale kernel is unnecessary
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_covar_module, num_tasks=output_dim, rank=output_dim
        )

        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class BatchIndependentExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, input_X, input_Y, likelihood, kernel):
        if not isinstance(input_X, torch.Tensor):
            input_X = torch.tensor(input_X, dtype=torch.float64)
        if not isinstance(input_Y, torch.Tensor):
            input_Y = torch.tensor(input_Y, dtype=torch.float64)

        input_dim = input_X.shape[-1]
        output_dim = input_Y.shape[-1]

        super().__init__(input_X, input_Y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([output_dim]))

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel(batch_shape=torch.Size([output_dim]), ard_num_dims=input_dim),
            batch_shape=torch.Size([output_dim])
        )
        
        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

class GPyTorchExactModel(Model, ABC):
    """Assumes known noise variance."""
    def __init__(
        self, input_dim, output_dim, noise_var, model_kind: gpytorch.models.ExactGP
    ) -> None:
        super().__init__()

        self.device = 'cpu'

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_kind = model_kind

        # Data containers.
        self.clear_data()

        # Set up likelihood
        self.noise_var = torch.tensor(noise_var)
        if self.noise_var.dim() > 1:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.output_dim,
                rank=len(self.noise_var),
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
                has_global_noise=False
            ).to(self.device)
            self.likelihood.task_noise_covar = self.noise_var
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.output_dim,
                rank=0,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
                has_task_noise=False
            ).to(self.device)
            self.likelihood.noise = self.noise_var
        self.likelihood.requires_grad_(False)

        self.kernel_type = gpytorch.kernels.RBFKernel
        self.model = None

    def to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64).to(self.device)
        
        return data
    
    def add_sample(self, X_t, Y_t):
        # Last column of X_t are sample space indices.
        X_t = self.to_tensor(X_t[..., :self.input_dim])
        Y_t = self.to_tensor(Y_t)

        self.X_T = torch.cat([self.X_T, X_t], 0)
        self.Y_T = torch.cat([self.Y_T, Y_t], 0)

    def clear_data(self):
        self.X_T = torch.empty((0, self.input_dim)).to(self.device)
        self.Y_T = torch.empty((0, self.output_dim)).to(self.device)

    def update(self):
        # Create the model with new data and sparsity setting.
        if self.model == None:
            self.model = MultitaskExactGPModel(
                self.X_T, self.Y_T, self.likelihood, self.kernel_type
            ).to(self.device)
        else:
            self.model.set_train_data(self.X_T, self.Y_T, strict=False)

        self.model.eval()
        self.likelihood.eval()

    def train(self):
        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        logging.info("Training started.")
        fit_gpytorch_model(mll)
        logging.info("Training done.")

        self.model.eval()
        self.likelihood.eval()


class CorrelatedExactGPyTorchModel(GPyTorchExactModel):
    def __init__(self, input_dim, output_dim, noise_var) -> None:
        super().__init__(input_dim, output_dim, noise_var, MultitaskExactGPModel)
    
    def predict(self, test_X):
        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[:, :self.input_dim])

        test_X = test_X[:, None, :]  # Prepare for batch inference

        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().numpy()  # Squeeze the sample dimension
            variances = res.covariance_matrix
            variances = variances.cpu().numpy()
        
        return means, variances

class IndependentExactGPyTorchModel(GPyTorchExactModel):
    def __init__(self, input_dim, output_dim, noise_var) -> None:
        super().__init__(input_dim, output_dim, noise_var, BatchIndependentExactGPModel)
    
    def predict(self, test_X):
        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[..., :self.input_dim])

        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().numpy()  # Squeeze the sample dimension
            variances = torch.einsum(
                'ij,ki->kij', torch.eye(self.output_dim), res.variance
            ).cpu().numpy()
        
        return means, variances


def get_gpytorch_model_w_known_hyperparams(
    model_class, problem: Problem, X:np.ndarray, Y:np.ndarray,
    noise_var: float, initial_sample_cnt: int
) -> GPyTorchExactModel:
    in_dim = X.shape[1]
    out_dim = Y.shape[1]

    model = model_class(in_dim, out_dim, noise_var=noise_var)

    model.add_sample(X, Y)
    model.update()
    model.train()
    model.clear_data()

    if initial_sample_cnt > 0:
        initial_indices = np.random.choice(len(X), initial_sample_cnt)
        initial_points = X[initial_indices]
        initial_values = problem.evaluate(initial_points)

        model.add_sample(initial_points, initial_values)
        model.update()

    return model
