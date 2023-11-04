import logging

import numpy as np

import torch
import gpytorch


torch.set_default_dtype(torch.float64)


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

class IndependentGPModel:
    def __init__(self, input_dim, output_dim, noise_var, kernel):
        super().__init__()

        self.device = 'cpu'

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Training data containers.
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

        self.kernel = kernel
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
            self.model = BatchIndependentExactGPModel(
                self.X_T, self.Y_T, self.likelihood, self.kernel
            ).to(self.device)
        else:
            self.model.set_train_data(self.X_T, self.Y_T, strict=False)

        self.model.eval()
        self.likelihood.eval()

    def train(self, iter=5, lr=0.01):
        self.model.train()
        self.likelihood.train()

        training_iterations = iter
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iterations):
            should_print = True if i == 0 or i == training_iterations-1 else False
            def closure():
                nonlocal should_print
                
                # Zero backprop gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Get output from model
                output = self.model(self.X_T)

                # Calculate loss and backprop derivatives
                loss = -mll(output, self.Y_T)
                loss.backward()

                log_text = f'Train iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f}'
                if should_print:
                    logging.info(log_text)
                    should_print = False
                
                return loss
            
            optimizer.step(closure)

        self.model.eval()
        self.likelihood.eval()

    def predict(self, test_X):
        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[..., :self.input_dim])

        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().numpy()  # Squeeze the sample dimension
            variances = torch.einsum('ij,ki->kij', torch.eye(self.output_dim), res.variance).cpu().numpy()
        
        return means, variances
