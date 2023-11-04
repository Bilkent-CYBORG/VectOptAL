# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:47:58 2023

@author: EfeMertKaragozlu
"""
import math
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn
torch.set_default_dtype(torch.float64)


class BNN:
    def __init__(
            self, input_dim, output_dim, noise_var, kernel,
            architecture=[10, 10], sample_cnt=100, epochs=1000, kl_weight=0.001
        ):
        #param_var is the variance for model's parameters
        #architecture is a list of hidden layer neuron numbers
        #sampling_no works like alpha_t, i.e shrinks or exagerates the effect of variance, it's a function
        #epochs controls makes the model more overfitting via more epochs run on the same data

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = 'cpu'

        self.noise_var = np.sqrt(noise_var)
        self.sample_cnt = sample_cnt
        self.epochs = epochs
        self.architecture = architecture

        self.model = self.gen_model(self.architecture)
                
        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = kl_weight
        
        self.clear_data()

    def gen_model(self, architecture):
        layers = [
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=self.noise_var, in_features=self.input_dim,
                out_features=self.architecture[0]
            ),
        ]

        for i in range(len(architecture)-1):
            layers.append(nn.ReLU())
            layers.append(
                bnn.BayesLinear(
                    prior_mu=0, prior_sigma=self.noise_var,
                    in_features=self.architecture[i], out_features=self.architecture[i+1]
                )
            )
        
        layers.append(nn.ReLU())
        layers.append(
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=self.noise_var,
                in_features=self.architecture[-1], out_features=self.output_dim
            )
        )

        return nn.Sequential(*layers)

    def to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64).to(self.device)
        
        return data

    def add_sample(self, X_t, Y_t):
        X_t = self.to_tensor(X_t[..., :self.input_dim])
        Y_t = self.to_tensor(Y_t)

        self.X_T = torch.cat([self.X_T, X_t], 0)
        self.Y_T = torch.cat([self.Y_T, Y_t], 0)

    def clear_data(self):
        self.model = self.gen_model(self.architecture)

        self.X_T = torch.empty((0, self.input_dim))
        self.Y_T = torch.empty((0, self.output_dim))

    def train(self, iter=5, lr=0.01):
        pass

    def update(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        for step in range(self.epochs):
            pre = self.model(self.X_T)
            mse = self.mse_loss(pre, self.Y_T)
            kl = self.kl_loss(self.model)
            cost = mse + self.kl_weight*kl

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if step % 50 == 0:
                logging.info('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

    def predict(self, test_X):
        test_X = self.to_tensor(test_X[..., :self.input_dim])

        with torch.no_grad():
            samples = torch.empty((self.sample_cnt, len(test_X), self.output_dim))
            for s in range(self.sample_cnt):
                samples[s] = self.model(test_X)
            
            means = samples.mean(dim=0)
            variances = torch.empty((len(test_X), self.output_dim, self.output_dim))
            for x_ind in range(len(test_X)):
                variances[x_ind] = torch.cov(samples[:, x_ind].T)

        return means, variances

        # samples_tensor = torch.stack(samples)
        # maximal_points, ind1 = torch.max(samples_tensor, 0)
        # minimal_points, ind2 = torch.min(samples_tensor, 0)
        # return np.array(maximal_points.data), np.array(minimal_points.data)
