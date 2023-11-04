'''
@article{Tanabe_2020,
	author = {Ryoji Tanabe and Hisao Ishibuchi},
	title = {An easy-to-use real-world multi-objective optimization problem suite},
	year = 2020,
	journal = {Applied Soft Computing}
    github = https://github.com/ryojitanabe/reproblems
}
'''


import os
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
from sklearn.preprocessing import StandardScaler

import torch

from problem import MarineDesign


def get_delta_90(mu):
    n = mu.shape[0]
    Delta = np.zeros(n)
    for i in range(n):
        vi = mu[i, :].reshape(1, -1)
        difs = mu - vi
        difs[difs < 0] = 0
        smallmij = np.min(difs, axis=1)
        Delta[i] = np.max(smallmij)

    return Delta.reshape(-1,1)

SOBOL = True

if __name__ == "__main__":
    np.random.seed(0)

    problem = MarineDesign(noise_std=0, negate=True)
    input_bounds = problem._bounds

    sample_count = 2000

    if SOBOL:
        sampler = Sobol(problem.dim, scramble=False)
        sampler.fast_forward(np.random.randint(low=1, high=max(2, sample_count)))
        samples = sampler.random(sample_count)

        for i, (l, u) in enumerate(input_bounds):
            samples[:, i] = samples[:, i] * (u - l) + l
        
        X = samples
    else:
        Xi = []
        for l, u in input_bounds:
            Xi.append(np.random.rand(sample_count) * (u - l) + l)
        
        X = np.stack(Xi, axis=1)

    Y = problem(torch.tensor(X)).detach().cpu().numpy()

    np.save(
        f"marine2k.npy", np.hstack((X, Y))
    )

    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X)
    output_scaler = StandardScaler()
    Y_scaled = output_scaler.fit_transform(Y)

    delta = get_delta_90(Y_scaled)
    print(sorted(delta[np.nonzero(delta)])[:5])
    print("Avg. gap:", np.mean(delta[np.nonzero(delta)]))
    print()

