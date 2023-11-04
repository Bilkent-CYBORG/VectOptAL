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

def _branin(X):
    x_0 = 15 * X[..., 0] - 5
    x_1 = 15 * X[..., 1]
    X = np.stack([x_0, x_1], axis=1)

    t1 = (
        X[..., 1]
        - 5.1 / (4 * math.pi**2) * X[..., 0] ** 2
        + 5 / math.pi * X[..., 0]
        - 6
    )
    t2 = 10 * (1 - 1 / (8 * math.pi)) * np.cos(X[..., 0])
    return t1**2 + t2 + 10

def _currin(X):
    x_0 = X[..., 0]
    x_1 = X[..., 1]
    factor1 = 1 - np.exp(-1 / (2 * x_1))
    numer = 2300 * np.power(x_0, 3) + 1900 * np.power(x_0, 2) + 2092 * x_0 + 60
    denom = 100 * np.power(x_0, 3) + 500 * np.power(x_0, 2) + 4 * x_0 + 20
    return factor1 * numer / denom

def branin_currin(X):
    branin = _branin(X)
    currin = _currin(X)
    
    Y = np.stack([-branin, -currin], axis=1)
    return Y


if __name__ == "__main__":
    np.random.seed(0)

    input_bounds = np.array([
        [0, 1],
        [0, 1],
    ])

    sample_count = 500

    if SOBOL:
        sampler = Sobol(2, scramble=False)
        sampler.fast_forward(np.random.randint(sample_count))
        samples = sampler.random(sample_count)

        for i, (l, u) in enumerate(input_bounds):
            samples[:, i] = samples[:, i] * (u - l) + l
        
        X = samples
    else:
        sample_count = sample_count
        Xi = []
        for l, u in input_bounds:
            Xi.append(np.random.rand(sample_count) * (u - l) + l)
        
        X = np.stack(Xi, axis=1)

    Y = branin_currin(X)

    np.save("bc500.npy", np.hstack((X, Y)))

    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X)
    output_scaler = StandardScaler()
    Y_scaled = output_scaler.fit_transform(Y)

    plt.scatter(Y_scaled[:, 0], Y_scaled[:, 1])
    plt.savefig("bc500.png")

    delta = get_delta_90(Y_scaled)
    print(sorted(delta[np.nonzero(delta)])[:5])
    print("Avg. gap:", np.mean(delta[np.nonzero(delta)]))
    print()
