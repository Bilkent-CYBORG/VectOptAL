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

def disc_brake(X):
    t1 = (np.power(X[:, 1], 2) - np.power(X[:, 0], 2))
    f1 = 4.90 * 1e-5 * t1 * (X[:, 3] - 1)
    f2 = 9.82 * 1e+6 * (t1 / (X[:, 2] * X[:, 3] *  (np.power(X[:, 1], 3) - np.power(X[:, 0], 3))))
    
    # Minimize both objectives
    Y = np.stack([-f1, -f2], axis=1)
    return Y


if __name__ == "__main__":
    np.random.seed(0)

    input_bounds = np.array([
        [55, 80],
        [75, 110],
        [1000, 3000],
        [11, 20],
    ])

    sample_count = 500

    if SOBOL:
        sampler = Sobol(4, scramble=False)
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

    Y = disc_brake(X)

    np.save("brake500.npy", np.hstack((X, Y)))

    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X)
    output_scaler = StandardScaler()
    Y_scaled = output_scaler.fit_transform(Y)

    delta = get_delta_90(Y_scaled)
    print(sorted(delta[np.nonzero(delta)])[:5])
    print("Avg. gap:", np.mean(delta[np.nonzero(delta)]))
    print()

    plt.scatter(Y_scaled[:, 0], Y_scaled[:, 1])
    plt.savefig("brake.png")
