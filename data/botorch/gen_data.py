'''
@inproceedings{
    tu2022joint,
    author={Ben Tu and Axel Gandy and Nikolas Kantas and Behrang Shafei},
    title={Joint Entropy Search for Multi-Objective Bayesian Optimization},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022},
    github={https://github.com/benmltu/JES/}
    url={https://openreview.net/forum?id=ZChgD8OoGds
}
'''


import os
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
from sklearn.preprocessing import StandardScaler

import torch
import botorch.test_functions.multi_objective as multi_objective


SOBOL = True
SAVE_DATA = True

if __name__ == "__main__":
    problems = [
        # "Penicillin",
        "VehicleSafety",
        # "CarSideImpact",
        # "WeldedBeam",
        # "DiscBrake",
    ]
    sample_counts = [
        # 500,
        2000,
        # 500,
        # 500,
        # 500,
    ]

    fig = plt.figure(figsize=(10, 6))
    for prob_i, problem_name in enumerate(problems):
        np.random.seed(0)
        problem = getattr(multi_objective, problem_name)(noise_std=0, negate=True)
        input_bounds = problem._bounds

        sample_count = sample_counts[prob_i]

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

        if SAVE_DATA:
            np.save(
                f"{problem_name}2K.npy", np.hstack((X, Y))
            )

        input_scaler = StandardScaler()
        X_scaled = input_scaler.fit_transform(X)
        output_scaler = StandardScaler()
        Y_scaled = output_scaler.fit_transform(Y)

        if problem.num_objectives == 2:
            tmp_ax = fig.add_subplot(1, 5, prob_i+1)
            tmp_ax.scatter(Y_scaled[:, 0], Y_scaled[:, 1])
        if problem.num_objectives == 3:
            tmp_ax = fig.add_subplot(1, 5, prob_i+1, projection='3d')
            tmp_ax.scatter(Y_scaled[:, 0], Y_scaled[:, 1], Y_scaled[:, 2])
        tmp_ax.set_title(problem_name)
    
    fig.tight_layout()
    plt.savefig(f"paretos.png")
