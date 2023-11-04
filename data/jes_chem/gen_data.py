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

import chembench


SOBOL = True
SAVE_DATA = True

if __name__ == "__main__":
    problems = [
        "SnAr",     # nucleophilic aromatic substitution
        # "VdV",      # Van de Vusse reaction
        # "PK1",      # Paal-Knorr reaction, fixed temperature
        # "PK2",      # Paal-Knorr reaction
        # "Lactose",  # isomerisation of lactose to lactulose
    ]
    sample_counts = [
        500,
        # 2000,
        # 2000,
        # 2000,
        # 250
    ]

    fig = plt.figure(figsize=(10, 6))
    for prob_i, problem_name in enumerate(problems):
        np.random.seed(0)
        problem = getattr(chembench, problem_name)(noise_std=0, negate=True)
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
                f"{problem_name}HALF.npy", np.hstack((X, Y))
            )

        input_scaler = StandardScaler()
        X_scaled = input_scaler.fit_transform(X)
        output_scaler = StandardScaler()
        Y_scaled = output_scaler.fit_transform(Y)

        tmp_ax = fig.add_subplot(2, 3, prob_i+1)
        tmp_ax.scatter(Y_scaled[:, 0], Y_scaled[:, 1])
        tmp_ax.set_title(problem_name)
    
    fig.tight_layout()
    plt.savefig(f"paretos.png")
