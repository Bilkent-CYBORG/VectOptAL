from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool

import torch
import gpytorch
import numpy as np

import utils.dataset
from utils.seed import SEED
from utils.utils import set_seed, save_new_result

from algorithms.ADELGP.ADELGP.Polyhedron import Polyhedron
from algorithms.ADELGP.ADELGP.VectorEpsilonPAL import VectorEpsilonPAL
from algorithms.ADELGP.ADELGP.OptimizationProblem import OptimizationProblem
from algorithms.ADELGP.ADELGP.GaussianProcessModel import GaussianProcessModelDependent


def simulate_once(i, dataset_name, cone_degree, noise_var, epsilon, delta):
    set_seed(SEED + i + 1)
    
    TrainGPDuringAlgorithm = False
    batched = False
    DEVICE = 'cpu'

    std = np.sqrt(noise_var)

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)

    x = torch.from_numpy(dataset.in_data).to(DEVICE)
    y = torch.from_numpy(dataset.out_data).to(DEVICE)

    problem_model = OptimizationProblem(x, y, std)

    kernel = gpytorch.kernels.MultitaskKernel(
        dataset.model_kernel(), num_tasks=dataset.out_dim, rank=dataset.out_dim
    )
    
    x_sample = x
    y_sample = y

    A = dataset.W

    b = np.zeros((dataset.out_dim,))
    C = Polyhedron(A = A, b = b)

    gp = GaussianProcessModelDependent(
        d = dataset.in_dim, m = dataset.out_dim, noise_variance=noise_var, x_sample = x_sample,
        y_sample=y_sample, kernel=kernel, verbose=True, device=DEVICE,
        train_during_alg=TrainGPDuringAlgorithm
    )

    alg = VectorEpsilonPAL(
        problem_model = problem_model, cone = C, epsilon = epsilon, delta = delta,
        gp = gp,obj_dim=dataset.out_dim,maxiter=None, batched= batched
    )

    pareto_set = alg.algorithm()

    pareto_indices = [pareto_pt.design_index for pareto_pt in pareto_set]

    return [alg.sample_count, pareto_indices]


def run_adelgp(
    datasets_and_workers, cone_degrees, noise_var, delta, epsilons, iteration, output_folder_path
):    
    # dset, eps, cone, conf, batch
    for dataset_name, dataset_worker in datasets_and_workers:
        for eps in epsilons:
            for cone_degree in cone_degrees:
                simulate_part = partial(
                    simulate_once,
                    dataset_name=dataset_name,
                    cone_degree=cone_degree,
                    delta=delta,
                    noise_var=noise_var,
                    epsilon=eps,
                )

                with Pool(max_workers=dataset_worker) as pool:
                    results = pool.map(
                        simulate_part,
                        range(iteration),
                    )

                results = list(results)
                
                experiment_res_dict = {
                    "dataset_name": dataset_name,
                    "cone_degree": cone_degree,
                    "alg": "ADELGP",
                    "delta": delta,
                    "noise_var": noise_var,
                    "eps": eps,
                    "disc": -1,
                    "conf_contraction": 20,
                    "batch_size": 1,
                    "results": results
                }

                save_new_result(output_folder_path, experiment_res_dict)
