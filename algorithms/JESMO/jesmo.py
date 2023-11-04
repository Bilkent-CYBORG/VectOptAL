import logging
from concurrent.futures import ProcessPoolExecutor as Pool

import numpy as np

import torch
from botorch.utils.transforms import unnormalize, normalize

import utils.dataset
from utils.seed import SEED
from utils.utils import set_seed, get_pareto_set, get_cone_params, save_new_result
from algorithms.MESMO.mesmo_dataset_wrapper import MESMODatasetWrapper

from algorithms.JESMO.jes.utils.bo_loop import bo_loop, fit_model


def jesmo_run(
    alg_name, dataset_worker, sampling_budgets, noise_var, epsilons, iteration, output_folder_path
):
    global iter_func  # !!! NASTY SOLUTION !!!

    if alg_name == "JESMO":
        acq_type = "jes_0"
    elif alg_name == "MES_0":
        acq_type = "mes_0"
    else:
        raise RuntimeError

    dataset_name = dataset_worker[0]
    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(90)
    dataset = MESMODatasetWrapper(dataset, noise_var)

    W, alpha_vec, _ = get_cone_params(90, dim=dataset.out_dim)

    input_dim = dataset.in_dim
    output_dim = dataset.out_dim
    function_bounds = dataset.bounds
    bounds = torch.tensor(function_bounds).T
    designs = dataset.designs
    designs_tensor = normalize(torch.tensor(designs), bounds)
    grid_size = len(dataset.designs)


    standard_bounds = torch.zeros(2, input_dim)
    standard_bounds[1] = 1.0

    def iter_func(iter_i):
        set_seed(SEED + iter_i + 1)

        logging.info(f"Iteration {iter_i} started.")

        # GP Initialisation
        # This has a Gamma noise prior thus it is not noiseless.
        pretrained_model = fit_model(
            designs_tensor,
            torch.tensor(dataset.dataset.out_data),
            num_outputs=output_dim
        )
        params = {
            "length_scales": pretrained_model.covar_module.base_kernel.lengthscale,
            "output_scales": torch.sqrt(pretrained_model.covar_module.outputscale),
            "noise": torch.sqrt(noise_var * torch.ones_like(pretrained_model.likelihood.noise)),
        }

        initial_number = 1  # Initial sample count
        evaluated_designs_X = torch.empty((0, input_dim))
        evaluated_designs_Y = torch.empty((0, output_dim))

        def observe_and_update_design_lists(x):
            nonlocal evaluated_designs_X, evaluated_designs_Y
            
            y = dataset.evaluate(x)

            X_t = torch.tensor(x, dtype=torch.float64).reshape(1, input_dim)
            Y_t = torch.tensor(y, dtype=torch.float64).reshape(1, output_dim)

            evaluated_designs_X = torch.cat([evaluated_designs_X, X_t], 0)
            evaluated_designs_Y = torch.cat([evaluated_designs_Y, Y_t], 0)


        # Choose initial points and add to each objective
        initial_design_inds = np.random.choice(grid_size, initial_number, replace=False)
        initial_designs = designs[initial_design_inds]
        for k in range(initial_number):
            observe_and_update_design_lists(initial_designs[k])

        iter_results = [[] for _ in range(len(epsilons))]

        # Main loop
        sample_count = initial_number
        for eps_i, sampling_budget in enumerate(sampling_budgets[:, iter_i]):
            while sample_count < sampling_budget:
                logging.info(f"Sampling round {sample_count}.")
                sample_count += 1
                best_design = bo_loop(
                    params=params,
                    designs=designs_tensor,
                    train_X=normalize(evaluated_designs_X, bounds),
                    train_Y=evaluated_designs_Y,
                    num_outputs=output_dim,
                    bounds=standard_bounds,
                    acquisition_type=acq_type,
                    num_pareto_samples=3,
                    num_pareto_points=10,
                    num_greedy=10,
                    num_samples=128,
                    num_restarts=10,
                    raw_samples=1000,
                    batch_size=1
                )
                best_design = unnormalize(best_design, bounds).squeeze().cpu().numpy()
                observe_and_update_design_lists(best_design)

            logging.info("Finished a run.")
            # mu_hat
            model = fit_model(
                normalize(torch.tensor(evaluated_designs_X), bounds),
                torch.tensor(evaluated_designs_Y),
                num_outputs=output_dim,
                params=params
            )
            model.eval()
            with torch.no_grad():
                result = model(designs_tensor).mean.squeeze().cpu().T.numpy()

            # p_hat
            indices_of_returned_pareto = get_pareto_set(result, W, alpha_vec)
            iter_results[eps_i] = [sampling_budget, list(indices_of_returned_pareto)]

        return iter_results

    with Pool(max_workers=dataset_worker[1]) as pool:
        results = pool.map(
            iter_func,
            range(iteration),
        )
    results = list(results)
    
    # results = []
    # for it in range(iteration):
    #     results.append(iter_func(it))

    for eps_i, eps in enumerate(epsilons):
        iter_res_for_eps = [iter_results[eps_i] for iter_results in results]

        experiment_res_dict = {
            "dataset_name": dataset_name,
            "cone_degree": 90,
            "alg": alg_name,
            "delta": -1,
            "noise_var": noise_var,
            "eps": eps,
            "disc": -1,
            "conf_contraction": -1,
            "batch_size": -1,
            "results": iter_res_for_eps
        }
        save_new_result(output_folder_path, experiment_res_dict)
