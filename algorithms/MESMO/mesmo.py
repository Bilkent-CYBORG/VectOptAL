import logging
from concurrent.futures import ProcessPoolExecutor as Pool

import numpy as np
from platypus import NSGAII, Problem, Real

import utils.dataset
from utils.seed import SEED
from utils.utils import set_seed, get_pareto_set, get_cone_params, save_new_result
from models.mesmo import GaussianProcess
from algorithms.MESMO.singlemes import MaxvalueEntropySearch
from algorithms.MESMO.mesmo_dataset_wrapper import MESMODatasetWrapper


platypus_logger = logging.getLogger("Platypus")
platypus_logger.setLevel(logging.ERROR)


def get_observation_and_update_gp(dataset, models, x):
    y = dataset.evaluate(x)
    for i in range(len(models)):
        models[i].addSample(x, y[i])

def mesmo_run(
    dataset_worker, sampling_budgets, noise_var, epsilons, iteration, output_folder_path
):
    global iter_func  # !!! NASTY SOLUTION !!!

    set_seed(SEED)

    dataset_name = dataset_worker[0]
    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(90)
    dataset = MESMODatasetWrapper(dataset, noise_var)

    W, alpha_vec, _ = get_cone_params(90, dim=dataset.out_dim)

    input_dim = dataset.in_dim
    output_dim = dataset.out_dim
    designs = dataset.designs
    grid_size = len(dataset.designs)

    function_bounds = dataset.bounds
    bound = function_bounds[0]

    initial_number = 1  # Initial sample count
    sample_number = 1  # ASSUME: Number of cheap samplings

    def iter_func(iter_i):
        set_seed(SEED + iter_i + 1)

        logging.info(f"Iteration {iter_i} started.")
        # GP Initialisation
        GPs = []
        Multiplemes = []

        # Generate GP for each objective
        for i in range(output_dim):
            GPs.append(GaussianProcess(input_dim, noise_var))

        # Add all points and fit the model
        for k in range(grid_size):
            get_observation_and_update_gp(dataset, GPs, designs[k])
        for i in range(output_dim):
            GPs[i].fitModel()
            GPs[i].clear_data()

        # Choose initial points and add to each objective
        initial_design_inds = np.random.choice(grid_size, initial_number, replace=False)
        initial_designs = designs[initial_design_inds]
        for k in range(initial_number):
            get_observation_and_update_gp(dataset, GPs, initial_designs[k])

        for i in range(output_dim):
            GPs[i].fitModel()
            Multiplemes.append(MaxvalueEntropySearch(GPs[i]))

        iter_results = [[] for _ in range(len(epsilons))]

        # Main loop
        sample_count = initial_number
        for eps_i, sampling_budget in enumerate(sampling_budgets[:, iter_i]):
            while sample_count < sampling_budget:
                logging.info(f"Sampling round {sample_count}.")
                sample_count += 1

                for i in range(output_dim):
                    Multiplemes[i] = MaxvalueEntropySearch(GPs[i])
                    Multiplemes[i].Sampling_RFM()
                
                # max_samples = np.empty((sample_number, output_dim))
                # for i in range(output_dim):
                #     Ys = GPs[i].getSample(designs, sample_number)
                #     max_samples[:, i] = np.max(Ys, axis=0)

                max_samples = []
                for _ in range(sample_number):
                    for i in range(output_dim):
                        Multiplemes[i].weigh_sampling()
                
                    cheap_pareto_front = []
                    def CMO(xi):
                        xi = np.asarray(xi)
                        y = [Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
                        return y
                    
                    problem = Problem(input_dim, output_dim)
                    problem.types[:] = Real(bound[0], bound[1])
                    problem.function = CMO
                    algorithm = NSGAII(problem)
                    algorithm.run(1500)
                    cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
                    
                    # Picking the max over the pareto: best case
                    maxoffunctions = [-1*min(f) for f in list(zip(*cheap_pareto_front))]
                    max_samples.append(maxoffunctions)

                def mesmo_acq(x):
                    multi_obj_acq_total = 0
                    for j in range(sample_number):
                        multi_obj_acq_sample = sum(
                            Multiplemes[i].single_acq(x, max_samples[j][i]) for i in range(output_dim)
                        )
                        multi_obj_acq_total += multi_obj_acq_sample
                    return multi_obj_acq_total / sample_number

                # Choose best point
                acq_designs = [mesmo_acq(x) for x in designs]
                sorted_indices = np.argsort(acq_designs)
                best_design = designs[sorted_indices[0]]

                # Updating and fitting the GPs
                get_observation_and_update_gp(dataset, GPs, best_design)
                for i in range(output_dim):
                    GPs[i].fitModel()

            logging.info("Finished a run.")
            # mu_hat
            result = np.empty((len(designs), len(GPs)))
            for d_i, design in enumerate(designs):
                for i in range(len(GPs)):
                    mu, _ = GPs[i].getPrediction(design)
                    result[d_i, i] = mu

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

    for eps_i, eps in enumerate(epsilons):
        iter_res_for_eps = [iter_results[eps_i] for iter_results in results]

        experiment_res_dict = {
            "dataset_name": dataset_name,
            "cone_degree": 90,
            "alg": "MESMO",
            "delta": -1,
            "noise_var": noise_var,
            "eps": eps,
            "disc": -1,
            "conf_contraction": -1,
            "batch_size": -1,
            "results": iter_res_for_eps
        }
        save_new_result(output_folder_path, experiment_res_dict)
