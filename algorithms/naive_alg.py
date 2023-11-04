import logging

import numpy as np

import utils.dataset
from utils.utils import get_cone_params, get_pareto_set, save_new_result

from utils.seed import SEED


def naive_with_sample_count(
    dataset_name, cone_angle, noise_var, delta, sample_eps, nrun, output_folder_path
):
    W, alpha_vec, _ = get_cone_params(cone_angle)
    
    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_angle)

    mu = dataset.out_data
    
    D = mu.shape[1]
    K = mu.shape[0]

    sigma = np.sqrt(noise_var)

    # Simulation model
    smp_cnt = sample_eps[:, 0]
    eps_vals = sample_eps[:, 1]

    np.random.seed(SEED)

    eval_num = np.ceil(smp_cnt / len(mu)).astype(int)
    logging.debug(eval_num * len(mu))

    nsample = eval_num.shape[0]
    noisemat = sigma * np.random.randn(nrun, nsample, K, D)

    # Compute epsilon independent results

    noisemat_interval = noisemat * np.sqrt(
        np.hstack((eval_num[0], np.diff(eval_num)))
    )[np.newaxis,:,np.newaxis,np.newaxis]
    noisemat_avg = np.cumsum(noisemat_interval,axis=1)/eval_num[np.newaxis,:,np.newaxis,np.newaxis]

    mu_hat = mu[np.newaxis,np.newaxis,:,:] + noisemat_avg

    for j in range(eval_num.shape[0]):
        logging.info("New evaluation.")
        results = []
        for i in range(nrun):
            logging.debug(i)
            p_opt_45_hat = get_pareto_set(mu_hat[i,j,:,:], W, alpha_vec)
            results.append([eval_num[j] * len(mu), list(p_opt_45_hat)])
        logging.debug("")

        experiment_res_dict = {
            "dataset_name": dataset_name,
            "cone_degree": cone_angle,
            "alg": "Naive",
            "delta": delta,
            "noise_var": sigma*sigma,
            "eps": eps_vals[j],
            "disc": -1,
            "conf_contraction": -1,
            "batch_size": -1,
            "results": results
        }

        save_new_result(output_folder_path, experiment_res_dict)