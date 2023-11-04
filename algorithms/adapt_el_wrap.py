import os
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor as Pool

import numpy as np

from algorithms import adapt_el_algorithm
from algorithms import adapt_el_algorithm_gp
import utils.dataset
from utils.seed import SEED
from utils.utils import set_seed, plot_pareto_front, get_cone_params, save_new_result


def simulate_once(
    i, gp_dict,
    dataset_name, cone_degree,
    alg_id, delt, noise, eps, conf_contraction,
    batch_size, disc=50, plot=False
):
    set_seed(SEED + i + 1)

    use_gp = False if gp_dict is None or not gp_dict["use_gp"] else True
    alg_module = adapt_el_algorithm_gp if use_gp else adapt_el_algorithm

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)
    delta_cone, t_true_i_cone = dataset.get_params()

    W_CONE, alpha_vec, cone_text = get_cone_params(cone_degree, dataset.out_dim)


    # Run the algorithm and return the pareto set and the sample count
    if alg_id == 1:
        alg = alg_module.Algorithm1(
            *dataset.in_data.shape, 2, W_CONE, noise, delt, dataset.model_kernel,
            batch_size
        )
    elif alg_id == 2:
        alg = alg_module.Algorithm2(
            *dataset.in_data.shape, 2, W_CONE, noise, delt, dataset.model_kernel,
            batch_size
        )
    elif alg_id == 3 and not use_gp:
        alg = alg_module.Algorithm3(
            *dataset.in_data.shape, 2, W_CONE, noise, delt, dataset.model_kernel, eps, disc,
            conf_contraction, batch_size
        )
    elif alg_id == 3 and use_gp:
        alg = alg_module.Algorithm3(
            *dataset.in_data.shape, dataset.out_dim, W_CONE, noise, delt, dataset.model_kernel, eps, disc,
            gp_dict["independent"], conf_contraction, batch_size, use_ellipsoid=gp_dict["ellipsoid"]
        )
    elif alg_id == 4:
        alg = alg_module.AlgorithmAuer(
            *dataset.in_data.shape, 2, W_CONE, noise, delt, dataset.model_kernel, eps,
            conf_contraction, batch_size
        )

    alg.prepare(dataset.in_data, dataset.out_data)

    pred_indices, samples = alg.run()
    print(f"DONE {i}.")

    if plot and i == 0:
        figure_name = dataset_name + '_' + '_'.join(
            f'{k}={v}' for k, v in [
                ("cone", cone_degree), ("alg", alg_id), ("delt", delt),
                ("noise", noise), ("eps", eps), ("disc", disc)
            ]
        ) + '.png'
        
        n = len(dataset.in_data)
        title = (
            r"$n = "
            + str(n)
            + r" \;|\; G\ddot{o}zlemler = "
            + str(samples)
            + r" \;|\; Veri\; k\ddot{u}mesi = "
            + str(dataset_name)
            + r" \;|\; $"
        )

        alg_id_text = f"UE{alg_id}" if alg_id != 4 else "Auer"
        alg_text = rf'$ \;|\; {alg_id_text}$'
        title += cone_text + alg_text

        pred_pareto_y = dataset.out_data[pred_indices]
        pareto_mask = np.zeros(n, dtype=bool)
        pareto_mask[t_true_i_cone] = True
        plot_pareto_front(
            dataset.out_data[:, 0], dataset.out_data[:, 1], pareto_mask,
            y1=pred_pareto_y[:, 0], y2=pred_pareto_y[:, 1],
            plotfront=False,
            title=title, f1label=dataset.f1label, f2label=dataset.f2label,
            save_path=os.path.join("plots", figure_name)
        )

    return [samples, list(pred_indices)]


def adaptive_elimination(
    alg_id, gp_dict, datasets_and_workers, cone_degrees, noise_var, delta, epsilons, iteration,
    conf_contractions, output_folder_path
):
    batch_sizes = gp_dict.get("batch_sizes", [None]) if gp_dict else [None]
    
    # dset, eps, cone, conf, batch
    for dataset_name, dataset_worker in datasets_and_workers:
        for eps in epsilons:
            alg_independent_params = product(cone_degrees, conf_contractions)

            for cone_degree, conf_contraction in alg_independent_params:
                for batch_size in batch_sizes:
                    simulate_part = partial(
                        simulate_once,
                        dataset_name=dataset_name,
                        gp_dict=gp_dict,
                        cone_degree=cone_degree,
                        alg_id=alg_id,
                        delt=delta,
                        noise=noise_var,
                        eps=eps,
                        conf_contraction=conf_contraction,
                        batch_size=batch_size
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
                        "alg": alg_id,
                        "delta": delta,
                        "noise_var": noise_var,
                        "eps": eps,
                        "disc": -1,
                        "conf_contraction": conf_contraction,
                        "batch_size": batch_size if batch_size else -1,
                        "results": results
                    }

                    save_new_result(output_folder_path, experiment_res_dict)
