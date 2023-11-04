import os
import pickle
from glob import glob
from functools import partial

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

import utils.dataset
from utils.utils import get_cone_params, get_uncovered_set, read_sorted_results


def uncover_with_cache(p_opt_miss, p_opt_hat, mu, eps, W, dataset_name, cone_degree):
    """
    Wrapper for get_uncovered_set method.
    Caches the calculations to improve return time.
    """
    uncover_cache_path = os.path.join("outputs", "uncovered_cache.pkl")

    if not os.path.exists(uncover_cache_path):
        cache_dict = {}
    else:
        with open(uncover_cache_path, "rb") as f:
            cache_dict = pickle.load(f)

    key = (dataset_name, cone_degree, eps, tuple(p_opt_hat))
    cache_result = cache_dict.get(key)

    if cache_result is not None:
        return cache_result

    result = get_uncovered_set(p_opt_miss, p_opt_hat, mu, eps, W)
    cache_dict[key] = result

    with open(uncover_cache_path, "wb") as f:
        pickle.dump(cache_dict, f)
    
    return result


def evaluate_experiment(exp_dict):
    dataset_name = exp_dict["dataset_name"]
    cone_degree = exp_dict["cone_degree"]
    cover_eps = exp_dict["eps"]

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)
    delta_cone, true_pareto_indices = dataset.get_params()

    W_CONE, _, _ = get_cone_params(cone_degree, dim=dataset.out_dim)

    hypervolume_instance = Hypervolume(torch.tensor([-10.0] * dataset.out_dim))

    result_keys = ['F1T', 'F1E', 'SC'] # ['F1T', 'F1E', 'HV', 'SC']  # 'ER', 'NF1', 'NF2', 
    result_sum = np.zeros((len(exp_dict["results"]), len(result_keys)))
    for res_i, (samples, pred_pareto_indices) in enumerate(exp_dict["results"]):
        pred_set = set(pred_pareto_indices)
        gt_set = set(true_pareto_indices)

        indices_of_missed_pareto = list(gt_set - pred_set)

        hypervol = hypervolume_instance.compute(torch.tensor(dataset.out_data[pred_pareto_indices]))

        # Returns non-covered pareto indices that are missed
        uncovered_missed_pareto_indices = get_uncovered_set(
            indices_of_missed_pareto, pred_pareto_indices, dataset.out_data, cover_eps, W_CONE
        )
        # uncovered_missed_pareto_indices = uncover_with_cache(
        #     indices_of_missed_pareto, pred_pareto_indices, dataset.out_data, cover_eps, W_CONE,
        #     dataset_name, cone_degree
        # )

        nf1_eps = len(uncovered_missed_pareto_indices)
        nf2_eps = np.sum(delta_cone[pred_pareto_indices] > cover_eps, axis=0)[0]

        true_eps = np.sum(delta_cone[pred_pareto_indices] <= cover_eps, axis=0)[0]
        
        successful_eps = (nf1_eps == 0) and (nf2_eps == 0)

        tp_true = len(gt_set) - len(indices_of_missed_pareto)
        fp_true = len(pred_set) - tp_true
        f1_true = (2 * tp_true) / (2*tp_true + fp_true + len(indices_of_missed_pareto))

        tp_eps = true_eps
        fp_eps = len(pred_set) - true_eps
        f1_eps = (2 * tp_eps) / (2*tp_eps + fp_eps + len(uncovered_missed_pareto_indices))

        # ['ER', 'NF1', 'NF2', 'SR', 'F1', 'HV', 'SC']
        result_sum[res_i] = [
            # true_eps / len(pred_set) * 100,
            # nf1_eps / len(gt_set) * 100,
            # nf2_eps / len(pred_set) * 100,
            # successful_eps * 100,
            f1_true,
            f1_eps,
            # hypervol,
            samples
        ]

    result = result_sum.mean(axis=0)
    result_std = result_sum.std(axis=0)

    result_dict = dict(zip(result_keys, np.around(result, 2).tolist()))
    result_std_dict = dict(zip(
        list(map(lambda x: x+' Std', result_keys)),
        np.around(result_std, 2).tolist()
    ))

    return result_dict, result_std_dict


if __name__ == "__main__":
    exp_path = None
    
    # Folder path of the experiment to evaluate
    exp_path = os.path.join("outputs", "10_14_2023__13_18_03")
    # exp_path = os.path.join("outputs", "10_23_2023__03_50_19")
    # exp_path = os.path.join("outputs", "10_16_2023__19_54_07")
    # exp_path = os.path.join("outputs", "10_15_2023__12_09_06_comb")
    # exp_path = os.path.join("outputs", "10_15_2023__12_09_06")
    # exp_path = os.path.join("outputs", "07_27_2023__01_35_54_onetwo")
    
    # If no path is given, just evaluate the last experiment
    if exp_path is None:
        exp_path = sorted([
            subpath
            for subpath in glob(os.path.join("outputs", "*"))
            if os.path.isdir(subpath)
        ])[-1]

    algorithm_names = sorted(
        [
            subpath
            for subpath in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, subpath))
        ]
    )
    for alg_name in algorithm_names:
        alg_text = alg_name.split('-')[-1]

        # if alg_text not in ["MESMO", "JESMO"]:
        #     continue

        # Load results file
        alg_path = os.path.join(exp_path, alg_name)
        results_list = read_sorted_results(alg_path)

        print(
            "---   "
            f"Algorithm: {alg_text}"
            f", Iteration count: {len(results_list[0]['results'])}"
            "   ---"
        )

        # Evaluate each config
        for exp_dict in results_list:
            result, result_std = evaluate_experiment(exp_dict)
            
            for (k, v), std_v in zip(result.items(), result_std.values()):
                result[k] = f"{v:06.2f} ± {std_v:05.2f}"

            print(
                f"D.set: {exp_dict['dataset_name']:<16}"
                f"Cone: {exp_dict['cone_degree']:<4}"
                f"Eps.: {exp_dict['eps']:<6}",
                f"Cont.: {exp_dict['conf_contraction']:<4}",
                f"B.S.: {exp_dict['batch_size']:<4}",
                result
            )
        print()
