import os
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product
from multiprocessing import set_start_method

import numpy as np
import oyaml as yaml

from utils.seed import SEED
from utils.config import Config
from utils.dataset import DATASET_SIZES
from utils.utils import set_seed, overwrite_makedirs, read_sorted_results

from algorithms.JESMO.jesmo import jesmo_run
from algorithms.MESMO.mesmo import mesmo_run
from algorithms.naive_alg import naive_with_sample_count
from algorithms.adapt_el_wrap import adaptive_elimination
from algorithms.ADELGP.adelgp import run_adelgp


def sample_means_to_compare(compare_experiment_id):
    os.listdir(experiment_folder)
    compare_experiment_name = [
        exp_name
        for exp_name in os.listdir(experiment_folder)
        if f'exp{compare_experiment_id}-' in exp_name
    ][0]
    compare_results_list = read_sorted_results(
        os.path.join(experiment_folder, compare_experiment_name), sort=False
    )
    sample_counts = []
    normalizations = []
    for exp_dict in compare_results_list:
        sample_counts.append(list(zip(*exp_dict["results"]))[0])
        normalizations.append(DATASET_SIZES[exp_dict["dataset_name"]])
    return sample_counts, normalizations

def parse_and_run_experiment(exp_id, exp_d):
    start = time.time()

    config = Config(exp_d)

    datasets = config.datasets_and_workers
    output_folder_path = os.path.join(experiment_folder, f'exp{exp_id}-' + config.algorithm)
    if config.algorithm == "Naive":
        overwrite_makedirs(output_folder_path)

        datasets = [dataset[0] for dataset in datasets]

        if config.Naive.samples is not None:
            samples = config.Naive.samples
        else:
            # Get samples from compared experiment
            sample_counts, normalizations = sample_means_to_compare(
                config.Naive.compare_experiment_id
            )
            sample_means = [np.mean(exp_sample_counts) for exp_sample_counts in sample_counts]
            sample_means = np.array(sample_means)
            normalizations = np.array(normalizations)
            sample_means = (np.ceil(sample_means / normalizations) * normalizations).astype(int)
            samples = np.array(sample_means).reshape(len(datasets), len(config.epsilons), -1)
            samples = samples.transpose((0, 2, 1)).reshape(-1, len(config.epsilons))

        # Shape: (len_configurations, len_epsilons, 2(sample size, epsilon))
        samples = np.array(samples).T
        samples_with_eps = np.hstack(
            np.split(
                np.concatenate(
                    (
                        samples.reshape(-1, 1),
                        np.repeat(config.epsilons, samples.shape[1]).reshape(-1, 1)
                    ), axis=1
                ), 2
            )
        ).reshape(-1, len(config.epsilons), 2)

        dset_and_angle = list(product(datasets, config.cone_degrees))

        assert(len(dset_and_angle) == len(samples_with_eps))

        for ((dataset_name, cone_angle), sample_with_eps) in zip(dset_and_angle, samples_with_eps):
            naive_with_sample_count(
                dataset_name, cone_angle, config.noise_var, config.delta, sample_with_eps,
                config.iteration, output_folder_path
            )
    elif config.algorithm in ["MESMO", "JESMO"]:
        overwrite_makedirs(output_folder_path)

        alg_attribute = getattr(config, config.algorithm)

        if alg_attribute.samples is not None:
            samples = alg_attribute.samples
        else:
            # Get samples from compared experiment
            sample_counts, _ = sample_means_to_compare(alg_attribute.compare_experiment_id)
            samples = np.array(sample_counts).reshape(len(datasets), -1, config.iteration)
        
        # Check that it is monotonically non-decreasing, convert it if it's not;
        samples_nondec = np.maximum.accumulate(samples, axis=1)
        # and since the algorithm does not depend on epsilon, run continuously at once.
        for dset_i in range(len(datasets)):
            if config.algorithm == "JESMO":
                jesmo_run(
                    config.algorithm, datasets[dset_i], samples_nondec[dset_i],
                    config.noise_var, config.epsilons, config.iteration, output_folder_path
                )
            elif config.algorithm == "MESMO":
                mesmo_run(
                    datasets[dset_i], samples_nondec[dset_i], config.noise_var,
                    config.epsilons, config.iteration, output_folder_path
                )
            else:
                raise RuntimeError
    elif config.algorithm == "ADELGP":
        overwrite_makedirs(output_folder_path)

        run_adelgp(
            datasets, config.cone_degrees, config.noise_var, config.delta,
            config.epsilons, config.iteration, output_folder_path
        )
    else:
        if config.algorithm == "Auer":
            alg_id = 4
            gp_dict = None
        else:
            alg_id = 3
            gp_dict = config.PaVeBa.GP
            if gp_dict.use_gp:
                suffix = '_'
                suffix += 'I' if gp_dict.independent else 'D'
                suffix += 'H' if not gp_dict.ellipsoid else 'E'
                output_folder_path = os.path.join(experiment_folder, f'exp{exp_id}-' + config.algorithm + suffix)
            gp_dict = gp_dict.dict

        overwrite_makedirs(output_folder_path)

        adaptive_elimination(
            alg_id, gp_dict, datasets, config.cone_degrees, config.noise_var, config.delta,
            config.epsilons, config.iteration, config.conf_contractions, output_folder_path
        )

    end = time.time()

    with open(os.path.join(experiment_folder, "times.txt"), 'a') as f:
        print(f"Experiment ID={exp_id} done in {end - start:.2f} seconds.", file=f)


if __name__ == "__main__":
    # set_start_method("spawn")

    # Disable warnings, especially for CVXPY
    import warnings
    warnings.filterwarnings("ignore")

    # Set up logging level
    logging.basicConfig(level=logging.INFO)

    # Set seed
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', type=Path, required=True)
    args = parser.parse_args()

    # Read experiment config
    experiment_file = args.experiment_file  # os.path.join("experiments", "experiment_jesmo.yaml")
    with open(experiment_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Continue experiment
    if config["experiment_name"] != "":
        if config["experiment_ids"] == 1:
            raise Exception("Check start ID, it overwrites whole experiment.")
        experiment_name = config["experiment_name"]
    else:  # New experiment
        experiment_name = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    experiment_folder = os.path.join("outputs", experiment_name)

    # Copy experiment config if does not exists
    os.makedirs(experiment_folder, exist_ok=True)
    copy_experiment_file = os.path.join(experiment_folder, "experiment.yaml")
    if not os.path.exists(copy_experiment_file):
        shutil.copy(src=experiment_file, dst=copy_experiment_file)

    # Which experiments to run
    experiment_ids = config["experiment_ids"]
    if isinstance(experiment_ids, int):
        experiment_ids = range(experiment_ids, config["num_experiments"]+1)

    # Run experiments
    for i in experiment_ids:
        parse_and_run_experiment(i, config[f"experiment{i}"])
