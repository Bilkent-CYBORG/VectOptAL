from vectoptal.utils.seed import SEED
from vectoptal.datasets.dataset import *
from vectoptal.order import ConeTheta2DOrder
from vectoptal.algorithms import NaiveElimination
from vectoptal.utils import set_seed, calculate_epsilonF1_score


if __name__ == "__main__":
    # Set seed
    set_seed(SEED)

    order = ConeTheta2DOrder(cone_degree=135)
    dataset_name = "DiskBrake"
    dataset = globals()[dataset_name]()

    epsilon = 0.01
    delta = 0.05
    noise_var = epsilon

    algorithm = NaiveElimination(
        epsilon=epsilon, delta=delta,
        dataset_name=dataset_name, order=order, noise_var=noise_var,
        # L=500
    )

    while True:
        is_done = algorithm.run_one_step()

        if is_done:
            break

    print("Done!")

    print(f"Found Pareto front indices are: {str(sorted(algorithm.P))}")

    pareto_indices = order.get_pareto_set(dataset.out_data)
    print(f"True Pareto front indices are: {str(sorted(set(pareto_indices)))}")

    eps_f1 = calculate_epsilonF1_score(dataset, order, pareto_indices, list(algorithm.P), epsilon)
    print(f"epsilon-F1 Score is: {eps_f1:.2f}")
