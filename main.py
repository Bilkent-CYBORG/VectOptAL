from vectoptal.utils.seed import SEED
from vectoptal.datasets.dataset import *
from vectoptal.maximization_problem import *
from vectoptal.order import ConeTheta2DOrder
from vectoptal.algorithms import NaiveElimination, PaVeBa, PaVeBaGP, VOGP, VOGP_AD
from vectoptal.utils import set_seed
from vectoptal.utils.evaluate import (
    calculate_epsilonF1_score, calculate_hypervolume_discrepancy_for_model
)


def test_discrete():
    set_seed(SEED)

    order = ConeTheta2DOrder(cone_degree=135)
    dataset_name = "DiskBrake"
    dataset = globals()[dataset_name]()

    epsilon = 0.01
    delta = 0.05
    noise_var = epsilon

    iter_count = 10
    eps_f1_values = []
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = VOGP(
            epsilon=epsilon, delta=delta,
            dataset_name=dataset_name, order=order, noise_var=noise_var,
            conf_contraction=16,
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break

        pareto_indices = order.get_pareto_set(dataset.out_data)
        eps_f1 = calculate_epsilonF1_score(dataset, order, pareto_indices, list(algorithm.P), epsilon)
        eps_f1_values.append(eps_f1)

    print(
        f"Avg. eps-F1 score over {iter_count} iterations:"
        f" {sum(eps_f1_values)/len(eps_f1_values):.2f}"
    )

def test_continuous():
    set_seed(SEED)

    order = ConeTheta2DOrder(cone_degree=90)

    epsilon = 0.1
    delta = 0.05
    noise_var = epsilon

    problem_name = "BraninCurrin"
    problem: ContinuousProblem = globals()[problem_name](noise_var=noise_var)

    iter_count = 1
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = VOGP_AD(
            epsilon=epsilon, delta=delta,
            problem=problem, order=order, noise_var=noise_var,
            conf_contraction=16,
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break
        
        log_hv_discrepancy = calculate_hypervolume_discrepancy_for_model(
            order, problem, algorithm.model
        )
        print(f"Logarithm HV discrepancy is: {log_hv_discrepancy:.2f}")

if __name__ == "__main__":
    # test_discrete()
    test_continuous()
