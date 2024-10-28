import logging

import numpy as np

from vectoptal.utils.seed import SEED
from vectoptal.datasets.dataset import Dataset, get_dataset_instance
from vectoptal.maximization_problem import ContinuousProblem, get_continuous_problem
from vectoptal.order import ConeTheta2DOrder, ComponentwiseOrder
from vectoptal.algorithms import (  # noqa: F401
    NaiveElimination,
    Auer,
    PaVeBa,
    PaVeBaGP,
    PaVeBaPartialGP,
    DecoupledGP,
    EpsilonPAL,
    VOGP,
    VOGP_AD,
)
from vectoptal.utils import set_seed
from vectoptal.utils.evaluate import (
    calculate_epsilonF1_score,
    calculate_hypervolume_discrepancy_for_model,
)
from vectoptal.models import GPyTorchModelListExactModel


def test_discrete():
    set_seed(SEED)

    # order = ConeTheta2DOrder(cone_degree=90)
    order = ComponentwiseOrder(2)
    dataset_name = "DiskBrake"
    dataset = get_dataset_instance(dataset_name)

    epsilon = 0.1
    delta = 0.1
    noise_var = 0.00001

    iter_count = 1
    eps_f1_values = []
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = PaVeBa(
            epsilon=epsilon,
            delta=delta,
            dataset_name=dataset_name,
            order=order,
            noise_var=noise_var,
            conf_contraction=16,
            # type="IH",
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break

        pareto_indices = order.get_pareto_set(dataset.out_data)
        eps_f1 = calculate_epsilonF1_score(
            dataset, order, pareto_indices, list(algorithm.P), epsilon
        )
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
    problem: ContinuousProblem = get_continuous_problem(name=problem_name, noise_var=noise_var)

    iter_count = 1
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = VOGP_AD(
            epsilon=epsilon,
            delta=delta,
            problem=problem,
            order=order,
            noise_var=noise_var,
            conf_contraction=32,
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break

        log_hv_discrepancy = calculate_hypervolume_discrepancy_for_model(
            order, problem, algorithm.model
        )
        print(f"Logarithm HV discrepancy is: {log_hv_discrepancy:.2f}")


def test_partial_model():
    # train_x = np.array([[0,0], [10,1], [1,0], [0,0]])
    # train_y = np.array(([[4,5], [20,-20], [3,4], [5,4]]))

    train_x = np.array([[0, 0], [1, 0], [0, 1]])
    # train_y = np.array(([[3], [20], [2.5]]))
    train_y = np.array(([[3, 3], [20, 3.5], [2.5, 3.5]]))

    model = GPyTorchModelListExactModel(2, 2, 0.1)

    def print_model_params():
        for model_i, m in enumerate(model.model.models):
            print(f"Model: {model_i}, Parameter: Noise, " f"value = {m.likelihood.noise.tolist()}")
            print(
                f"Model: {model_i}, Parameter: Mean, " f"value = {m.mean_module.constant.tolist()}"
            )
            print(
                f"Model: {model_i}, Parameter: Variance, "
                f"value = {m.covar_module.outputscale.tolist()}"
            )
            print(
                f"Model: {model_i}, Parameter: Lengthscale, "
                f"value = {m.covar_module.base_kernel.lengthscale.tolist()}"
            )

    model.add_sample(train_x, train_y[:, 0], dim_index=0)
    model.add_sample(train_x, train_y[:, 1], dim_index=1)

    model.update()

    print(model.predict(train_x)[0])
    print(model.predict(train_x)[1])

    print_model_params()

    model.train()

    print_model_params()

    # model.clear_data()
    # model.update()
    # print(model.model.models[0].eval())
    print(model.model.models[1].training)
    print(model.predict(train_x)[0])
    print(model.predict(train_x)[1])

    return

    model.add_sample(train_x, np.array([[4], [20], [3], [5]]), dim_index=0)
    model.add_sample(train_x, np.array([[5], [-20], [4], [4]]), dim_index=1)

    model.update()

    print(model.predict(train_x))
    print(model.sample_from_posterior(train_x))
    print(model.sample_from_posterior(train_x).shape)

    # SNW
    # datafile = 'sort_256.csv'
    # designs = np.genfromtxt(datafile, delimiter=';')
    # out_snw = np.copy(designs[:,3:])
    # out_snw[:,0] = -out_snw[:,0]
    # in_snw = np.copy(designs[:,:3])

    # kernel = gpytorch.kernels.MultitaskKernel(
    #         gpytorch.kernels.RBFKernel, num_tasks=2, rank=2
    #     )

    # kernel = gpytorch.kernels.RBFKernel

    # kernel2= gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    # a = PartialGPModel(3,2,0.001,kernel)
    # a.add_sample(in_snw, out_snw[:,0], 0)
    # a.add_sample(in_snw, out_snw[:,1], 1)
    # a.update(0)
    # a.update(1)
    # print(a.predict(in_snw))


def test_partial():
    set_seed(SEED)

    order = ConeTheta2DOrder(cone_degree=85)
    dataset_name = "DiskBrake"
    dataset = get_dataset_instance(dataset_name)

    epsilon = 0.1
    delta = 0.1
    noise_var = 0.0001

    iter_count = 1
    eps_f1_values = []
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = PaVeBaPartialGP(
            epsilon=epsilon,
            delta=delta,
            dataset_name=dataset_name,
            order=order,
            noise_var=noise_var,
            conf_contraction=32,
            costs=[1.0, 1.2],
            cost_budget=None,
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break

        pareto_indices = order.get_pareto_set(dataset.out_data)
        eps_f1 = calculate_epsilonF1_score(
            dataset, order, pareto_indices, list(algorithm.P), epsilon
        )
        eps_f1_values.append(eps_f1)

    print(
        f"Avg. eps-F1 score over {iter_count} iterations:"
        f" {sum(eps_f1_values)/len(eps_f1_values):.2f}"
    )


def test_partial_fixed_budget():
    set_seed(SEED)

    order = ConeTheta2DOrder(cone_degree=85)
    dataset_name = "DiskBrake"
    dataset = get_dataset_instance(dataset_name)

    epsilon = 0.1
    # delta = 0.1
    noise_var = 0.0001

    iter_count = 5
    eps_f1_values = []
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = DecoupledGP(
            dataset_name=dataset_name,
            order=order,
            noise_var=noise_var,
            costs=[1.0, 1.2],
            cost_budget=50,
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break

        pareto_indices = order.get_pareto_set(dataset.out_data)
        eps_f1 = calculate_epsilonF1_score(
            dataset, order, pareto_indices, list(algorithm.P), epsilon
        )
        eps_f1_values.append(eps_f1)

    print(
        f"Avg. eps-F1 score over {iter_count} iterations:"
        f" {sum(eps_f1_values)/len(eps_f1_values):.2f}"
    )


def test_moo():
    set_seed(SEED)

    dataset_name = "DiskBrake"
    dataset: Dataset = get_dataset_instance(dataset_name)
    order = ComponentwiseOrder(dataset._out_dim)

    epsilon = 0.01
    delta = 0.05
    noise_var = epsilon

    iter_count = 5
    eps_f1_values = []
    for iter_i in range(iter_count):
        set_seed(SEED + iter_i + 1)

        algorithm = Auer(
            epsilon=epsilon,
            delta=delta,
            dataset_name=dataset_name,
            noise_var=noise_var,
            conf_contraction=128,
            use_empirical_beta=False,
        )

        while True:
            is_done = algorithm.run_one_step()

            if is_done:
                break

        pareto_indices = order.get_pareto_set(dataset.out_data)
        eps_f1 = calculate_epsilonF1_score(
            dataset, order, pareto_indices, list(algorithm.P), epsilon
        )
        eps_f1_values.append(eps_f1)

    print(
        f"Avg. eps-F1 score over {iter_count} iterations:"
        f" {sum(eps_f1_values)/len(eps_f1_values):.2f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_discrete()
    # test_continuous()
    # test_partial_model()
    # test_partial()
    # test_partial_fixed_budget()
    # test_moo()
