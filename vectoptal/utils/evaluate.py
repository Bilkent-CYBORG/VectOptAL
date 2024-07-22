import numpy as np

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

from vectoptal.order import Order
from vectoptal.models import Model
from vectoptal.datasets import Dataset
from vectoptal.maximization_problem import ContinuousProblem
from vectoptal.utils import get_uncovered_set, get_delta, generate_sobol_samples


def calculate_epsilonF1_score(
    dataset: Dataset, order: Order, true_indices: np.ndarray, pred_indices: np.ndarray,
    epsilon: float
):
    indices_of_missed_pareto = list(set(true_indices) - set(pred_indices))

    uncovered_missed_pareto_indices = get_uncovered_set(
        indices_of_missed_pareto, pred_indices, dataset.out_data, epsilon, order.ordering_cone.W
    )

    delta_values = get_delta(dataset.out_data, order.ordering_cone.W, order.ordering_cone.alpha)

    true_eps = np.sum(delta_values[np.array(list(pred_indices)).astype(int)] <= epsilon, axis=0)[0]

    tp_eps = true_eps
    fp_eps = len(pred_indices) - true_eps
    f1_eps = (2 * tp_eps) / (2*tp_eps + fp_eps + len(uncovered_missed_pareto_indices))

    return f1_eps

def calculate_hypervolume_discrepancy_for_model(
    order: Order, problem: ContinuousProblem, model: Model
):
    x = generate_sobol_samples(problem.in_dim, 2048)  # TODO: magic number

    f = problem.evaluate(x, noisy=False)
    true_pareto_indices = order.get_pareto_set(f)
    y, _ = model.predict(x)
    pred_pareto_indices = order.get_pareto_set(y)

    f_W = f @ order.ordering_cone.W.T
    reference_point = torch.tensor(np.min(f_W, axis=0))
    hypervolume_instance = Hypervolume(reference_point)

    hypervolume_true = hypervolume_instance.compute(torch.tensor(f_W[true_pareto_indices]))
    hypervolume_pred = hypervolume_instance.compute(torch.tensor(f_W[pred_pareto_indices]))

    assert hypervolume_true - hypervolume_pred > 0, "Hypervolumes are the same."
    log_hv_disc = np.log(hypervolume_true - hypervolume_pred)

    return log_hv_disc
