import logging

from vectoptal.utils import set_seed
from vectoptal.utils.seed import SEED
from vectoptal.algorithms import VOGP, PaVeBa, PaVeBaGP, NaiveElimination
from vectoptal.datasets.dataset import DiskBrake, SNW
from vectoptal.order import ConeTheta2DOrder


if __name__ == "__main__":
    # Set up logging level
    logging.basicConfig(level=logging.INFO)

    # Set seed
    set_seed(SEED)

    order = ConeTheta2DOrder(cone_degree=135)

    # algorithm = VOGP(
    #     epsilon=0.01, delta=0.1,
    #     dataset_name="DiskBrake", order=order, noise_var=0.01,
    #     conf_contraction=16
    # )
    # algorithm = PaVeBa(
    #     epsilon=0.01, delta=0.1,
    #     dataset_name="DiskBrake", order=order, noise_var=0.0001,
    #     conf_contraction=16
    # )
    # algorithm = PaVeBaGP(
    #     epsilon=0.01, delta=0.1,
    #     dataset_name="DiskBrake", order=order, noise_var=0.0001,
    #     conf_contraction=64, type="DE"
    # )
    algorithm = NaiveElimination(
        epsilon=0.01, delta=0.05,
        dataset_name="SNW", order=order, noise_var=0.01,
        L=500
    )

    while True:
        is_done = algorithm.run_one_step()

        if is_done:
            break

    logging.info("Done!")

    logging.info(f"Found Pareto front indices are: {str(sorted(algorithm.P))}")
    
    # dataset = DiskBrake()
    dataset = SNW()
    pareto_indices = order.get_pareto_set(dataset.out_data)
    # figure = order.plot(path="try.png")
    logging.info(f"True Pareto front indices are: {str(sorted(set(pareto_indices)))}")
