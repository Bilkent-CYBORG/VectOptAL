#!/usr/bin/env python3

r"""
The Bayesian optimization loop.
"""

import torch
import time
from typing import Optional
from torch import Tensor

# kernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors import GammaPrior

# fit gp
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.optim.optimize import optimize_acqf_list
from botorch.models.model import Model

# sampling
from botorch.utils.sampling import draw_sobol_samples

# jes
from algorithms.JESMO.jes.acquisition.mes import qLowerBoundMaximumEntropySearch
from algorithms.JESMO.jes.acquisition.jes import compute_box_decomposition, qLowerBoundJointEntropySearch
from algorithms.JESMO.jes.acquisition.pes import qPredictiveEntropySearch
from algorithms.JESMO.jes.utils.sample_pareto import sample_pareto_sets_and_fronts
from algorithms.JESMO.jes.utils.optim_fd import optimize_acqf_fd

# botorch
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement, qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement, qNoisyExpectedImprovement
)
from botorch.models.transforms.outcome import Standardize


def fit_model(
        train_X: Tensor,
        train_Y: Tensor,
        num_outputs: int,
        params: Optional[dict] = None,
) -> Model:
    r"""Fit the Gaussian process model.

    Args:
        train_X: A `N x d`-dim Tensor containing the training inputs.
        train_Y: A `N x M`-dim Tensor containing the training inputs.
        num_outputs: The number of objectives.
        params: A dictionary containing the GP parameters

    Returns:
        Fitted Gaussian process model.
    """

    covar_module = ScaleKernel(
        RBFKernel(
            ard_num_dims=train_X.shape[-1],
            batch_shape=torch.Size([train_Y.shape[-1]]),
            lengthscale_prior=GammaPrior(3.0, 6.0)
        ), batch_shape=torch.Size([train_Y.shape[-1]]), outputscale_prior=GammaPrior(2.0, 0.15)
    )
    model = SingleTaskGP(
        train_X, train_Y, outcome_transform=Standardize(m=num_outputs), covar_module=covar_module
    )
    if params is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
    else:
        model.covar_module.base_kernel.lengthscale = params["length_scales"]
        model.covar_module.outputscale = params["output_scales"] ** 2
        model.likelihood.noise = params["noise"] ** 2

    return model


def bo_loop(
        params: dict,
        designs: Tensor,
        train_X: Tensor,
        train_Y: Tensor,
        num_outputs: int,
        bounds: Tensor,
        acquisition_type: str = None,
        num_pareto_samples: Optional[int] = 10,
        num_pareto_points: Optional[int] = 10,
        num_greedy: Optional[int] = 10,
        num_samples: Optional[int] = 128,
        num_restarts: int = 10,
        raw_samples: int = 1000,
        batch_size: int = 1,
) -> Tensor:
    r"""Performs a single Bayesian optimization loop for a multi-objective problem.

    Args:
        train_X: A `N x d`-dim Tensor containing the training inputs.
        train_Y: A `N x M`-dim Tensor containing the training inputs.
        num_outputs: The number of objectives.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        acquisition_type: The acquisition type.
        num_pareto_samples: The number of Thompson samples.
        num_pareto_points: The number of Pareto points to generate.
        num_greedy: The number of Pareto points to obtain greedily based on the
            hypervolume improvement when truncating the oversampled Pareto points.
        num_samples: The number of Monte Carlo samples.
        num_restarts: The number of restarts for the acquisition optimizer.
        raw_samples: The number of raw samples for the acquisition optimizer.
        batch_size: The batch size.

    Returns:
        x_n: A `batch_size x d`-dim Tensor containing the candidate.
    """
    tkwargs = {
        "dtype": torch.double,
        "device": "cpu",
    }
    ###############################################################################
    # FIT MODEL
    ###############################################################################
    before_model_time = time.time()
    model = fit_model(train_X, train_Y, num_outputs, params=params)
    model_time = time.time() - before_model_time
    ###############################################################################
    # THOMPSON SAMPLE
    ###############################################################################
    before_pareto_time = time.time()
    if acquisition_type in ["jes_0", "jes_lb", "jes_lb2", "jes_mc", "mes_0", "mes_lb", "mes_lb2", "mes_mc", "pes"]:
        # `num_pareto_samples x num_fantasies x num_pareto_points x d`
        # `num_pareto_samples x num_fantasies x num_pareto_points x M`
        for _ in range(3):
            try:
                pareto_sets, pareto_fronts = sample_pareto_sets_and_fronts(
                    model=model,
                    num_pareto_samples=num_pareto_samples,
                    num_pareto_points=num_pareto_points,
                    bounds=bounds,
                    generations=500,
                    pop_size=100,
                    num_greedy=num_greedy
                )
                break
            except RuntimeError:
                pass
        else:
            raise RuntimeError
        pareto_time = time.time() - before_pareto_time
    else:
        pareto_time = 0.0
    ###############################################################################
    # BOX DECOMPOSITION
    ###############################################################################
    before_box_time = time.time()
    if acquisition_type in ["jes_0", "jes_lb", "jes_lb2", "jes_mc", "mes_0", "mes_lb", "mes_lb2", "mes_mc"]:
        expanded_shape = pareto_fronts.shape[0:2] + train_Y.shape
        ty = train_Y.unsqueeze(0).unsqueeze(0).expand(expanded_shape)
        aug_pfs = torch.cat([ty, pareto_fronts], dim=-2)

        # `num_pareto_samples x num_fantasies x 2 x J x M`
        hypercell_bounds = compute_box_decomposition(aug_pfs)
        box_time = time.time() - before_box_time
    else:
        box_time = 0.0
    ###############################################################################
    # INITIALIZE ACQUISITION
    ###############################################################################
    before_acq_init_time = time.time()

    if acquisition_type == "jes_0":
        acq = qLowerBoundJointEntropySearch(
            model=model,
            pareto_sets=pareto_sets.squeeze(1),
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Noiseless",
            sampling_noise=True,
            adjustment=True,
        )

    if acquisition_type == "jes_lb":
        acq = qLowerBoundJointEntropySearch(
            model=model,
            pareto_sets=pareto_sets.squeeze(1),
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Lower bound",
            sampling_noise=True,
        )

    if acquisition_type == "jes_lb2":
        acq = qLowerBoundJointEntropySearch(
            model=model,
            pareto_sets=pareto_sets.squeeze(1),
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Lower bound",
            sampling_noise=True,
            only_diagonal=True,
        )

    if acquisition_type == "jes_mc":
        acq = qLowerBoundJointEntropySearch(
            model=model,
            pareto_sets=pareto_sets.squeeze(1),
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Monte Carlo",
            sampling_noise=True,
            num_samples=num_samples,
        )

    if acquisition_type == "mes_0":
        acq = qLowerBoundMaximumEntropySearch(
            model=model,
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Noiseless",
            sampling_noise=True,
        )

    if acquisition_type == "mes_lb":
        acq = qLowerBoundMaximumEntropySearch(
            model=model,
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Lower bound",
            sampling_noise=True,
        )

    if acquisition_type == "mes_lb2":
        acq = qLowerBoundMaximumEntropySearch(
            model=model,
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Lower bound",
            sampling_noise=True,
            only_diagonal=True,
        )

    if acquisition_type == "mes_mc":
        acq = qLowerBoundMaximumEntropySearch(
            model=model,
            pareto_fronts=pareto_fronts.squeeze(1),
            hypercell_bounds=hypercell_bounds.squeeze(1),
            estimation_type="Monte Carlo",
            sampling_noise=True,
            num_samples=num_samples,
        )

    if acquisition_type == "pes":
        jitter = 1e-4
        max_retries = 2
        t = 0
        acq = None
        while acq is None and (t < max_retries):
            try:
                acq = qPredictiveEntropySearch(
                    model=model,
                    pareto_sets=pareto_sets.squeeze(1),
                    max_ep_iterations=100,
                    threshold=5e-2,
                    ep_jitter=jitter,
                    test_jitter=jitter,
                )
            except:
                jitter = jitter * 10
                t = t + 1

    if acquisition_type == "ehvi":
        ref_point = torch.amin(train_Y, dim=0) - 0.1 * abs(
            torch.amin(train_Y, dim=0)
        )
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point, Y=train_Y
        )
        if batch_size == 1:
            acq = ExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                partitioning=partitioning
            )
        else:
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
            acq = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                partitioning=partitioning,
                sampler=sampler,
            )

    if acquisition_type == "nehvi":
        ref_point = torch.amin(train_Y, dim=0) - 0.1 * abs(
            torch.amin(train_Y, dim=0)
        )

        sampler = SobolQMCNormalSampler(num_samples=num_samples)

        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X,
            prune_baseline=True,
            sampler=sampler,
        )

    if acquisition_type == "parego":
        acq_list = []
        for _ in range(batch_size):
            weights = sample_simplex(model.num_outputs, **tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=train_Y)
            )

            sampler = SobolQMCNormalSampler(num_samples=num_samples)

            acq = qExpectedImprovement(
                model=model,
                objective=objective,
                best_f=max(objective(train_Y)),
                sampler=sampler
            )
            acq_list.append(acq)

    if acquisition_type == "nparego":
        with torch.no_grad():
            pred_Y = model.posterior(train_X).mean

        acq_list = []
        for _ in range(batch_size):
            weights = sample_simplex(model.num_outputs, **tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred_Y)
            )

            sampler = SobolQMCNormalSampler(num_samples=num_samples)

            acq = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=train_X,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_list.append(acq)

    if acquisition_type == "sobol":
        # `q x d`
        x_n = draw_sobol_samples(bounds=bounds, n=1, q=batch_size).squeeze(0)

    if acquisition_type == "ts":
        pareto_set, pareto_front = sample_pareto_sets_and_fronts(
            model=model,
            num_pareto_samples=1,
            num_pareto_points=batch_size,
            bounds=bounds,
            generations=500,
            pop_size=100,
            num_greedy=batch_size
        )
        # `1 x d`
        x_n = pareto_set.squeeze(0).squeeze(0)

    acq_init_time = time.time() - before_acq_init_time
    ###############################################################################
    # OPTIMIZE ACQUISITION
    ###############################################################################
    before_opt_time = time.time()
    if acquisition_type not in ["sobol", "ts", "ots", "pes", "parego", "nparego", "ehvi", "nehvi"]:
        # candidates, acq_value = optimize_acqf(
        #     acq_function=acq,
        #     bounds=bounds,
        #     q=batch_size,
        #     num_restarts=num_restarts,
        #     raw_samples=raw_samples,
        # )
        candidates, acq_value = optimize_acqf_discrete(
            acq_function=acq,
            q=batch_size,
            choices=designs,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        # `q x d`
        x_n = candidates
        
    if acquisition_type in ["ehvi", "nehvi"]:
        candidates, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
        )

        # `q x d`
        x_n = candidates

    if acquisition_type in ["parego", "nparego"]:
        candidates, acq_value = optimize_acqf_list(
            acq_function_list=acq_list,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        # `q x d`
        x_n = candidates

    if acquisition_type == "pes":
        try:
            # candidates, acq_value = optimize_acqf(
            #     acq_function=acq,
            #     bounds=bounds,
            #     q=1,
            #     num_restarts=num_restarts,
            #     raw_samples=raw_samples,
            # )

            candidates, acq_value = optimize_acqf_fd(
                acq_function=acq,
                bounds=bounds,
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            # `q x d`
            x_n = candidates
        except:
            # failed = int(len(train_X))
            # torch.save(failed, "pes_except_" + str(failed) + ".pt")

            # just take the first candidate from the pareto set
            if batch_size == 1:
                x_n = pareto_sets[0, 0, 0:1, :]
            else:
                x_n = pareto_sets.reshape(
                    num_pareto_samples * num_pareto_points, pareto_sets.shape[-1]
                )[0:batch_size, :]

    opt_time = time.time() - before_opt_time

    return x_n
