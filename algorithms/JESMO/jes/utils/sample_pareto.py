#!/usr/bin/env python3

r"""
Some methods for sampling the Pareto optimal points.
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import torch
import numpy as np
from botorch.models.converter import batched_to_model_list
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.gp_sampling import (
    RandomFourierFeatures,
    get_weights_posterior,
    get_deterministic_model
)
from botorch.models.multitask import MultiTaskGP
from torch import Tensor

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_sampling,
    get_termination,
)
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.config import Config as pymooConfig
pymooConfig.show_compile_hint = False
from math import ceil
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)
from scipy.stats import norm


def get_gp_samples_with_fantasies(
    model: Model,
    fantasy_model: Model,
    num_outputs: int,
    num_samples: int,
    num_rff_features: int = 500,
) -> List[List[GenericDeterministicModel]]:
    r"""Sample functions from GP posterior using random Fourier features.

    This is a hacky way to get samples from the model conditioned at fantasy 
    points. This functionality is only needed when using sequentially greedy
    optimization for batch selection with the Joint Entropy Search acquisition
    function.

    This code was adapted from `botorch.utils.gp_sampling.get_gp_samples`.
    https://github.com/pytorch/botorch

    Args:
        model: The model excluding fantasies.
        fantasy_model: The model including fantasies.
        num_outputs: The number of outputs.
        num_samples: The number of sampled functions to draw.
        num_rff_features: The number of random fourier features.

    Returns:
        A list of `num_fantasies` list of `num_samples` sampled functions.
        `num_fantasies = 1` for non-fantasized models.
    """
    if num_outputs > 1:
        if not isinstance(model, ModelListGP):
            models = batched_to_model_list(model).models
        else:
            models = model.models
    else:
        models = [model]
    if isinstance(models[0], MultiTaskGP):
        raise NotImplementedError

    if len(fantasy_model.batch_shape) > 0:
        num_fantasies = fantasy_model.batch_shape[0]
    else:
        num_fantasies = 1
    
    # check transform
    # intf = None
    # octf = None
    # if hasattr(model, "input_transform"):
        # intf = model.input_transform
    # if hasattr(model, "outcome_transform"):
        # octf = model.outcome_transform
        # model.outcome_transform = None
        
    list_of_samples = []

    # iterate over each fantasy
    for j in range(num_fantasies):
        weights = []
        bases = []
        for m in range(num_outputs):
            # extract the training data from the fantasy model
            if num_fantasies > 1:
                if num_outputs > 1:
                    train_X = fantasy_model.train_inputs[0][j, m, :, :]
                    train_y = fantasy_model.train_targets[j, m, :]
                else:
                    train_X = fantasy_model.train_inputs[0][j, :, :]
                    train_y = fantasy_model.train_targets[j, :]
            else:
                if num_outputs > 1:
                    train_X = fantasy_model.train_inputs[0][m, :, :]
                    train_y = fantasy_model.train_targets[m, :]
                else:
                    train_X = fantasy_model.train_inputs[0]
                    train_y = fantasy_model.train_targets

            # get random fourier features
            # TODO: Add a way to generate rff using Sobol generated base samples.
            basis = RandomFourierFeatures(
                kernel=models[m].covar_module,
                input_dim=train_X.shape[-1],
                num_rff_features=num_rff_features,
            )
            bases.append(basis)
            phi_X = basis(train_X)
            # sample weights from bayesian linear model
            mvn = get_weights_posterior(
                X=phi_X,
                y=train_y,
                sigma_sq=models[m].likelihood.noise.mean().item(),
            )
            weights.append(mvn.sample(torch.Size([num_samples])))
        # construct a deterministic, multi-output model for each sample
        # Note: we need an individual deterministic model for each sample to run the
        # genetic algorithm
        sample_j = [
            get_deterministic_model(
                weights=[weights[m][i] for m in range(num_outputs)],
                bases=bases,
            )
            for i in range(num_samples)
        ]
        
        # check transform
        # for i in range(num_samples):
            # if intf is not None:
                # sample_j[i].input_transform = intf
            # if octf is not None:
                # sample_j[i].outcome_transform = octf
        
        list_of_samples = list_of_samples + [sample_j]

    return list_of_samples


def pareto_solver(
    model: GenericDeterministicModel,
    bounds: Tensor,
    num_objectives: int,
    generations: int = 100,
    pop_size: int = 100,
    maximize: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Runs pymoo genetic algorithm NSGA2 to compute the Pareto set and front.
        https://pymoo.org/algorithms/moo/nsga2.html

    Args:
        model: The random Fourier feature GP sample.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_objectives: The number of objectives.
        generations: The number of generations of NSGA2.
        pop_size: The population size maintained at each step of NSGA2.
        maximize: If true we solve for the Pareto maximum.

    Returns:
        The `num_pareto_points` pareto optimal set and front, where
        `num_pareto_points <= pop_size` depends randomly on the model and
        parameter choices.

        pareto_set: A `num_pareto_points x d`-dim Tensor.
        pareto_front: A `num_pareto_points x num_objectives`-dim Tensor.
    """
    d = bounds.shape[-1]
    weight = -1.0 if maximize else 1.0

    class PymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=d,
                n_obj=num_objectives,
                n_constr=0,
                xl=bounds[0].detach().numpy(),
                xu=bounds[1].detach().numpy(),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            xt = torch.tensor(x, dtype=torch.float)
            out["F"] = weight * model(xt).detach().numpy()
            return out

    # use NSGA2 to generate a number of Pareto optimal points.
    results = minimize(
        problem=PymooProblem(),
        algorithm=NSGA2(
            pop_size=pop_size,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
        ),
        termination=get_termination("n_eval", 1500),  # ("n_gen", generations)
    )
    pareto_set = torch.tensor(results.X, dtype=torch.float)
    pareto_front = weight * torch.tensor(results.F, dtype=torch.float)

    if pareto_set.ndim == 1:
        return pareto_set.unsqueeze(0), pareto_front.unsqueeze(0)
    else:
        return pareto_set, pareto_front


def get_optimistic_samples(
        model: Model,
        num_samples: int,
        beta: Optional[Tensor] = None,
) -> List[List[GenericDeterministicModel]]:
    r""" Generates the optimistic samples of the form `mean + beta * sqrt(var)`.
    Args:
        model: The fitted model.
        num_samples: The number of samples to generate.
        beta: A `num_samples`-dim Tensor that contains the parameter controlling the
            optimism for the samples. Defaults to an evenly spaced sequence between
            `1` and `5`.

    Returns:
        A list of `num_fantasies` list of `num_samples` sampled functions.
        `num_fantasies = 1` for non-fantasized models.
    """
    if len(model.batch_shape) > 0:
        num_fantasies = model.batch_shape[0]
    else:
        num_fantasies = 1

    if beta is None:
        beta = torch.tensor(norm.ppf(np.linspace(0.55, 0.999, num_samples)))
        # beta = torch.linspace(1, 5, num_samples)
    list_of_samples = []

    if num_fantasies == 1:
        for i in range(num_samples):
            def sample_i(X):
                mean = model.posterior(X).mean
                var = model.posterior(X).variance
                return mean + beta[i] * torch.sqrt(var)
            function_i = GenericDeterministicModel(sample_i)

            list_of_samples = list_of_samples + [function_i]

        return [list_of_samples]

    else:
        # iterate over each fantasy
        for j in range(num_fantasies):
            list_of_samples_j = []
            for i in range(num_samples):
                # hacky way to get fantasized optimistic sample
                def sample_ij(X):
                    mean = model.posterior(X).mean[j, ...]
                    var = model.posterior(X).mean[j, ...]
                    return mean + beta[i] * torch.sqrt(var)

                function_ij = GenericDeterministicModel(sample_ij)
                list_of_samples_j = list_of_samples + [function_ij]

            list_of_samples = list_of_samples + [list_of_samples_j]

        return list_of_samples


def sample_pareto_sets_and_fronts(
    model: Model,
    num_pareto_samples: int,
    num_pareto_points: int,
    bounds: Tensor,
    maximize: bool = True,
    generations: int = 100,
    pop_size: int = 100,
    num_rff_features: int = 500,
    max_tries: Optional[int] = 3,
    fantasy_model: Optional[Model] = None,
    num_greedy: Optional[int] = 0,
    optimistic_sampling: Optional[bool] = False,
    beta: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the Pareto optimal set and front from samples of the GP.

    (i) Samples are generated using random Fourier features.
    (ii) Samples are optimized using NSGA2 (a genetic algorithm).

    Args:
        model: The model excluding fantasies.
        num_pareto_samples: The number of GP samples.
        num_pareto_points: The number of Pareto optimal points to be outputted.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        maximize: If true we solve for the Pareto maximum.

        generations: The number of generations of NSGA2.
        pop_size: The population size maintained at each step of NSGA2.
        num_rff_features: The number of random Fourier features used for GP
            sampling. Defaults to `500`.
        max_tries: The maximum number of runs of NSGA2 to find num_pareto_points.

        fantasy_model: The model including fantasies.
        num_greedy: The number of points to select via the hypervolume improvement
            truncation.

        optimistic_sampling: If true we use the posterior mean and variance to
            construct the samples instead of sampling using rff.
        beta: A `num_samples`-dim Tensor containing the level of optimism.

    Returns:
        pareto_sets: A `num_samples x num_fantasies x num_pareto_points x d`-dim
            Tensor
        pareto_fronts: A `num_samples x num_fantasies x num_pareto_points x M`-dim
            Tensor.
    """
    M = model.num_outputs
    d = bounds.shape[-1]

    if fantasy_model is None:
        fantasy_model = model

    if not optimistic_sampling:
        samples = get_gp_samples_with_fantasies(
            model, fantasy_model, M, num_pareto_samples, num_rff_features
        )
    else:
        samples = get_optimistic_samples(
            fantasy_model, num_pareto_samples, beta
        )
    
    num_fantasies = len(samples)

    # `num_fantasies x M x N`
    train_y = fantasy_model.train_targets
    if M == 1:
        train_y = train_y.unsqueeze(-2)
    if num_fantasies == 1:
        train_y = train_y.unsqueeze(0)
    # `num_fantasies x N x M`
    train_y = train_y.swapaxes(-2, -1)

    # `num_fantasies x M x N x d`
    train_x = fantasy_model.train_inputs[0]
    if M == 1:
        train_x = train_x.unsqueeze(-3)
    if num_fantasies == 1:
        train_x = train_x.unsqueeze(0)
    # `num_fantasies x N x d`
    train_x = train_x[:, 0, :, :]

    pareto_sets = torch.zeros(
        (num_pareto_samples, num_fantasies, num_pareto_points, d)
    )
    pareto_fronts = torch.zeros(
        (num_pareto_samples, num_fantasies, num_pareto_points, M)
    )
    for i in range(num_pareto_samples):
        for j in range(num_fantasies):

            ratio = 2
            pop_size_new = pop_size
            generations_new = generations

            n_tries = 0
            # run solver until more than `num_pareto_points` found or exceeds
            # the maximum number of tries
            while (ratio > 1) and (n_tries < max_tries):
                ps, pf = pareto_solver(
                    model=samples[j][i],
                    bounds=bounds,
                    num_objectives=M,
                    generations=generations_new,
                    pop_size=pop_size_new,
                    maximize=maximize,
                )
                num_pareto_generated = ps.shape[0]
                ratio = ceil(num_pareto_points / num_pareto_generated)
                # pop_size_new = pop_size_new * 2
                # generations_new = generations_new * 2

                n_tries = n_tries + 1

            # untransform pareto front
            try:
                pf = model.outcome_transform.untransform(pf)[0]
            except AttributeError:
                pf = pf

            if ratio > 1:
                error_text = "Only found " + str(num_pareto_generated) + \
                    " Pareto efficient points instead of " + \
                    str(num_pareto_points) + "."
                raise RuntimeError(error_text)

            # Randomly truncate the Pareto set and front
            if num_greedy == 0:
                indices = torch.randperm(num_pareto_generated)[:num_pareto_points]

            # Truncate Pareto set and front based on the immediate hypervolume
            # improvement.
            else:
                # get `num_pareto_points - num_greedy` indices randomly
                indices = torch.randperm(
                    num_pareto_generated
                )[:num_pareto_points-num_greedy].tolist()
                
                fantasized_vec = samples[j][i](train_x[j, :, :])
                # fantasized_vec = train_y[j, :, :]
                
                try:
                    ty = model.outcome_transform.untransform(train_y[j, :, :])[0]
                    fantasized_pf = model.outcome_transform.untransform(
                        fantasized_vec
                    )[0]
                except AttributeError:
                    ty = train_y[j, :, :]
                    fantasized_pf = fantasized_vec

                # get the `num_greedy` indices greedily
                ref_point = torch.amin(ty, dim=0) - 0.1 * abs(torch.amin(ty, dim=0))
                
                for k in range(num_greedy):
                    partitioning = FastNondominatedPartitioning(
                        ref_point=ref_point, Y=fantasized_pf
                    )
                    hypercell_bounds = partitioning.hypercell_bounds
                    # `1 x num_boxes x M`
                    lo = hypercell_bounds[0].unsqueeze(0)
                    up = hypercell_bounds[1].unsqueeze(0)
                    # compute hv
                    hvi = torch.max(
                        torch.min(pf.unsqueeze(-2), up) - lo,
                        torch.zeros(lo.shape)
                    ).prod(dim=-1).sum(dim=-1)
                    # zero out pending points
                    hvi[indices] = 0
                    # store update
                    am = torch.argmax(hvi).tolist()
                    indices = indices + [am]
                    fantasized_pf = torch.cat([fantasized_pf, pf[am:am+1, :]], dim=0)
                indices = torch.tensor(indices)

            pareto_sets[i, j, :, :] = ps[indices, :]
            pareto_fronts[i, j, :, :] = pf[indices, :]
    return pareto_sets, pareto_fronts
