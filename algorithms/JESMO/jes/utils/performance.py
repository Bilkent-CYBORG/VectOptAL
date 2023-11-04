#!/usr/bin/env python3

r"""
An example to show the operations that were used to generate the Pareto set.
In particular, we maximize the posterior mean at each iteration using NSGA2 to obtain
an approximate Pareto set. We then restrict the pareto set to 50 points using a
hypervolume truncation. Then we can compute the different multi-objective statistics
such as the weighted hypervolume.
"""
import torch
import time
import numpy as np

from typing import Optional, Tuple
from torch import Tensor
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning
)

from scipy.stats import beta
from scipy.stats.qmc import Sobol
from scipy.special import gamma

from algorithms.JESMO.jes.utils.sample_pareto import pareto_solver
from algorithms.JESMO.jes.utils.bo_loop import fit_model


def hv_truncation(
        num_gen: int,
        sample_pf: Tensor,
        num_objectives: int,
        train_Y: Tensor
) -> Tensor:
    r"""Perform the hypervolume truncation.

    Args:
        num_gen: The desired number of Pareto points.
        sample_pf: A `P x M`-dim Tensor containing the oversampled Pareto front .
        num_objectives: The number of objectives.
        train_Y: A `n x M`-dim Tensor containing the training outputs.

    Returns:
        The indices of the Pareto optimal points to keep.
    """
    M = num_objectives
    indices = []
    ref_point = torch.amin(train_Y, dim=0) - 0.1 * abs(torch.amin(train_Y, dim=0))

    for k in range(num_gen):
        if k == 0:
            hypercell_bounds = torch.zeros(2, M)
            hypercell_bounds[0] = ref_point
            hypercell_bounds[1] = 1e+10
        else:
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point, Y=fantasized_pf
            )

            hypercell_bounds = partitioning.hypercell_bounds

        # `1 x num_boxes x M`
        lo = hypercell_bounds[0].unsqueeze(0)
        up = hypercell_bounds[1].unsqueeze(0)
        # compute hv
        hvi = torch.max(
            torch.min(sample_pf.unsqueeze(-2), up) - lo,
            torch.zeros(lo.shape)
        ).prod(dim=-1).sum(dim=-1)

        # zero out pending points
        hvi[indices] = 0
        # store update
        am = torch.argmax(hvi).tolist()
        indices = indices + [am]

        if k == 0:
            fantasized_pf = sample_pf[am:am + 1, :]
        else:
            fantasized_pf = torch.cat([fantasized_pf, sample_pf[am:am + 1, :]],
                                      dim=0)

    indices = torch.tensor(indices)

    return indices


def get_recommendation(
        train_X: Tensor,
        train_Y: Tensor,
        bounds: Tensor,
        num_samples: int,
        n_initial: int,
        seed: Optional[int] = 123,
) -> Tuple[Tensor, Tensor]:
    r"""Get the recommended Pareto sets and the estimated Pareto fronts.

    Args:
        train_X: A `N x d`-dim Tensor containing the training inputs.
        train_Y: A `N x M`-dim Tensor containing the training outputs.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_samples: The number of recommendation points.
        n_initial: The number of initial samples.
        seed: The seed for the multi-objective solver.

    Returns:
        ps_total: A `(N + 1) x num_samples x d`-dim Tensor containing the collection
            of recommended Pareto sets.
        pf_total: A `(N + 1) x num_samples x M`-dim Tensor containing the collection
            of estimated Pareto fronts.
    """
    M = train_Y.shape[-1]
    d = train_X.shape[-1]
    N = train_X.shape[0] - n_initial
    ps_total = torch.zeros(N + 1, num_samples, d)
    pf_total = torch.zeros(N + 1, num_samples, M)

    # get params
    standard_bounds = torch.zeros(2, d)
    standard_bounds[1] = 1.0

    ps_old = None
    for n in range(N + 1):
        init_time = time.time()
        tx = normalize(train_X[:(n_initial + n)], bounds)
        ty = train_Y[:(n_initial + n)]
        model = fit_model(tx, ty, M)

        def model_mean(x):
            return model.posterior(x).mean

        np.random.seed(seed)
        # get pareto set and front
        ps_opt, pf_opt = pareto_solver(
            model=model_mean,
            bounds=standard_bounds,
            num_objectives=M,
            generations=500,
            pop_size=500,
            maximize=True
        )
        ps_opt = unnormalize(ps_opt, bounds)

        if ps_old is not None:
            pf_old = model_mean(normalize(ps_old, bounds))

            ps = torch.cat([ps_opt, ps_old])
            pf = torch.cat([pf_opt, pf_old])
        else:
            ps = ps_opt
            pf = pf_opt

        # truncate
        trunc_indices = hv_truncation(num_samples, pf, M, ty)

        trunc_ps = ps[trunc_indices, :]
        trunc_pf = pf[trunc_indices, :]

        ps_total[n, ...] = trunc_ps
        pf_total[n, ...] = trunc_pf

        ps_old = trunc_ps

        print("n = {:3d}/{:3d}, Time taken={:5.2f}".format(n+1, N+1, time.time() - init_time))

    return ps_total, pf_total


def spherical_polar(x: np.ndarray) -> np.ndarray:
    r"""Transform an element on the hypercube onto the positive orthant of the unit
    sphere.

    Args:
        x: A `n x (M-1)`-dim array containing elements of the (M-1)-dimensional 
            hypercube.

    Returns:
        polar_x: A `n x M`-dim Tensor containing the spherical coordinates.
    """

    y = np.pi * x / 2
    M = x.shape[-1] + 1
    polar_x = np.zeros(shape=(len(x), M))
    for m in range(0, M - 1):
        polar_x[:, m] = np.cos(y[:, m])
        if m > 0:
            polar_x[:, m] = polar_x[:, m] * np.prod(np.sin(y[:, 0:m]), axis=-1)

    polar_x[:, -1] = np.prod(np.sin(y), axis=-1)
    return polar_x


def get_beta_distributions(
        num_gen: int
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes a number of local Beta distributions shape parameters.

    Args:
        num_gen: The number of local modes for the Beta distribution.

    Returns:
        alphas: A `num_gen`-dim numpy array containing the shape parameters alpha.
        betas: A `num_gen`-dim numpy array containing the shape parameters beta.
    """
    # get alphas and betas
    means = np.linspace(0.05, 0.95, num_gen)
    var = 0.001
    alphas = np.zeros(num_gen)
    betas = np.zeros(num_gen)

    for i in range(num_gen):
        mean = means[i]
        alphas[i] = mean * mean * (1 - mean) / var - mean
        betas[i] = alphas[i] * (1 - mean) / mean
    return alphas, betas


def compute_ghv(
        a: Tensor,
        b: Tensor,
        pf: Tensor,
        ref_point: Tensor,
        num_mc_samples: int = 2 ** 10,
        seed: Optional[int] = 123,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    r"""Compute the weighted hypervolume.

    Args:
        a: A `(M-1)`-dim Tensor containing the shape parameters alpha.
        b: A `(M-1)`-dim Tensor containing the shape parameters alpha.
        pf: A `num_samples x M`-dim Tensor containing a Pareto front approximation.
        ref_point: A `M`-dim Tensor containing the reference point for the
            hypervolume calculation.
        num_mc_samples: The number of Monte Carlo samples.
        seed: The seed for the Monte Carlo samples.

    Returns:
        The weighted hypervolume.
    """
    A = (pf - ref_point).detach().numpy()
    M = pf.shape[-1]
    c = np.power(np.pi, M / 2) / np.power(2, M) / gamma(M / 2 + 1)

    # mc sample
    eng = Sobol(d=M-1, seed=seed)
    sample = eng.random(num_mc_samples)

    # weights
    weights = []
    for m in range(M - 1):
        beta_dist_m = beta(a=a[m], b=b[m])
        weights = weights + [beta_dist_m.ppf(sample[:, m])]

    w = np.column_stack(weights)
    sw = spherical_polar(w)

    ghv = np.mean(
        np.max(
            np.min(np.clip(np.expand_dims(A, -2) / sw, 0, None) ** M, axis=-1),
            axis=0
        )
    )

    return c * ghv

