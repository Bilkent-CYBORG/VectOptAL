#!/usr/bin/env python3

r"""
Acquisition functions for maximum value entropy search for multi-objective Bayesian
optimization (MES).

"""
from __future__ import annotations
from typing import Any, Callable, Optional
import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning
)
from gpytorch.functions import logdet

from algorithms.JESMO.jes.acquisition.jes import (
    compute_box_decomposition,
    _compute_entropy_noiseless,
    _compute_entropy_noiseless_upper_bound,
    _compute_entropy_upper_bound,
    _compute_entropy_monte_carlo,
)

from math import pi

CLAMP_LB = 1.0e-8
NEG_INF = -1e+10


class qMaximumEntropySearch(AcquisitionFunction):
    r"""The acquisition function for Maximum Entropy Search.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal output.

    The batch case `q > 1` is supported through cyclic optimization and fantasies.
    """

    def __init__(
        self,
        model: Model,
        num_pareto_samples: int,
        num_pareto_points: int,
        sample_pareto_fronts: Callable[
            [Model, Model, int, int, Tensor], Tensor
        ],
        bounds: Tensor,
        num_fantasies: int,
        partitioning: BoxDecomposition = DominatedPartitioning,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Noiseless",
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Maximum entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k.
            num_pareto_samples: The number of samples for the Pareto optimal input
                and outputs.
            num_pareto_points: The number of Pareto points for each sample.
            sample_pareto_fronts: A callable that takes the initial model,
                the fantasy model, the number of pareto samples, the input bounds
                and returns the Pareto optimal outputs:
                - pareto_fronts: a `num_pareto_samples x num_fantasies x
                    num_pareto_points x M`-dim Tensor.
            bounds: a `2 x d`-dim Tensor containing the input bounds for
                multi-objective optimization.
            num_fantasies: Number of fantasies to generate. Ignored if `X_pending`
                is `None`.
            partitioning: A `BoxDecomposition` module that is used to obtain the
                hyper-rectangle bounds for integration. In the unconstrained case,
                this gives the partition of the dominated space. In the constrained
                case, this gives the partition of the feasible dominated space union
                the infeasible space.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "Noiseless", "Noiseless lower bound", "Lower bound" or
                "Monte Carlo".
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples=64,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.

        """
        super().__init__(model=model)

        self._init_model = model
        self.model = model

        self.num_pareto_samples = num_pareto_samples
        self.num_pareto_points = num_pareto_points
        self.sample_pareto_fronts = sample_pareto_fronts

        self.fantasies_sampler = SobolQMCNormalSampler(num_fantasies)
        self.num_fantasies = num_fantasies

        self.partitioning = partitioning

        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0
        self.bounds = bounds

        self.estimation_type = estimation_type
        if estimation_type not in ["Noiseless", "Noiseless lower bound",
                                   "Lower bound", "Monte Carlo"]:
            raise NotImplementedError(
                "Currently the only supported estimation type are: "
                "['Noiseless', 'Noiseless lower bound', 'Lower bound', 'Monte Carlo'"
                "]."
            )
        self.only_diagonal = only_diagonal
        if sampler is None:
            self.sampler = SobolQMCNormalSampler(
                num_samples=num_samples, collapse_batch_dims=True
            )
        else:
            self.sampler = sampler

        self.num_constraints = num_constraints

        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points, fantasizes the
        model on the pending points and draws pareto optimal samples from the
        fantasized model posterior.

        Args:
            X_pending: `num_pending x d` Tensor with `num_pending` `d`-dim design
                points that have been submitted for evaluation but have not yet been
                evaluated.
        """

        if X_pending is not None:
            # fantasize the model on pending points
            fantasy_model = self._init_model.fantasize(
                X=X_pending,
                sampler=self.fantasies_sampler,
                observation_noise=True
            )
            self.model = fantasy_model

        self._sample_pareto_fronts()

        # Compute the box decompositions
        with torch.no_grad():
            self.hypercell_bounds = compute_box_decomposition(
                self.pareto_fronts,
                self.partitioning,
                self.maximize,
                self.num_constraints
            )

    def _sample_pareto_fronts(self) -> None:
        r"""Sample Pareto optimal output for the Monte Carlo tion of the entropy in
        Maximum Entropy Search.

        Note: sampling exactly `num_pareto_points` of Pareto optimal inputs and
        outputs is achieved by over generating points and then truncating the sample.
        """
        with torch.no_grad():
            # pareto_fronts shape:
            # `num_pareto_samples x num_fantasies x num_pareto_points x M`
            pareto_fronts = self.sample_pareto_fronts(
                model=self._init_model,
                fantasy_model=self.model,
                num_pareto_samples=self.num_pareto_samples,
                num_pareto_points=self.num_pareto_points,
                bounds=self.bounds,
                maximize=self.maximize,
            )

            self.pareto_fronts = pareto_fronts

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute maximum entropy search at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches with `1`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MES values at the given design points `X`.
        """
        K = self.num_constraints
        M = self._init_model.num_outputs - K

        # compute the prior entropy term depending on `X`
        posterior_plus_noise = self.model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )
        # additional constant term
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))

        # the variance initially has shape `batch_shape x num_fantasies x 1 x
        # (M + K)`
        # prior_entropy has shape `batch_shape x num_fantasies`
        prior_entropy = add_term + .5 * torch.log(
            posterior_plus_noise.variance.clamp_min(CLAMP_LB)
        ).sum(-1).squeeze(-1)

        # compute the posterior entropy term
        posterior = self.model.posterior(
            X.unsqueeze(-2), observation_noise=False
        )

        # shapes `batch_shape x num_fantasies x 1 x (M + K)`
        mean = posterior.mean
        var = posterior.variance.clamp_min(CLAMP_LB)
        var_plus_noise = posterior_plus_noise.variance.clamp_min(CLAMP_LB)

        # expand shapes to `batch_shape x num_pareto_samples x num_fantasies x 1 x
        # (M + K)`
        new_shape = mean.shape[:-3] + torch.Size([self.num_pareto_samples]) + \
            mean.shape[-3:]
        mean = mean.unsqueeze(-4).expand(new_shape)
        var = var.unsqueeze(-4).expand(new_shape)
        var_plus_noise = var_plus_noise.unsqueeze(-4).expand(new_shape)

        # `batch_shape x num_fantasies` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds,
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
            )
        if self.estimation_type == "Noiseless lower bound":
            post_entropy = _compute_entropy_noiseless_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=mean,
                variance=var,
                initial_variance=var,
                initial_variance_plus_noise=var_plus_noise,
            )
        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
                only_diagonal=self.only_diagonal
            )
        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x num_fantasies x 1 x (M+K)`
            samples = self.sampler(posterior_plus_noise)

            # `num_mc_samples x batch_shape x num_fantasies`
            if (M + K) == 1:
                samples_log_prob = posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = posterior_plus_noise.mvn.log_prob(
                    samples
                )

            # expand shape to `num_mc_samples x batch_shape x num_pareto_samples x
            # num_fantasies x 1 x (M+K)`
            new_shape = samples.shape[:-3] \
                + torch.Size([self.num_pareto_samples]) + samples.shape[-3:]
            samples = samples.unsqueeze(-4).expand(new_shape)

            # expand shape to `num_mc_samples x batch_shape x num_pareto_samples
            # x num_fantasies`
            new_shape = samples_log_prob.shape[:-1] \
                + torch.Size([self.num_pareto_samples]) + samples_log_prob.shape[-1:]
            samples_log_prob = samples_log_prob.unsqueeze(-2).expand(new_shape)

            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds,
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
                samples=samples,
                samples_log_prob=samples_log_prob
            )
        # average over the fantasies
        return (prior_entropy - post_entropy).mean(dim=-1)


class qLowerBoundMaximumEntropySearch(AcquisitionFunction):
    r"""The acquisition function for the Maximum Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal outputs.
    """

    def __init__(
        self,
        model: Model,
        pareto_fronts: Tensor,
        hypercell_bounds: Tensor,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Noiseless",
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Maximum entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints.
            pareto_fronts: a `num_pareto_samples x num_pareto_points x M`-dim Tensor.
            hypercell_bounds:  a `num_pareto_samples x 2 x J x (M + K)`-dim
                Tensor containing the hyper-rectangle bounds for integration. In the
                unconstrained case, this gives the partition of the dominated space.
                In the constrained case, this gives the partition of the feasible
                dominated space union the infeasible space.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "Noiseless", "Noiseless lower bound", "Lower bound" or
                "Monte Carlo".
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples=64,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.
        """
        super().__init__(model=model)
        self.model = model

        self.pareto_fronts = pareto_fronts
        self.num_pareto_samples = pareto_fronts.shape[0]
        self.num_pareto_points = pareto_fronts.shape[-2]

        self.hypercell_bounds = hypercell_bounds
        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0

        self.estimation_type = estimation_type
        if estimation_type not in ["Noiseless", "Noiseless lower bound",
                                   "Lower bound", "Monte Carlo"]:
            raise NotImplementedError(
                "Currently the only supported estimation type are: "
                "['Noiseless', 'Noiseless lower bound', 'Lower bound', 'Monte Carlo'"
                "]."
            )
        self.only_diagonal = only_diagonal
        if sampler is None:
            self.sampler = SobolQMCNormalSampler(
                num_samples=num_samples, collapse_batch_dims=True
            )
        else:
            self.sampler = sampler

        self.num_constraints = num_constraints
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute maximum entropy search at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MES values at the given design points `X`.
        """
        K = self.num_constraints
        M = self.model.num_outputs - K

        # compute the initial entropy term depending on `X`
        posterior_plus_noise = self.model.posterior(X, observation_noise=True)

        # additional constant term
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))
        # the variance initially has shape `batch_shape x (q*(M+K)) x (q*(M+K))`
        # prior_entropy has shape `batch_shape x num_fantasies`
        prior_entropy = add_term + .5 * logdet(
            posterior_plus_noise.mvn.covariance_matrix
        )

        # compute the posterior entropy term
        posterior = self.model.posterior(X.unsqueeze(-2), observation_noise=False)
        posterior_plus_noise = self.model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )

        # `batch_shape x q x 1 x (M+K)`
        mean = posterior.mean
        var = posterior.variance.clamp_min(CLAMP_LB)
        var_plus_noise = posterior_plus_noise.variance.clamp_min(CLAMP_LB)

        # expand shapes to `batch_shape x num_pareto_samples x q x 1 x (M + K)`
        new_shape = mean.shape[:-3] + torch.Size([self.num_pareto_samples]) + \
            mean.shape[-3:]
        mean = mean.unsqueeze(-4).expand(new_shape)
        var = var.unsqueeze(-4).expand(new_shape)
        var_plus_noise = var_plus_noise.unsqueeze(-4).expand(new_shape)

        # `batch_shape x q` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
            )
        if self.estimation_type == "Noiseless lower bound":
            post_entropy = _compute_entropy_noiseless_upper_bound(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                initial_variance=var,
                initial_variance_plus_noise=var_plus_noise,
            )
        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
                only_diagonal=self.only_diagonal
            )

        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x q x 1 x (M+K)`
            samples = self.sampler(posterior_plus_noise)

            # `num_mc_samples x batch_shape x q`
            if (M + K) == 1:
                samples_log_prob = posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = posterior_plus_noise.mvn.log_prob(
                    samples
                )

            # expand shape to `num_mc_samples x batch_shape x num_pareto_samples x
            # q x 1 x (M+K)`
            new_shape = samples.shape[:-3] \
                + torch.Size([self.num_pareto_samples]) + samples.shape[-3:]
            samples = samples.unsqueeze(-4).expand(new_shape)

            # expand shape to `num_mc_samples x batch_shape x num_pareto_samples x q`
            new_shape = samples_log_prob.shape[:-1] \
                + torch.Size([self.num_pareto_samples]) + samples_log_prob.shape[-1:]
            samples_log_prob = samples_log_prob.unsqueeze(-2).expand(new_shape)

            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=mean,
                variance=var,
                variance_plus_noise=var_plus_noise,
                samples=samples,
                samples_log_prob=samples_log_prob
            )

        # sum over the batch
        return prior_entropy - post_entropy.sum(dim=-1)
