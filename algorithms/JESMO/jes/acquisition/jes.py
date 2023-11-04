#!/usr/bin/env python3

r"""
Acquisition functions for joint entropy search for multi-objective Bayesian
optimization (JES).

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

from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from botorch import settings
from torch.distributions import Normal
from gpytorch.functions import logdet

from math import pi


CLAMP_LB = 1.0e-8
NEG_INF = -1e+10


class qJointEntropySearch(AcquisitionFunction):
    r"""The acquisition function for Joint Entropy Search.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal input-output pairs.

    The batch case `q > 1` is supported through cyclic optimization and fantasies.

    TODO: Implement a user-defined tolerance for the sampling noise.
    """

    def __init__(
        self,
        model: Model,
        num_pareto_samples: int,
        num_pareto_points: int,
        sample_pareto_sets_and_fronts: Callable[
            [Model, Model, int, int, Tensor], Tensor
        ],
        bounds: Tensor,
        num_fantasies: int,
        partitioning: BoxDecomposition = DominatedPartitioning,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Noiseless",
        sampling_noise: Optional[bool] = True,
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k.
            num_pareto_samples: The number of samples for the Pareto optimal input
                and outputs.
            num_pareto_points: The number of Pareto points for each sample.
            sample_pareto_sets_and_fronts: A callable that takes the initial model,
                the fantasy model, the number of pareto samples, the input bounds
                and returns the Pareto optimal set of inputs and outputs:
                - pareto_sets: a `num_pareto_samples x num_fantasies x
                    num_pareto_points x d`-dim Tensor
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
                computed: "Noiseless", "Noiseless lower bound", "Lower bound"
                or "Monte Carlo".
            sampling_noise: If True we assume that there is noise when sampling the
                Pareto optimal points. We advise always setting `sampling_noise =
                True`. The JES estimate tends to exhibit a large variance when using
                the `Noiseless`, `Noiseless lower bound` or `Lower Bound` entropy
                estimation strategy if this is turned off.
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.

        """
        super().__init__(model=model)

        self._init_model = model
        self.prior_model = model
        self.posterior_model = model

        self.num_pareto_samples = num_pareto_samples
        self.num_pareto_points = num_pareto_points
        self.sample_pareto_sets_and_fronts = sample_pareto_sets_and_fronts

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
        self.sampling_noise = sampling_noise
        self.only_diagonal = only_diagonal
        if sampler is None:
            self.sampler = SobolQMCNormalSampler(
                num_samples=num_samples, collapse_batch_dims=True
            )
        else:
            self.sampler = sampler

        self.num_constraints = num_constraints
        self.hypercell_bounds = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points, fantasizes the
        model on the pending points and draws pareto optimal samples from the
        fantasized model posterior.

        Args:
            X_pending: A `num_pending x d` Tensor with `num_pending` `d`-dim design
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
            self.prior_model = fantasy_model

        self._sample_pareto_points()

        # Condition the model on the sampled pareto optimal points
        # Need to call posterior otherwise gpytorch runtime error
        # "Fantasy observations can only be added after making predictions with a
        # model so that all test independent caches exist."
        with fantasize_flag():
            with settings.propagate_grads(False):
                post_ps = self.prior_model.posterior(
                    self.pareto_sets, observation_noise=False
                )
            if self.sampling_noise:
                # condition with observation noise
                self.posterior_model = self.prior_model.condition_on_observations(
                    X=self.prior_model.transform_inputs(self.pareto_sets),
                    Y=self.pareto_fronts
                )
            else:
                # condition without observation noise
                self.posterior_model = self.prior_model.condition_on_observations(
                    X=self.prior_model.transform_inputs(self.pareto_sets),
                    Y=self.pareto_fronts,
                    noise=torch.zeros(self.pareto_fronts.shape)
                )

        # Compute the box decompositions
        with torch.no_grad():
            self.hypercell_bounds = compute_box_decomposition(
                self.pareto_fronts,
                self.partitioning,
                self.maximize,
                self.num_constraints
            )

    def _sample_pareto_points(self) -> None:
        r"""Sample the Pareto optimal input-output pairs for the Monte Carlo
        approximation of the entropy in Joint Entropy Search.

        Note: Sampling exactly `num_pareto_points` of Pareto optimal inputs and
        outputs is achieved by over generating points and then truncating the sample.
        """
        with torch.no_grad():
            # pareto_sets shape:
            # `num_pareto_samples x num_fantasies x num_pareto_points x d`
            # pareto_fronts shape:
            # `num_pareto_samples x num_fantasies x num_pareto_points x M`
            pareto_sets, pareto_fronts = self.sample_pareto_sets_and_fronts(
                model=self._init_model,
                fantasy_model=self.prior_model,
                num_pareto_samples=self.num_pareto_samples,
                num_pareto_points=self.num_pareto_points,
                bounds=self.bounds,
                maximize=self.maximize,
            )

            self.pareto_sets = pareto_sets
            self.pareto_fronts = pareto_fronts

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute joint entropy search at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches with `1`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """

        K = self.num_constraints
        M = self._init_model.num_outputs - K

        # compute the prior entropy term depending on `X`
        prior_posterior_plus_noise = self.prior_model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )

        # additional constant term
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))
        # the variance initially has shape `batch_shape x num_fantasies x 1 x
        # (M + K)`
        # prior_entropy has shape `batch_shape x num_fantasies`
        prior_entropy = add_term + .5 * torch.log(
            prior_posterior_plus_noise.variance.clamp_min(CLAMP_LB)
        ).sum(-1).squeeze(-1)

        # compute the posterior entropy term
        # Note: we compute the posterior twice here because we need access to
        # the variance with observation noise.
        # [There is probably a better way to do this.]
        post_posterior = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-2), observation_noise=False
        )
        post_posterior_plus_noise = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-2), observation_noise=True
        )

        post_mean = post_posterior.mean
        post_var = post_posterior.variance.clamp_min(CLAMP_LB)
        post_var_plus_noise = post_posterior_plus_noise.variance.clamp_min(CLAMP_LB)

        # `batch_shape x num_fantasies` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
            )
        if self.estimation_type == "Noiseless lower bound":
            prior_posterior = self.prior_model.posterior(
                X.unsqueeze(-2), observation_noise=False
            )
            # the variance initially has shape `batch_shape x num_fantasies x 1 x
            # (M + K)`
            prior_var = prior_posterior.variance.clamp_min(CLAMP_LB)
            prior_var_plus_noise = prior_posterior_plus_noise.variance.clamp_min(
                CLAMP_LB
            )
            # new shape `batch_shape x num_pareto_samples x num_fantasies x 1 x
            # (M + K)`
            new_shape = prior_var.shape[:-3] \
                + torch.Size([self.num_pareto_samples]) + prior_var.shape[-3:]

            prior_var = prior_var.unsqueeze(-4).expand(new_shape)
            prior_var_plus_noise = prior_var_plus_noise.unsqueeze(-4).expand(
                new_shape
            )

            post_entropy = _compute_entropy_noiseless_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                initial_variance=prior_var,
                initial_variance_plus_noise=prior_var_plus_noise,
            )
        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                only_diagonal=self.only_diagonal
            )
        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 1
            # x (M+K)`
            samples = self.sampler(post_posterior_plus_noise)

            # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies`
            if (M + K) == 1:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples
                )

            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                samples=samples,
                samples_log_prob=samples_log_prob
            )

        # average over the fantasies
        return (prior_entropy - post_entropy).mean(dim=-1)


class qLowerBoundJointEntropySearch(AcquisitionFunction):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal input-output pairs.
    """

    def __init__(
        self,
        model: Model,
        pareto_sets: Tensor,
        pareto_fronts: Tensor,
        hypercell_bounds: Tensor,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Noiseless",
        sampling_noise: Optional[bool] = True,
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k.
            pareto_sets: a `num_pareto_samples x num_pareto_points x d`-dim Tensor.
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
            sampling_noise: If True we assume that there is noise when sampling the
                Pareto optimal points. We advise always setting `sampling_noise =
                True`. The JES estimate tends to exhibit a large variance when using
                the `Noiseless`, `Noiseless lower bound` or `Lower Bound` entropy
                estimation strategy if this is turned off.
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
        self.prior_model = model

        self.pareto_sets = pareto_sets
        self.pareto_fronts = pareto_fronts

        self.num_pareto_samples = pareto_fronts.shape[0]
        self.num_pareto_points = pareto_fronts.shape[-2]

        self.sampling_noise = sampling_noise

        # Condition the model on the sampled pareto optimal points
        # Need to call posterior otherwise gpytorch runtime error
        # "Fantasy observations can only be added after making predictions with a
        # model so that all test independent caches exist."
        with fantasize_flag():
            with settings.propagate_grads(False):
                post_ps = self.prior_model.posterior(
                    self.pareto_sets, observation_noise=False
                )
            if self.sampling_noise:
                # condition with observation noise
                self.posterior_model = self.prior_model.condition_on_observations(
                    X=self.prior_model.transform_inputs(self.pareto_sets),
                    Y=self.pareto_fronts,
                )
            else:
                # condition without observation noise
                self.posterior_model = self.prior_model.condition_on_observations(
                    X=self.prior_model.transform_inputs(self.pareto_sets),
                    Y=self.pareto_fronts,
                    noise=torch.zeros(self.pareto_fronts.shape)
                )

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
        r"""Compute joint entropy search at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """
        K = self.num_constraints
        M = self.prior_model.num_outputs - K

        # compute the prior entropy term depending on `X`
        prior_posterior_plus_noise = self.prior_model.posterior(
            X, observation_noise=True
        )

        # additional constant term
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))
        # the variance initially has shape `batch_shape x (q*(M+K)) x (q*(M+K))`
        # prior_entropy has shape `batch_shape`
        prior_entropy = add_term + .5 * logdet(
            prior_posterior_plus_noise.mvn.covariance_matrix
        )

        # compute the posterior entropy term
        post_posterior = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=False
        )
        post_posterior_plus_noise = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=True
        )
        # `batch_shape x num_pareto_samples x q x 1 x (M+K)`
        post_mean = post_posterior.mean.swapaxes(-4, -3)
        post_var = post_posterior.variance.clamp_min(CLAMP_LB).swapaxes(-4, -3)

        post_var_plus_noise = post_posterior_plus_noise.variance.clamp_min(
            CLAMP_LB
        ).swapaxes(-4, -3)

        # `batch_shape x q` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
            )

        if self.estimation_type == "Noiseless lower bound":
            prior_posterior = self.prior_model.posterior(
                X, observation_noise=False
            )
            # the variances initially has shape `batch_shape x q x (M+K)`
            # unsqueeze to get shape `batch_shape x q x 1 x (M+K)`
            prior_var = prior_posterior.variance.clamp_min(CLAMP_LB).unsqueeze(-2)
            prior_var_plus_noise = prior_posterior_plus_noise.variance.clamp_min(
                CLAMP_LB
            ).unsqueeze(-2)

            # new shape `batch_shape x num_pareto_samples x q x 1 x (M + K)`
            new_shape = prior_var.shape[:-3] \
                + torch.Size([self.num_pareto_samples]) + prior_var.shape[-3:]

            prior_var = prior_var.unsqueeze(-4).expand(new_shape)
            prior_var_plus_noise = prior_var_plus_noise.unsqueeze(-4).expand(
                new_shape
            )

            post_entropy = _compute_entropy_noiseless_upper_bound(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                initial_variance=prior_var,
                initial_variance_plus_noise=prior_var_plus_noise,
            )

        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                only_diagonal=self.only_diagonal
            )
        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x q x num_pareto_samples x 1 x (M+K)`
            samples = self.sampler(post_posterior_plus_noise)

            # `num_mc_samples x batch_shape x q x num_pareto_samples`
            if (M + K) == 1:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples
                )

            # swap axes to get
            # samples shape `num_mc_samples x batch_shape x num_pareto_samples x q
            # x 1 x (M+K)`
            # log prob shape `num_mc_samples x batch_shape x num_pareto_samples x q`
            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds.unsqueeze(1),
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                samples=samples.swapaxes(-4, -3),
                samples_log_prob=samples_log_prob.swapaxes(-2, -1)
            )

        # sum over the batch
        return prior_entropy - post_entropy.sum(dim=-1)


def compute_box_decomposition(
        pareto_fronts: Tensor,
        partitioning: BoxDecomposition = DominatedPartitioning,
        maximize: bool = True,
        num_constraints: Optional[int] = 0,
) -> Tensor:
    r"""Compute the box decomposition associated with the sampled Pareto fronts.
    The resulting hypercell bounds is a Tensor of shape `num_pareto_samples x
    num_fantasies x 2 x J x (M + K)`, where `J`= max(num_boxes) i.e. the smallest
    number of boxes needed to partition all the Pareto samples.

    To take advantage of batch computations, we pad the bounds with a `2 x
    (M + K)`-dim Tensor [ref_point, ref_point], when the number of boxes
    required is smaller than `max(num_boxes)`.

    An input x is considered feasible if f_k(x) <= 0.

    Args:
        pareto_fronts: A num_pareto_samples x num_fantasies x num_pareto_points x M`
            -dim Tensor containing the sampled pareto fronts.
        partitioning: A `BoxDecomposition` module that is used to obtain the
            hyper-rectangle bounds for integration. In the unconstrained case,
            this gives the partition of the dominated space. In the constrained
            case, this gives the partition of the feasible dominated space union
            the infeasible space.
        maximize: If true the box-decomposition is computed assuming maximization.
        num_constraints: the number of constraints.

    Returns:
        A `num_pareto_samples x num_fantasies x 2 x J x (M + K)`-dim Tensor
        containing the box bounds.
    """
    num_pareto_samples = pareto_fronts.shape[0]
    num_fantasies = pareto_fronts.shape[-3]
    M = pareto_fronts.shape[-1]
    K = num_constraints
    ref_point = (torch.ones(M) * NEG_INF).to(pareto_fronts)
    weight = 1.0 if maximize else -1.0

    if M == 1:
        # only consider a Pareto front with one element
        extreme_values = torch.max(weight * pareto_fronts, dim=-2).values
        ref_point = weight * ref_point.expand(extreme_values.shape)

        if maximize:
            hypercell_bounds = torch.stack(
                [ref_point, extreme_values], axis=-2
            ).unsqueeze(-1)
        else:
            hypercell_bounds = torch.stack(
                [extreme_values, ref_point], axis=-2
            ).unsqueeze(-1)
    else:
        box_bounds = []
        num_boxes = []

        # iterate through the samples and compute the box decompositions
        for i in range(num_pareto_samples):
            for j in range(num_fantasies):
                # Dominated partitioning assumes maximization
                # If minimizing we consider negative the Pareto front
                box_bounds_ij = partitioning(
                    ref_point, weight * pareto_fronts[i, j, :, :]
                ).hypercell_bounds

                # reverse the transformation
                if not maximize:
                    box_bounds_ij = weight * torch.flip(box_bounds_ij, dims=[0])

                num_boxes = num_boxes + [box_bounds_ij.shape[-2]]
                box_bounds = box_bounds + [box_bounds_ij]

        # create a Tensor containing to contain the padded box bounds
        hypercell_bounds = torch.ones(
            (num_pareto_samples, num_fantasies, 2, max(num_boxes), M)
        ) * NEG_INF

        for i in range(num_pareto_samples):
            for j in range(num_fantasies):
                box_bounds_ij = box_bounds[i * num_fantasies + j]
                num_boxes_ij = num_boxes[i * num_fantasies + j]
                hypercell_bounds[i, j, :, 0:num_boxes_ij, :] = box_bounds_ij

    # add extra constraint dimension and extra box
    if K > 0:
        # `num_pareto_samples x num_fantasies x 2 x (J - 1) x K`
        feasible_boxes = torch.zeros(hypercell_bounds.shape[:-1] + torch.Size([K]))
        feasible_boxes[..., 0, :, :] = NEG_INF
        # `num_pareto_samples x num_fantasies x 2 x (J - 1) x (M + K)`
        hypercell_bounds = torch.cat([hypercell_bounds, feasible_boxes], dim=-1)

        # `num_pareto_samples x num_fantasies x 2 x 1 x (M + K)`
        infeasible_box = torch.zeros(
            hypercell_bounds.shape[:-2] + torch.Size([1, M + K])
        )
        infeasible_box[..., 1, :, :] = -NEG_INF
        # `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
        hypercell_bounds = torch.cat([hypercell_bounds, infeasible_box], dim=-2)

    # `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
    return hypercell_bounds


def _compute_entropy_noiseless(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        variance_plus_noise: Tensor,
) -> Tensor:
    r"""Computes the entropy estimate at the design points `X` assuming noiseless
    observations.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
            -dim Tensor containing the box decomposition bounds, where
            `J = max(num_boxes)`.
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        variance_plus_noise: A `batch_shape x num_pareto_samples x num_fantasies x 1
            x (M + K)`-dim Tensor containing the posterior variance at X including
            observation noise.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the given
        design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    # standardize the box decomposition bounds and compute normal quantities
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M + K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # compute the differences between the upper and lower terms
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]

    # compute W
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    # compute the sum of ratios
    ratios = .5 * (Wj * (Vjm / Wjm)) / W
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    ratio_term = torch.sum(ratios, dim=(-2, -1), keepdims=True)

    # compute the logarithm of the variance
    log_term = .5 * torch.log(variance_plus_noise).sum(-1, keepdims=True)

    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    log_term = log_term + torch.log(W)

    # additional constant term
    M_plus_K = mean.shape[-1]
    add_term = .5 * M_plus_K * (1 + torch.log(torch.ones(1) * 2 * pi))

    # `batch_shape x num_pareto_samples x num_fantasies`
    entropy = add_term + (log_term - ratio_term).squeeze(-1).squeeze(-1)

    return entropy.mean(-2)


def _compute_entropy_noiseless_upper_bound(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        initial_variance: Tensor,
        initial_variance_plus_noise: Tensor
) -> Tensor:
    r"""Computes the lower bound entropy estimate at the design points `X` assuming
    noiseless observations.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
            -dim Tensor containing the box decomposition bounds, where
            `J = max(num_boxes)`.
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        initial_variance: A `batch_shape x num_pareto_samples x num_fantasies x 1
            x (M + K)`-dim Tensor containing the initial variance (before any
            conditioning) at X excluding observation noise.
        initial_variance_plus_noise: A `batch_shape x num_pareto_samples x
            num_fantasies x 1 x (M + K)`-dim Tensor containing the initial variance
            (before any conditioning) at X including observation noise.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the given
        design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    # standardize the box decomposition bounds and compute normal quantities
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M + K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)

    # compute the differences between the upper and lower terms
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)

    # compute W
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    # compute the logarithm of the variance
    log_term = .5 * torch.log(initial_variance_plus_noise).sum(-1, keepdims=True)

    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    log_term = log_term + torch.log(W)

    # additional constant term
    M_plus_K = mean.shape[-1]
    add_term = .5 * M_plus_K * (1 + torch.log(torch.ones(1) * 2 * pi))

    # `batch_shape x num_pareto_samples x num_fantasies`
    entropy = add_term + log_term.squeeze(-1).squeeze(-1)

    return entropy.mean(-2)


def _compute_entropy_upper_bound(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        variance_plus_noise: Tensor,
        only_diagonal: Optional[bool] = False,
) -> Tensor:
    r"""Computes the entropy upper bound at the design points `X`.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M + K)`
            -dim Tensor containing the box decomposition bounds, where
            `J` = max(num_boxes).
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        variance_plus_noise: A `batch_shape x num_pareto_samples x num_fantasies
            x 1 x (M + K)`-dim Tensor containing the posterior variance at X
            including observation noise.
        only_diagonal: If true we only compute the diagonal elements of the variance.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the
        given design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    # standardize the box decomposition bounds and compute normal quantities
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M + K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # compute the differences between the upper and lower terms
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)
    Vjm = g_times_gpdf[..., 1, :, :] - g_times_gpdf[..., 0, :, :]
    Gjm = gpdf[..., 1, :, :] - gpdf[..., 0, :, :]

    # compute W
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    Cjm = Gjm / Wjm

    # first moment
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)
    mom1 = mean - torch.sqrt(variance) * (Cjm * Wj / W).sum(-2, keepdims=True)
    # diagonal weighted sum
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)
    diag_weighted_sum = (Wj * variance * Vjm / Wjm / W).sum(-2, keepdims=True)

    if only_diagonal:
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
        mean_squared = mean * mean
        cross_sum = - 2 * (
                mean * torch.sqrt(variance) * Cjm * Wj / W
        ).sum(-2, keepdims=True)
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M + K)`
        mom2 = variance_plus_noise - diag_weighted_sum + cross_sum + mean_squared
        var = (mom2 - mom1 * mom1).clamp_min(CLAMP_LB)

        # `batch_shape x num_pareto_samples x num_fantasies
        log_det_term = .5 * torch.log(var).sum(dim=-1).squeeze(-1)
    else:
        # first moment x first moment
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K) x (M+K)
        cross_mom1 = torch.einsum('...i,...j->...ij', mom1, mom1)

        # second moment
        # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K) x (M+K)
        # firstly compute the general terms
        mom2_cross1 = - torch.einsum(
            '...i,...j->...ij', mean, torch.sqrt(variance) * Cjm
        )
        mom2_cross2 = - torch.einsum(
            '...i,...j->...ji', mean, torch.sqrt(variance) * Cjm
        )
        mom2_mean_squared = torch.einsum('...i,...j->...ij', mean, mean)

        mom2_weighted_sum = (
            (mom2_cross1 + mom2_cross2) * Wj.unsqueeze(-1) / W.unsqueeze(-1)
        ).sum(-3, keepdims=True)
        mom2_weighted_sum = mom2_weighted_sum + mom2_mean_squared

        # compute the additional off-diagonal terms
        mom2_off_diag = torch.einsum(
            '...i,...j->...ij', torch.sqrt(variance) * Cjm, torch.sqrt(variance) * Cjm
        )
        mom2_off_diag_sum = (
                mom2_off_diag * Wj.unsqueeze(-1) / W.unsqueeze(-1)
        ).sum(-3, keepdims=True)

        # compute the diagonal terms and subtract the diagonal computed before
        init_diag = torch.diagonal(mom2_off_diag_sum, dim1=-2, dim2=-1)
        diag_weighted_sum = torch.diag_embed(
            variance_plus_noise - diag_weighted_sum - init_diag
        )
        mom2 = mom2_weighted_sum + mom2_off_diag_sum + diag_weighted_sum
        # compute the variance
        var = (mom2 - cross_mom1).squeeze(-3)

        # jitter the diagonal
        # jitter is probably not needed here at all
        jitter_diag = 1e-6 * torch.diag_embed(torch.ones(var.shape[:-1]))
        # logdet is computed using gpytorch implementation
        log_det_term = .5 * logdet(var + jitter_diag)

    # additional terms
    M_plus_K = mean.shape[-1]
    add_term = .5 * M_plus_K * (1 + torch.log(torch.ones(1) * 2 * pi))

    # `batch_shape x num_pareto_samples x num_fantasies
    entropy = add_term + log_det_term
    return entropy.mean(-2)


def _compute_entropy_monte_carlo(
        hypercell_bounds: Tensor,
        mean: Tensor,
        variance: Tensor,
        variance_plus_noise: Tensor,
        samples: Tensor,
        samples_log_prob: Tensor,
) -> Tensor:
    r"""Computes the Monte Carlo entropy at the design points `X`.

    `num_fantasies = 1` for non-fantasized models.

    Args:
        hypercell_bounds: A `num_pareto_samples x num_fantasies x 2 x J x (M+K)`-dim
            Tensor containing the box decomposition bounds, where
            `J` = max(num_boxes).
        mean: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)`-dim
            Tensor containing the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)`
            -dim Tensor containing the posterior variance at X excluding observation
            noise.
        variance_plus_noise: A `batch_shape x num_pareto_samples x num_fantasies x 1
            x (M+K)`-dim Tensor containing the posterior variance at X including
            observation noise.
        samples: A `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies
            x 1 x (M+K)`-dim Tensor containing the noisy samples at `X` from the
            posterior conditioned on the Pareto optimal points.
        samples_log_prob:  A `num_mc_samples x batch_shape x num_pareto_samples
            num_fantasies`-dim Tensor containing the log probability densities
            of the samples.

    Returns:
        A `batch_shape x num_fantasies`-dim Tensor of entropy estimate at the given
        design points `X` (`num_fantasies=1` for non-fantasized models).
    """
    ####################################################################
    # standardize the box decomposition bounds and compute normal quantities
    # `batch_shape x num_pareto_samples x num_fantasies x 2 x J x (M+K)`
    g = (hypercell_bounds - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2))
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x (M+K)`
    rho = torch.sqrt(variance / variance_plus_noise)

    # compute the initial normal quantities
    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)

    # compute the differences between the upper and lower terms
    Wjm = (gcdf[..., 1, :, :] - gcdf[..., 0, :, :]).clamp_min(CLAMP_LB)

    # compute W
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    # `batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    ####################################################################
    g = g.unsqueeze(0)
    rho = rho.unsqueeze(0).unsqueeze(-2)
    # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 1 x 1 x
    # (M+K)`
    z = ((samples - mean) / torch.sqrt(variance_plus_noise)).unsqueeze(-2)
    # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 2 x J x
    # (M+K)`
    # clamping here is important because `1 - rho^2 = 0` at an input where
    # observation noise is zero
    g_new = (g - rho * z) / torch.sqrt((1 - rho * rho).clamp_min(CLAMP_LB))

    # compute the initial normal quantities
    normal_new = Normal(torch.zeros_like(g_new), torch.ones_like(g_new))
    gcdf_new = normal_new.cdf(g_new)

    # compute the differences between the upper and lower terms
    Wjm_new = (gcdf_new[..., 1, :, :] - gcdf_new[..., 0, :, :]).clamp_min(CLAMP_LB)

    # compute W+
    Wj_new = torch.exp(torch.sum(torch.log(Wjm_new), dim=-1, keepdims=True))
    # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 1 x 1`
    W_new = torch.sum(Wj_new, dim=-2, keepdims=True).clamp_max(1.0)

    ####################################################################
    # W_ratio = W+ / W
    W_ratio = torch.exp(torch.log(W_new) - torch.log(W).unsqueeze(0))
    samples_log_prob = samples_log_prob.unsqueeze(-1).unsqueeze(-1)

    # compute the Monte Carlo average: - E[W_ratio * log(W+ p(y))] + log(W)
    log_term = torch.log(W_new) + samples_log_prob
    mc_estimate = - (W_ratio * log_term).mean(0)
    # `batch_shape x num_pareto_samples x num_fantasies
    entropy = (mc_estimate + torch.log(W)).squeeze(-1).squeeze(-1)

    # alternative Monte Carlo estimate: - E[W_ratio * log(W_ratio p(y))]
    # log_term = torch.log(W_ratio) + samples_log_prob
    # mc_estimate = - (W_ratio * log_term).mean(0)
    # # `batch_shape x num_pareto_samples x num_fantasies
    # entropy = mc_estimate.squeeze(-1).squeeze(-1)

    return entropy.mean(-2)
