import torch
import gpytorch

torch.set_default_dtype(torch.float64)


class CategoricalKernel(gpytorch.kernels.Kernel):
    r"""
    A Kernel for categorical features.
    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.
    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = True,
        **kwargs,
    ) -> torch.Tensor:

        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        dists = delta / (self.lengthscale.unsqueeze(-2)*10)

        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)
        else:
            dists = dists.mean(-1)

        res = torch.exp(-dists)

        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)

        return res
