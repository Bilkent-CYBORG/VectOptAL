#!/usr/bin/env python3

r"""
Load the approximated Pareto set and front for the benchmark problems.
"""
from typing import Tuple
import torch
from torch import Tensor
from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()


def get_pareto(problem: str) -> Tuple[Tensor, Tensor]:
    r"""Fit the Gaussian process model.

    Args:
        problem: The problem name.

    Returns:
        ps: A `P x d`-dim Tensor containing the Pareto set.
        pf: A `P x M`-dim Tensor containing the Pareto front.
    """
    ps_title = problem + "_ps.pt"
    pf_title = problem + "_pf.pt"
    ps = torch.load(PACKAGEDIR / "pareto" / ps_title)
    pf = torch.load(PACKAGEDIR / "pareto" / pf_title)
    return ps, pf
