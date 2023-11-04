#!/usr/bin/env python3

r"""
The Marine problem was adapted from the implementation by Tanabe:
https://github.com/ryojitanabe/reproblems.
"""

import torch
from torch import Tensor
from botorch.test_functions.base import (
    MultiObjectiveTestProblem,
)


class MarineDesign(MultiObjectiveTestProblem):
    r"""Conceptual marine design.

    Adapted from https://github.com/ryojitanabe/reproblems
    """
    dim = 6
    num_objectives = 4
    _bounds = [
        (150.0, 274.32),
        (20.0, 32.31),
        (13.0, 25.0),
        (10.0, 11.71),
        (14.0, 18.0),
        (0.63, 0.75),
    ]
    _ref_point = [-250.0, 20000.0, 25000.0, 15.0]
    _num_original_constraints = 9

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        batch_size = X[..., 0].size()
        f = torch.zeros(batch_size + torch.Size([self.num_objectives]))
        constraintFuncs = torch.zeros(
            batch_size + torch.Size([self._num_original_constraints])
        )
        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / torch.pow(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (torch.pow(displacement, 2.0 / 3.0) *
                 torch.pow(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * torch.pow(x_L, 0.8) * \
                        torch.pow(x_B, 0.6) * \
                        torch.pow(x_D, 0.3) * \
                        torch.pow(x_CB, 0.1)
        steel_weight = 0.034 * torch.pow(x_L, 1.7) * \
                       torch.pow(x_B, 0.7) * \
                       torch.pow(x_D, 0.4) * \
                       torch.pow(x_CB, 0.5)
        machinery_weight = 0.17 * torch.pow(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
                (2000.0 * torch.pow(steel_weight, 0.85))
                + (3500.0 * outfit_weight)
                + (2400.0 * torch.pow(power, 0.8))
        )
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * torch.pow(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * torch.pow(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * torch.pow(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[..., 0] = annual_costs / annual_cargo
        f[..., 1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[..., 2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[..., 0] = (x_L / x_B) - 6.0
        constraintFuncs[..., 1] = -(x_L / x_D) + 15.0
        constraintFuncs[..., 2] = -(x_L / x_T) + 19.0
        constraintFuncs[..., 3] = 0.45 * torch.pow(DWT, 0.31) - x_T
        constraintFuncs[..., 4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[..., 5] = 500000.0 - DWT
        constraintFuncs[..., 6] = DWT - 3000.0
        constraintFuncs[..., 7] = 0.32 - Fn

        constraintFuncs = torch.where(
            constraintFuncs < 0, -constraintFuncs,
            torch.zeros(constraintFuncs.size())
        )
        f[..., 3] = torch.sum(constraintFuncs, dim=-1)

        return f
