#!/usr/bin/env python3

r"""
Multi-objective benchmark problems from chemistry.
"""

from typing import Optional
import torch
from torch import Tensor
from botorch.test_functions.base import (
    MultiObjectiveTestProblem,
)
import numpy as np
from scipy.integrate import solve_ivp


class SnAr(MultiObjectiveTestProblem):
    r"""A two objective optimization problem for a nucleophilic aromatic substitution
    (SnAr) reaction.

    Design space `x = (tau, equiv_pldn, conc_dfnb, temperature)`:
        - `tau` is the residence time in minutes.
        - `equiv_pldn` is the equivalents of pyrrolidine.
        - `conc_dfnb` is the concentration of 2,4 dinitrofluorobenenze at
            reactor inlet (after mixing) in M.
        - `temperature` is the reactor temperature in degrees celsius.

    Objective `min(-log(sty), log(e_factor))`:
        - `sty` is the space-time yield measured in kg/m^3/h.
        - `e_factor` is the environmental factor.

    This implementation is adapted from
    https://github.com/sustainable-processes/summit/blob/master/summit/benchmarks/snar.py
    """

    dim = 4
    num_objectives = 2
    _bounds = [(0.5, 2.0), (1.0, 5.0), (0.1, 0.5), (30, 120)]
    #_ref_point = [0, 120]
    _ref_point = [-5.5, 5]
    # Molecular weights (g/mol)
    molecular_weights = [159.09, 71.12, 210.21, 210.21, 261.33]
    # g/mL (should adjust to temp, but just using @ 25C)
    rho_ethanol = 0.789
    # Avogadro kJ/K/mol
    R = 8.314 / 1000
    # Absolute zero
    absolute_zero = 273.15
    # Reservoir concentration of 1 is 1 M = 1 mM
    C1_0 = 2.0
    # Reservoir concentration of  2 is 2 M = 2 mM
    C2_0 = 4.2
    # volume
    V = 5

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for SnAr.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        equiv_pldn = X_reshaped[:, 1]
        conc_dfnb = X_reshaped[:, 2]
        temperature = X_reshaped[:, 3]

        # concentrations
        C_i = torch.zeros(B, 5)
        C_i[:, 0] = conc_dfnb
        C_i[:, 1] = equiv_pldn * conc_dfnb

        # Flow rates
        q_tot = cls.V / tau
        # Flow rate of 1 (dfnb)
        q_1 = C_i[:, 0] / cls.C1_0 * q_tot
        # Flow rate of 2 (pldn)
        q_2 = C_i[:, 1] / cls.C2_0 * q_tot
        # Flow rate of ethanol [This quantity is not used]
        q_ethanol = q_tot - q_1 - q_2

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 5))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                C = concentration
                T = temperature + cls.absolute_zero
                T_ref = 90 + cls.absolute_zero
                # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
                k = (
                    lambda k_ref, E_a, temp:
                    0.6 * k_ref * torch.exp(-E_a / cls.R * (1 / temp - 1 / T_ref))
                )

                k_a = k(57.9, 33.3, T)
                k_b = k(2.70, 35.3, T)
                k_c = k(0.865, 38.9, T)
                k_d = k(1.63, 44.8, T)

                # Reaction Rates
                r = torch.zeros(5)
                # Set to reactants when close
                for i in [0, 1]:
                    C[i] = 0 if C[i] < 1e-6 * C_i[b][i] else C[i]

                r[0] = -(k_a + k_b) * C[0] * C[1]
                r[1] = -(k_a + k_b) * C[0] * C[1] \
                       - k_c * C[1] * C[2] \
                       - k_d * C[1] * C[3]
                r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
                r[3] = k_a * C[0] * C[1] - k_d * C[1] * C[3]
                r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]

                return r

            res_b = solve_ivp(
                _integrand_b, [0, tau[b]], C_i[b], args=(temperature[b],)
            )
            C_final[b, :] = res_b.y[:, -1]

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)

        # Calculate STY and E-factor
        # Convert to kg m^-3 h^-1
        sty = 6e4 / 1000 * cls.molecular_weights[2] * C_final[:, 2] * q_tot / cls.V
        sty = sty.clamp_min(1e-6)

        term_2 = 1e-3 * sum(
            [cls.molecular_weights[i] * C_final[:, i] * q_tot
             for i in range(5) if i != 2]
        )
        # Set to a large value if no product formed
        mask = torch.isclose(C_final[:, 2], torch.zeros(B, dtype=torch.double))

        e_factor = torch.where(
            mask,
            1e3 * torch.ones(B, dtype=torch.double),
            (q_tot * cls.rho_ethanol + term_2) /
            (1e-3 * cls.molecular_weights[2] * C_final[:, 2] * q_tot)
        )
        obj_values = torch.column_stack([-torch.log(sty), torch.log(e_factor)])

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)


class VdV(MultiObjectiveTestProblem):
    r"""A two objective optimization problem for a Van de Vusse reaction (VdV).

    Design space `x = (tau, temperature)`:
        - `tau` is the residence time.
        - `temperature` is the reactor temperature in degrees celsius.

    Objective `min(-log(product_yield), -log(sty))`:
        - `e_factor` is the product yield.
        - `sty` is the space-time yield.

    This implementation is adapted from the MATLAB code in
    https://github.com/adamc1994/MultiChem
    """

    dim = 2
    num_objectives = 2
    _bounds = [(0.5, 10), (25, 100)]
    _ref_point = [5, 1]

    # Molecular weight
    molecular_weight = 100
    # Avogadro
    R = 8.314
    # Absolute zero
    absolute_zero = 273.15
    # volume
    V = 8
    # discretization
    n = 5
    discrete_v = torch.linspace(0, V, n)
    # initial concentration
    C0 = 1

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for VdV.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        temperature = X_reshaped[:, 1]

        # concentration
        C_i = torch.zeros(B, cls.n * 4)
        C_i[:, 0] = cls.C0

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 4))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                flow_rate = cls.V / tau[b]
                conc = concentration[0:cls.n*4].reshape(4, cls.n)
                T = temperature + cls.absolute_zero

                k = (
                    lambda k_ref, E_a: k_ref * torch.exp(-E_a / (cls.R * T))
                )
                diff = lambda vec: vec[1:] - vec[0:-1]

                k_a = k(2.1450e10, 8.1131e4)
                k_b = k(2.1450e10, 8.1131e4)
                k_c = k(0.0151e10, 7.1168e4)

                A = conc[0, :]
                B = conc[1, :]
                C = conc[2, :]
                D = conc[3, :]

                dAdt = [
                    torch.zeros(1),
                    - flow_rate * diff(A) / diff(cls.discrete_v)
                    - k_a * A[1:] - k_c * A[1:] ** 2
                ]
                dAdt = torch.cat(dAdt)

                dBdt = [
                    torch.zeros(1),
                    - flow_rate * diff(B) / diff(cls.discrete_v)
                    + k_a * A[1:] - k_b * B[1:]
                ]
                dBdt = torch.cat(dBdt)

                dCdt = [
                    torch.zeros(1),
                    - flow_rate * diff(C) / diff(cls.discrete_v)
                    + k_b * B[1:]
                ]
                dCdt = torch.cat(dCdt)

                dDdt = [
                    torch.zeros(1),
                    - flow_rate * diff(D) / diff(cls.discrete_v)
                    + k_c * A[1:] ** 2
                ]
                dDdt = torch.cat(dDdt)

                return torch.cat([dAdt, dBdt, dCdt, dDdt])

            res_b = solve_ivp(
                _integrand_b, [0, 4*tau[b]], C_i[b], args=(temperature[b],)
            )

            C_final[b, :] = torch.tensor(
                [res_b.y[(i+1)*cls.n - 1, -1] for i in range(4)],
                dtype=torch.double
            )
        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)
        product_yield = C_final[:, 1] * 100 / cls.C0
        sty = 60 * C_final[:, 1] * cls.molecular_weight / tau
        obj_values = torch.column_stack(
            [-torch.log(product_yield), -torch.log(sty)]
        )

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)


class PK1(MultiObjectiveTestProblem):
    r"""A two objective optimization problem for the Paal-Knorr reaction (PK) between
    2,5-hexanedione and ethanolamine.

    Design space `x = (tau, equivalents)`:
        - `tau` is the residence time.
        - `equivalents` is the equivalents of 3.25 (???).
        - `temperature` is defaulted to 50 degrees Celsius.

    Objective `min(-log(sty), -log(rme))`:
        - `sty` is the space-time yield.
        - `rme` is the reaction mass efficiency.

    This implementation is adapted from the MATLAB code in
    https://github.com/adamc1994/MultiChem
    """

    dim = 2
    num_objectives = 2
    _bounds = [(0.5, 2), (1, 10)]
    _ref_point = [-4.5, 0.8]

    # Molecular weight
    molecular_weight = 139.20
    # Avogadro
    R = 8.314
    # Absolute zero
    absolute_zero = 273.15
    # volume
    V = 8
    # discretization
    n = 5
    discrete_v = torch.linspace(0, V, n)
    # initial concentration
    C0 = 1
    # temperature
    temperature = 50.0

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for PK1.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        equivalents = X_reshaped[:, 1]

        # concentration
        C_i = torch.zeros(B, cls.n * 4)
        C_i[:, 0] = cls.C0
        C_i[:, cls.n] = cls.C0 * equivalents

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 4))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                flow_rate = cls.V / tau[b]
                conc = concentration[0:cls.n*4].reshape(4, cls.n)
                T = temperature + cls.absolute_zero

                k = (
                    lambda k_ref, E_a: k_ref * np.exp(-E_a / (cls.R * T))
                )
                diff = lambda vec: vec[1:] - vec[0:-1]

                k_a = k(15.40, 12.2e3)
                k_b = k(405.19, 20.0e3)

                A = conc[0, :]
                B = conc[1, :]
                C = conc[2, :]
                D = conc[3, :]

                dAdt = [
                    torch.zeros(1),
                    - flow_rate * diff(A) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:]
                ]
                dAdt = torch.cat(dAdt)

                dBdt = [
                    torch.zeros(1),
                    - flow_rate * diff(B) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:]
                ]
                dBdt = torch.cat(dBdt)

                dCdt = [
                    torch.zeros(1),
                    - flow_rate * diff(C) / diff(cls.discrete_v)
                    + k_a * A[1:] * B[1:] - k_b * C[1:]
                ]
                dCdt = torch.cat(dCdt)

                dDdt = [
                    torch.zeros(1),
                    - flow_rate * diff(D) / diff(cls.discrete_v)
                    + k_b * C[1:]
                ]
                dDdt = torch.cat(dDdt)

                return torch.cat([dAdt, dBdt, dCdt, dDdt])

            res_b = solve_ivp(
                _integrand_b, [0, 4*tau[b]], C_i[b], args=(cls.temperature,)
            )

            C_final[b, :] = torch.tensor(
                [res_b.y[(i+1)*cls.n - 1, -1] for i in range(4)],
                dtype=torch.double
            )

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)
        sty = 60 * C_final[:, 3] * cls.molecular_weight / tau

        product = C_final[:, 3] * 100 / cls.C0
        rme = (139.20 * product) / (
                114.14 + (61.08 * equivalents)
        )
        obj_values = torch.column_stack(
            [-torch.log(sty), -torch.log(rme)]
        )

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)


class PK2(MultiObjectiveTestProblem):
    r"""A two objective optimization problem for the Paal-Knorr reaction (PK) between
    2,5-hexanedione and ethanolamine.

    Design space `x = (tau, temperature, equivalents)`:
        - `tau` is the residence time.
        - `temperature` is the temperature.
        - `equivalents` is the equivalents of 3.25 (???).

    Objective `min(log(yield), -log(sty))`:
        - `yield` is the yield of 3.26 %.
        - `sty` is the space-time yield.

    This implementation is adapted from the MATLAB code in
    https://github.com/adamc1994/MultiChem
    """

    dim = 3
    num_objectives = 2
    _bounds = [(0.5, 2), (25, 150), (1, 10)]
    _ref_point = [4.5, -3]

    # Molecular weight
    molecular_weight = 139.20
    # Avogadro
    R = 8.314
    # Absolute zero
    absolute_zero = 273.15
    # volume
    V = 8
    # discretization
    n = 5
    discrete_v = torch.linspace(0, V, n)
    # initial concentration
    C0 = 1

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for PK2.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        temperature = X_reshaped[:, 1]
        equivalents = X_reshaped[:, 2]

        # concentration
        C_i = torch.zeros(B, cls.n * 4)
        C_i[:, 0] = cls.C0
        C_i[:, cls.n] = cls.C0 * equivalents

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 4))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                flow_rate = cls.V / tau[b]
                conc = concentration[0:cls.n*4].reshape(4, cls.n)
                T = temperature + cls.absolute_zero

                k = (
                    lambda k_ref, E_a: k_ref * np.exp(-E_a / (cls.R * T))
                )
                diff = lambda vec: vec[1:] - vec[0:-1]

                k_a = k(15.40, 12.2e3)
                k_b = k(405.19, 20.0e3)

                A = conc[0, :]
                B = conc[1, :]
                C = conc[2, :]
                D = conc[3, :]

                dAdt = [
                    torch.zeros(1),
                    - flow_rate * diff(A) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:]
                ]
                dAdt = torch.cat(dAdt)

                dBdt = [
                    torch.zeros(1),
                    - flow_rate * diff(B) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:]
                ]
                dBdt = torch.cat(dBdt)

                dCdt = [
                    torch.zeros(1),
                    - flow_rate * diff(C) / diff(cls.discrete_v)
                    + k_a * A[1:] * B[1:] - k_b * C[1:]
                ]
                dCdt = torch.cat(dCdt)

                dDdt = [
                    torch.zeros(1),
                    - flow_rate * diff(D) / diff(cls.discrete_v)
                    + k_b * C[1:]
                ]
                dDdt = torch.cat(dDdt)

                return torch.cat([dAdt, dBdt, dCdt, dDdt])

            res_b = solve_ivp(
                _integrand_b, [0, 4*tau[b]], C_i[b], args=(temperature[b],)
            )

            C_final[b, :] = torch.tensor(
                [res_b.y[(i+1)*cls.n - 1, -1] for i in range(4)],
                dtype=torch.double
            )

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)
        sty = 60 * C_final[:, 3] * cls.molecular_weight / tau

        # product = C_final[:, 3] * 100
        # rme = (139.20 * product) / (
        #         114.14 + (61.08 * equivalents)
        # )

        intermediate_yield = C_final[:, 2] * 100 / cls.C0

        obj_values = torch.column_stack(
            [torch.log(intermediate_yield), -torch.log(sty)]
        )

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)


class Lactose(MultiObjectiveTestProblem):
    r"""A two objective optimization problem for the isomerisation of lactose to
    lactulose.

    Design space `x = (tau, temperature)`:
        - `tau` is the residence time.
        - `temperature` is the temperature.

    Objective `min(-log(lactulose_yield), -log(galactose_yield))`:
        - `lactulose_yield` is the yield of Lactulose 3.22 %.
        - `galactose_yield` is the yield of Galactose 3.23 %.

    This implementation is adapted from the MATLAB code in
    https://github.com/adamc1994/MultiChem
    """

    dim = 2
    num_objectives = 2
    _bounds = [(0.5, 10), (25, 100)]
    _ref_point = [4, 5]

    # Molecular weight
    molecular_weight = 342.30
    # Avogadro
    R = 8.314
    # Absolute zero
    absolute_zero = 273.15
    # volume
    V = 8
    # discretization
    n = 5
    discrete_v = torch.linspace(0, V, n)
    # initial concentration
    C0 = 1

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for Lactose.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        temperature = X_reshaped[:, 1]

        # concentration
        C_i = torch.zeros(B, cls.n * 3)
        C_i[:, 0] = cls.C0

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 3))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                flow_rate = cls.V / tau[b]
                conc = concentration[0:cls.n*3].reshape(3, cls.n)
                T = temperature + cls.absolute_zero

                k = (
                    lambda k_ref, E_a: k_ref * np.exp(-E_a / (cls.R * T))
                )
                diff = lambda vec: vec[1:] - vec[0:-1]
                k_a = k(9.5e14, 105.1e3)
                k_b = k(7.0e24, 174.0e3)
                k_c = k(4.0e7, 54.9e3)

                A = conc[0, :]
                B = conc[1, :]
                C = conc[2, :]

                dAdt = [
                    torch.zeros(1),
                    - flow_rate * diff(A) / diff(cls.discrete_v)
                    - k_a * A[1:] - k_b * A[1:]
                ]
                dAdt = torch.cat(dAdt)

                dBdt = [
                    torch.zeros(1),
                    - flow_rate * diff(B) / diff(cls.discrete_v)
                    - k_c * B[1:] + k_a * A[1:]
                ]
                dBdt = torch.cat(dBdt)

                dCdt = [
                    torch.zeros(1),
                    - flow_rate * diff(C) / diff(cls.discrete_v)
                    + k_b * A[1:] + k_c * B[1:]
                ]
                dCdt = torch.cat(dCdt)

                return torch.cat([dAdt, dBdt, dCdt])

            res_b = solve_ivp(
                _integrand_b, [0, 4*tau[b]], C_i[b], args=(temperature[b],)
            )

            C_final[b, :] = torch.tensor(
                [res_b.y[(i+1)*cls.n - 1, -1] for i in range(3)],
                dtype=torch.double
            )

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)

        product_yield = C_final[:, 1] * 100 / cls.C0
        side_yield = C_final[:, 2] * 100 / cls.C0

        obj_values = torch.column_stack(
            [-torch.log(product_yield), torch.log(side_yield)]
        )

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)


class SnAr1(MultiObjectiveTestProblem):
    r"""A three objective optimization problem for the nucleophilic aromatic
    substitution between 2,4-difluoronitrobenzene and morpholine (SnAr).

    Design space `x = (tau, temperature)`:
        - `tau` is the residence time.
        - `temperature` is the temperature.

    Objective `min(-log(yield_1), log(yield_2), log(yield_3))`:
        - `yield_1` is the yield of 3.16 %.
        - `yield_2` is the yield of 3.19 %.
        - `yield_3` is the yield of 3.20 %.

    This implementation is adapted from the MATLAB code in
    https://github.com/adamc1994/MultiChem
    """

    dim = 2
    num_objectives = 3
    _bounds = [(0.5, 20), (60, 140)]
    _ref_point = [-2.8, 2.5, 4.5]

    # Molecular weight
    molecular_weight = 226.21
    # Avogadro
    R = 8.314
    # Absolute zero
    absolute_zero = 273.15
    # volume
    V = 8
    # discretization
    n = 5
    discrete_v = torch.linspace(0, V, n)
    # initial concentration
    C0 = 1
    C1 = 3

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for SnAr1.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        temperature = X_reshaped[:, 1]

        # concentration
        C_i = torch.zeros(B, cls.n * 5)
        C_i[:, 0] = cls.C0
        C_i[:, cls.n] = cls.C1

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 5))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                flow_rate = cls.V / tau[b]
                conc = concentration[0:cls.n*5].reshape(5, cls.n)
                T = temperature + cls.absolute_zero

                k = (
                    lambda k_ref, E_a: k_ref * np.exp(-E_a / (cls.R * T))
                )
                diff = lambda vec: vec[1:] - vec[0:-1]

                k_a = k(1.5597e6, 4.32e4)
                k_b = k(13.9049e3, 3.53e4)
                k_c = k(10.4046e3, 4.08e4)
                k_d = k(370.3652e6, 6.89e4)

                A = conc[0, :]
                B = conc[1, :]
                C = conc[2, :]
                D = conc[3, :]
                E = conc[4, :]

                dAdt = [
                    torch.zeros(1),
                    - flow_rate * diff(A) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:] - k_b * A[1:] * B[1:]
                ]
                dAdt = torch.cat(dAdt)

                dBdt = [
                    torch.zeros(1),
                    - flow_rate * diff(B) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:]
                    - k_b * A[1:] * B[1:]
                    - k_c * B[1:] * C[1:]
                    - k_d * B[1:] * D[1:]
                ]
                dBdt = torch.cat(dBdt)

                dCdt = [
                    torch.zeros(1),
                    - flow_rate * diff(C) / diff(cls.discrete_v)
                    + k_a * A[1:] * B[1:]
                    - k_c * B[1:] * C[1:]
                ]
                dCdt = torch.cat(dCdt)

                dDdt = [
                    torch.zeros(1),
                    - flow_rate * diff(D) / diff(cls.discrete_v)
                    + k_b * A[1:] * B[1:]
                    - k_d * B[1:] * D[1:]
                ]
                dDdt = torch.cat(dDdt)

                dEdt = [
                    torch.zeros(1),
                    - flow_rate * diff(E) / diff(cls.discrete_v)
                    + k_c * B[1:] * C[1:]
                    + k_d * B[1:] * D[1:]
                ]
                dEdt = torch.cat(dEdt)

                return torch.cat([dAdt, dBdt, dCdt, dDdt, dEdt])

            res_b = solve_ivp(
                _integrand_b, [0, 4*tau[b]], C_i[b], args=(temperature[b],)
            )

            C_final[b, :] = torch.tensor(
                [res_b.y[(i+1)*cls.n - 1, -1] for i in range(5)],
                dtype=torch.double
            )

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)

        product_yield = C_final[:, 2] * 100 / cls.C0
        side1_yield = C_final[:, 3] * 100 / cls.C0
        side2_yield = C_final[:, 4] * 100 / cls.C0

        obj_values = torch.column_stack(
            [-torch.log(product_yield),
             torch.log(side1_yield),
             torch.log(side2_yield)]
        )

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)


class SnAr2(MultiObjectiveTestProblem):
    r"""A three objective optimization problem for the nucleophilic aromatic
    substitution between 2,4-difluoronitrobenzene and morpholine (SnAr).

    Design space `x = (tau, temperature, equivalents_a, equivalents_b)`:
        - `tau` is the residence time.
        - `temperature` is the temperature.
        - `equivalents_a` is the equivalents of 3.16 (???).
        - `equivalents_b` is the equivalents of 3.17 (???).

    Objective `min(log(yield), -log(rme), -log(sty))`:
        - `yield` is the yield of 3.19 %.
        - `rme` is the reaction mass efficiency.
        - `sty` is the space-time yield.

    This implementation is adapted from the MATLAB code in
    https://github.com/adamc1994/MultiChem
    """

    dim = 4
    num_objectives = 3
    _bounds = [(0.5, 20), (60, 140), (0.1, 2), (2, 5)]
    _ref_point = [5, 1]

    # Molecular weight
    molecular_weight = 226.21
    # Avogadro
    R = 8.314
    # Absolute zero
    absolute_zero = 273.15
    # volume
    V = 8
    # discretization
    n = 5
    discrete_v = torch.linspace(0, V, n)

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Constructor for SnAr2.
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)

    @classmethod
    def solve_ode(cls, X) -> Tensor:
        """
        Args:
            X: A `batch_shape x d`-dim Tensor containing the designs.

        Returns:
            A `batch_shape x M`-dim Tensor containing the objectives.

        """
        # We consider transforming X into `B x d`-dim Tensor and then performing `B`
        # function evaluations. Certain computations are executed in parallel when
        # simple.
        batch_shape = X.shape[:-1]
        B = torch.prod(torch.tensor(batch_shape))
        new_shape = torch.Size([B]) + torch.Size([cls.dim])
        X_reshaped = X.reshape(new_shape)

        # extract design variables
        tau = X_reshaped[:, 0]
        temperature = X_reshaped[:, 1]
        equivalents_a = X_reshaped[:, 2]
        equivalents_b = X_reshaped[:, 3]

        # concentration
        C_i = torch.zeros(B, cls.n * 5)
        C_i[:, 0] = equivalents_a
        C_i[:, cls.n] = equivalents_a * equivalents_b

        # Integration step
        # Ideally we should integrate in parallel, but scipy.stats.solve_ivp
        # does not have this feature.
        C_final = np.zeros(shape=(B, 5))
        for b in range(B):
            def _integrand_b(t, concentration, temperature):
                flow_rate = cls.V / tau[b]
                conc = concentration[0:cls.n*5].reshape(5, cls.n)
                T = temperature + cls.absolute_zero

                k = (
                    lambda k_ref, E_a: k_ref * np.exp(-E_a / (cls.R * T))
                )
                diff = lambda vec: vec[1:] - vec[0:-1]

                k_a = k(1.5597e6, 4.32e4)
                k_b = k(13.9049e3, 3.53e4)
                k_c = k(10.4046e3, 4.08e4)
                k_d = k(370.3652e6, 6.89e4)

                A = conc[0, :]
                B = conc[1, :]
                C = conc[2, :]
                D = conc[3, :]
                E = conc[4, :]

                dAdt = [
                    torch.zeros(1),
                    - flow_rate * diff(A) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:] - k_b * A[1:] * B[1:]
                ]
                dAdt = torch.cat(dAdt)

                dBdt = [
                    torch.zeros(1),
                    - flow_rate * diff(B) / diff(cls.discrete_v)
                    - k_a * A[1:] * B[1:]
                    - k_b * A[1:] * B[1:]
                    - k_c * B[1:] * C[1:]
                    - k_d * B[1:] * D[1:]
                ]
                dBdt = torch.cat(dBdt)

                dCdt = [
                    torch.zeros(1),
                    - flow_rate * diff(C) / diff(cls.discrete_v)
                    + k_a * A[1:] * B[1:]
                    - k_c * B[1:] * C[1:]
                ]
                dCdt = torch.cat(dCdt)

                dDdt = [
                    torch.zeros(1),
                    - flow_rate * diff(D) / diff(cls.discrete_v)
                    + k_b * A[1:] * B[1:]
                    - k_d * B[1:] * D[1:]
                ]
                dDdt = torch.cat(dDdt)

                dEdt = [
                    torch.zeros(1),
                    - flow_rate * diff(E) / diff(cls.discrete_v)
                    + k_c * B[1:] * C[1:]
                    + k_d * B[1:] * D[1:]
                ]
                dEdt = torch.cat(dEdt)

                return torch.cat([dAdt, dBdt, dCdt, dDdt, dEdt])

            res_b = solve_ivp(
                _integrand_b, [0, 4*tau[b]], C_i[b], args=(temperature[b],)
            )

            C_final[b, :] = torch.tensor(
                [res_b.y[(i+1)*cls.n - 1, -1] for i in range(5)],
                dtype=torch.double
            )

        # Convert numpy array to tensor
        C_final = torch.tensor(C_final, dtype=torch.double)
        side_yield = C_final[:, 3] * 100 / equivalents_a
        sty = 60 * C_final[:, 2] * cls.molecular_weight / tau

        product = C_final[:, 2] * 100 / equivalents_a
        rme = (226.21 * product) / (
                159.09 + (87.12 * equivalents_b)
        )

        obj_values = torch.column_stack(
            [torch.log(side_yield),
             -torch.log(rme),
             -torch.log(sty)]
        )

        return obj_values.reshape(batch_shape + torch.Size([cls.num_objectives]))

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.solve_ode(X)
