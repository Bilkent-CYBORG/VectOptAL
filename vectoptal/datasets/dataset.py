"""
Datasets for maximization problems.

References:

.. [Liao2008]
    Liao, Li, Yang, Zhang, Li.
    Multiobjective optimization for crash safety design of vehicles using stepwise regression model.
    Structural and Multidisciplinary Optimization, 2008.

.. [Tanabe2020]
    Tanabe, Ishibuchi.
    An easy-to-use real-world multi-objective optimization problem suite.
    Applied Soft Computing, 2020.

.. [Zuluaga2012]
    Zuluaga, Milder, Püschel.
    Computer generation of streaming sorting networks.
    Design Automation Conference, 2012.
"""

import os
from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Dataset(ABC):
    """
    Abstract base class for datasets that handles min-max scaling of input and standardization of
    output. Any class inheriting from this class should implement the following properties:
    - _in_dim: int
    - _out_dim: int
    - _cardinality: int
    """

    @property
    @abstractmethod
    def _in_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def _out_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def _cardinality(self) -> int:
        pass

    def __init__(self):
        if (
            self._cardinality != self.in_data.shape[0]
            or self.in_data.shape[0] != self.out_data.shape[0]
        ):
            raise ValueError("Cardinality mismatch.")

        # Standardize
        input_scaler = MinMaxScaler()
        self.in_data = input_scaler.fit_transform(self.in_data)
        self.in_dim = self.in_data.shape[1]

        output_scaler = StandardScaler(with_mean=True, with_std=True)
        self.out_data = output_scaler.fit_transform(self.out_data)

        self.out_dim = self.out_data.shape[1]


def get_dataset_instance(dataset_name: str) -> Dataset:
    """
    Returns an instance of the dataset class corresponding to the given dataset name. If the
    dataset name is not recognized, a ValueError is raised.

    :param dataset_name: Name of the dataset class to be instantiated.
    :type dataset_name: str
    :return: Instance of the dataset class.
    :rtype: Dataset
    """
    if dataset_name in globals():
        return globals()[dataset_name]()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


class Test(Dataset):
    """
    A miniature DiskBrake dataset variant for using in testing.
    """

    _in_dim = 4
    _out_dim = 2
    _cardinality = 32

    def __init__(self):
        datafile = os.path.join("data", "test", "test.npy")
        data = np.load(datafile, allow_pickle=True)
        self.out_data = np.copy(data[:, self._in_dim :])
        self.in_data = np.copy(data[:, : self._in_dim])

        super().__init__()


class SNW(Dataset):
    """
    Dataset for optimizing sorting network configurations in computational hardware design.
    The reward vector represents the trade-off between throughput and hardware area. The area is
    negated to maximize it. See [Zuluaga2012]_.
    """

    _in_dim = 3
    _out_dim = 2
    _cardinality = 206

    def __init__(self):
        datafile = os.path.join("data", "snw", "sort_256.csv")
        data = np.genfromtxt(datafile, delimiter=";")
        self.out_data = np.copy(data[:, self._in_dim :])
        self.in_data = np.copy(data[:, : self._in_dim])

        # Negate first objective to maximize
        self.out_data[:, 0] = -self.out_data[:, 0]

        super().__init__()


class DiskBrake(Dataset):
    """
    Disc brake optimization balancing mass and stopping time. Based on [Tanabe2020]_.
    """

    _in_dim = 4
    _out_dim = 2
    _cardinality = 128

    def __init__(self):
        datafile = os.path.join("data", "brake", "brake.npy")
        data = np.load(datafile, allow_pickle=True)
        self.out_data = np.copy(data[:, self._in_dim :])
        self.in_data = np.copy(data[:, : self._in_dim])

        super().__init__()


class VehicleSafety(Dataset):
    """
    Vehicle structure optimization dataset for enhancing crashworthiness. The reward vector
    includes weight, acceleration, and toe-board intrusion. See [Liao2008]_ and [Tanabe2020]_.
    """

    _in_dim = 5
    _out_dim = 3
    _cardinality = 500

    def __init__(self):
        datafile = os.path.join("data", "vehicle_safety", "VehicleSafety.npy")
        data = np.load(datafile, allow_pickle=True)
        self.out_data = np.copy(data[:, self._in_dim :])
        self.in_data = np.copy(data[:, : self._in_dim])

        super().__init__()
