import os

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Dataset:
    _in_dim: int
    _out_dim: int
    _cardinality: int

    def __init__(self):
        assert (
            self._cardinality == self.in_data.shape[0]
            and self.in_data.shape[0] == self.out_data.shape[0]
        ), "Cardinality mismatch"

        # Standardize
        input_scaler = MinMaxScaler()
        self.in_data = input_scaler.fit_transform(self.in_data)
        self.in_dim = self.in_data.shape[1]

        output_scaler = StandardScaler(with_mean=True, with_std=True)
        self.out_data = output_scaler.fit_transform(self.out_data)

        self.out_dim = self.out_data.shape[1]


def get_dataset_instance(dataset_name: str) -> Dataset:
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
    _in_dim = 5
    _out_dim = 3
    _cardinality = 500

    def __init__(self):
        datafile = os.path.join("data", "vehicle_safety", "VehicleSafety.npy")
        data = np.load(datafile, allow_pickle=True)
        self.out_data = np.copy(data[:, self._in_dim :])
        self.in_data = np.copy(data[:, : self._in_dim])

        super().__init__()
