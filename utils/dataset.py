import os

import numpy as np
import gpytorch.kernels
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.kernels import CategoricalKernel
from utils.utils import get_bigmij, get_smallmij, get_delta, get_cone_params


DATASET_SIZES = {
    "SNW": 206,
    "SINE": 1000,
    "JAHS": 785,
    "Brake": 128,
    "Brake500": 500,
    "BC": 250,
    "BC500": 500,
    "PK2": 500,
    "PK2Correlated": 500,
    "PK2Correlated2": 500,
    "PK2K": 2000,
    "PK1": 500,
    "PK12K": 500,
    "VdV": 500,
    "VdV2K": 2000,
    "SnAr": 2000,
    "SnArHALF": 2000,
    "VehicleSafety": 500,
    "VehicleSafety2K": 2000,
    "Marine": 500,
    "Marine2K": 2000,
    "Penicillin": 500,
    "BrakeBO": 500,
    "Lactose": 250,
    "CarSideImpact": 500,
    "WeldedBeam": 500,
}


DATASET_REF_PTS = {
    "SNW": 206,
    "SINE": 1000,
    "JAHS": 785,
    "Brake": [5.7771, 3.9651],
    "Brake500": [5.7771, 3.9651],
    "BC": 250,
    "BC500": 500,
    "PK2": 500,
    "VdV": 500,
    "SnAr": 2000,
    "VehicleSafety": [1864.72022, 11.81993945, 0.2903999384],
    "Marine": 500,
    "Penicillin": [1.85, 86.93, 514.70],
    "BrakeBO": [5.7771, 3.9651],
    "Lactose": 250,
    "CarSideImpact": [45.4872, 4.5114, 13.3394, 10.3942],
    "WeldedBeam": [40, 0.015],
    "PK2K": 2000,
}


class Dataset:
    def __init__(self, cone_degree):
        # Standardize
        # input_scale = True
        # input_scaler = StandardScaler(with_mean=input_scale, with_std=input_scale)
        input_scaler = MinMaxScaler()
        self.in_data = input_scaler.fit_transform(self.in_data)
        self.in_dim = len(self.in_data[0])

        output_scale = True
        output_scaler = StandardScaler(with_mean=output_scale, with_std=output_scale)
        self.out_data = output_scaler.fit_transform(self.out_data)
        self.out_dim = len(self.out_data[0])

        self.cone_degree = cone_degree
        self.W, self.alpha_vec, _ = get_cone_params(self.cone_degree, dim=self.out_dim)

        self.pareto_indices = None
        self.pareto = None
        self.delta = None

        self.f1label=r'$f_1$'
        self.f2label=r'$f_2$'

        self.delta = get_delta(self.out_data, self.W, self.alpha_vec)
    
    def set_pareto_indices(self):
        self.find_pareto()
        print(f"For cone degree {self.cone_degree}, the pareto set indices are:")
        print(self.pareto_indices)

    def find_pareto(self):
        """
        Find the indices of Pareto designs (rows of out_data)
        :param mu: An (n_points, D) array
        :param W: (n_constraint,D) ndarray
        :param alpha_vec: (n_constraint,1) ndarray of alphas of W
        :return: An array of indices of pareto-efficient points.
        """
        out_data = self.out_data.copy()
        
        n_points = out_data.shape[0]
        is_efficient = np.arange(out_data.shape[0])
        
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(out_data):
            nondominated_point_mask = np.zeros(out_data.shape[0], dtype=bool)
            vj = out_data[next_point_index].reshape(-1,1)
            for i in range(len(out_data)):
                vi = out_data[i].reshape(-1,1)
                nondominated_point_mask[i] = (
                    (get_smallmij(vi, vj, self.W, self.alpha_vec) == 0)
                    and (get_bigmij(vi, vj, self.W) > 0)
                )
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            out_data = out_data[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        
        self.pareto_indices = is_efficient

    def get_params(self):
        return self.delta, self.pareto_indices

class SNW(Dataset):
    def __init__(self, cone_degree):
        datafile = os.path.join('data', 'snw', 'sort_256.csv')
        designs = np.genfromtxt(datafile, delimiter=';')
        self.out_data = np.copy(designs[:,3:])
        self.out_data[:,0] = -self.out_data[:,0]
        self.in_data = np.copy(designs[:,:3])

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]
    
    def set_pareto_indices(self):
        if self.cone_degree not in [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                2,   3,   4,   5,   6,   7,   8,  10,  11,  12,  14,  18,  27,  28, 29, 30, 32, 33,
                34,  36,  37,  38,  39,  40,  42,  43,  44,  45,  46,  49,  50,  59,  61,  63,  64,
                66,  67,  80,  81,  96, 112, 128, 153, 154, 155, 160, 161, 162, 167, 168, 174, 187
            ]
        elif self.cone_degree == 60:
            self.pareto_indices =  [
                2,   3,   4,   5,   6,   7,   8,  10,  11,  12,  14,  18,  27,  28,  29,  30,  32,  33,
                36,  37,  38,  39,  40,  42,  43,  44,  45,  46,  49,  61,  63,  64,  66,  80,  81, 128,
                153, 154, 160, 161, 162, 167, 168, 174, 187,
            ]
        elif self.cone_degree == 90:
            self.pareto_indices = [
                2,   3,   4,   5,   6,   7,   8,  10,  11,  12,  14,  28,  29, 30, 32, 38, 40,  42,
                43,  45,  63, 160, 161, 167, 168, 174
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = [  2,  4,  6,  7,  8, 10, 11, 12,  14, 29, 160, 167, 168, 174  ]
        elif self.cone_degree == 135:
            self.pareto_indices = [  2,   4,   6,   7,   8,  10,  12,  14, 160, 167  ]

class SINE(Dataset):
    def __init__(self, cone_degree):
        self.in_data = np.load(os.path.join('data', 'sine', 'sinex.npy'), allow_pickle=True)
        self.out_data = np.load(os.path.join('data', 'sine', 'siney.npy'), allow_pickle=True)

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]
    
    def set_pareto_indices(self):
        if self.cone_degree not in [45, 90, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                22,  27,  30,  32,  66,  72,  98, 108, 110, 115, 126, 132, 139, 142, 145, 163, 164, 167,
                198, 203, 236, 256, 258, 273, 300, 303, 311, 314, 323, 326, 334, 337, 357, 369, 386, 389,
                391, 402, 416, 429, 431, 443, 448, 453, 458, 468, 470, 474, 477, 482, 493, 512, 518, 520,
                523, 536, 556, 576, 593, 608, 616, 620, 637, 643, 657, 662, 669, 672, 673, 675, 679, 680,
                684, 689, 712, 732, 733, 734, 739, 744, 758, 786, 804, 810, 819, 825, 835, 840, 857, 874,
                877, 883, 887, 895, 905, 921, 922, 948, 956, 957, 975, 979, 983, 993
            ]
        elif self.cone_degree == 90:
            self.pareto_indices = [
                22,  27,  66,  72,  98, 115, 126, 142, 145, 163, 167, 198, 236, 256, 273, 300, 303, 314,
                323, 337, 357, 391, 402, 429, 443, 458, 468, 474, 477, 482, 518, 523, 556, 576, 608, 620,
                643, 657, 662, 669, 672, 675, 679, 689, 712, 732, 733, 734, 739, 758, 786, 804, 810, 819,
                840, 857, 874, 877, 883, 887, 895, 921, 948, 956, 975, 979, 983
            ]
        elif self.cone_degree == 135:
            self.pareto_indices = [
                22,  72,  98, 126, 142, 163, 167, 198, 273, 300, 314, 337, 391, 402, 429, 443, 477, 482,
                518, 523, 556, 576, 608, 669, 672, 675, 689, 733, 734, 758, 786, 804, 810, 819, 840, 887,
                895, 921, 956, 979, 983
            ]

class JAHS(Dataset):
    def __init__(self, cone_degree):
        self.in_data = np.load(os.path.join('data', 'jahs', 'jahsx.npy'), allow_pickle=True)
        self.out_data = np.load(os.path.join('data', 'jahs', 'jahsy.npy'), allow_pickle=True)

        super().__init__(cone_degree)

        self.model_kernel = CategoricalKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]
    
    def set_pareto_indices(self):
        if self.cone_degree not in [45, 90, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                0,  25,  53,  86, 150, 153, 160, 166, 167, 208, 223, 242, 307, 309, 312, 328, 341, 345,
                356, 361, 362, 375, 399, 401, 403, 433, 481, 517, 519, 546, 560, 572, 620
            ]
        elif self.cone_degree == 90:
            self.pareto_indices = [  0,  53, 150, 153, 166, 312, 361, 362, 375, 519  ]
        elif self.cone_degree == 135:
            self.pareto_indices = [  150, 312  ]

class Brake(Dataset):
    def __init__(self, cone_degree):
        data = np.load(os.path.join('data', 'brake', 'brake.npy'), allow_pickle=True)
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                15, 25, 27, 38, 45, 46, 48, 53, 62, 70, 78, 81, 85, 88, 94, 106, 118, 121, 126
            ]
        elif self.cone_degree == 60:
            self.pareto_indices = [ 15, 25, 38, 45, 46, 53, 62, 70, 78, 88, 106, 118, 121, 126 ]
        elif self.cone_degree == 90:
            self.pareto_indices = [ 25, 53, 62, 78, 118, 121, 126 ]
        elif self.cone_degree == 120:
            self.pareto_indices = [ 25, 62, 118, 126 ]
        elif self.cone_degree == 135:
            self.pareto_indices = [ 62, 118, 126 ]

class Brake500(Dataset):
    def __init__(self, cone_degree):
        data = np.load(os.path.join('data', 'brake', 'brake500.npy'), allow_pickle=True)
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [ 13, 78, 126, 158, 198, 235, 254, 478, 495, ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class BC(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'branin_currin', 'branin_currin.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in []:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = []
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class BC500(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'branin_currin', 'bc500.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [ 3, 30, 83, 115, 131, 163, 227, 242 ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class BrakeBO(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'botorch', 'DiscBrake.npy'), allow_pickle=True
        )
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [ 12, 77, 125, 157, 197, 234, 253, 477, 494 ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class PK2(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'PK2.npy'), allow_pickle=True
        )
        self.in_data = data[:, :3]
        self.out_data = data[:, 3:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [118, 138, 204, 234, 264, 288, 331, 347, 388, 450, 456, 494, 499]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class PK2Correlated(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'PK2.npy'), allow_pickle=True
        )
        self.in_data = data[:, :3]
        self.out_data = data[:, 3:]

        # Add a new objective that is the sum of previous ones.
        self.out_data = np.hstack([self.out_data, np.sum(self.out_data, axis=1).reshape(-1, 1)])

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [118, 138, 204, 234, 264, 288, 331, 347, 388, 450, 456, 494, 499]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class PK2Correlated2(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'PK2.npy'), allow_pickle=True
        )
        self.in_data = data[:, :3]
        self.out_data = data[:, 3:]

        # Add a new objective that is the sum of previous ones.
        self.out_data = np.hstack([
            (2*self.out_data[:, 0] + 3*self.out_data[:, 1]).reshape(-1, 1),
            (1*self.out_data[:, 0] + 5*self.out_data[:, 1]).reshape(-1, 1),
            (4*self.out_data[:, 0] + 2*self.out_data[:, 1]).reshape(-1, 1),
        ])

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [288, 388, 456, 494]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []


class PK1(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'PK1.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                40, 61, 104, 168, 189, 232, 253, 264, 296, 360, 424, 445, 456, 488,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class PK12K(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'PK12K.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                40,  125,  168,  296,  424,  509,  552,  680,  808,  893,  936, 1064,
                1192, 1213, 1320, 1448, 1533, 1576, 1661, 1704, 1832, 1960,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class VdV(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'VdV.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                48, 82, 84, 114, 118, 152, 186, 220, 254, 288, 358, 388, 426, 456, 494
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class VdV2K(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'VdV2K.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                80,   86,  252,  290,  420,  496,  590,  600, 1070, 1100, 1114, 1204,
                1280, 1310, 1456, 1508, 1576, 1610, 1628, 1662, 1748, 1782, 1970,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class SnAr(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'SnAr.npy'), allow_pickle=True
        )
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [ 389, 545, 867, 1362, 1403, 1650, 1739 ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class SnArHALF(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'SnArHALF.npy'), allow_pickle=True
        )
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [121, 171, 211, 301,]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class VehicleSafety(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'botorch', 'VehicleSafety.npy'), allow_pickle=True
        )
        self.in_data = data[:, :5]
        self.out_data = data[:, 5:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                23,  43, 118, 127, 138, 159, 163, 170, 187, 192, 219, 235, 252, 259,
                264, 274, 286, 307, 314, 347, 370, 398, 401, 403, 420, 431, 491,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class VehicleSafety2K(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'botorch', 'VehicleSafety2K.npy'), allow_pickle=True
        )
        self.in_data = data[:, :5]
        self.out_data = data[:, 5:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                12,   26,   95,   99,  222,  234,  290,  374,  402,  435,  502,
                523,  562,  563, 590,  643,  690,  710,  720,  875,  978, 1043, 1154,
                1219, 1280, 1318, 1354, 1395, 1490, 1508, 1514, 1611, 1650, 1690, 1696,
                1731, 1890, 1902, 1938, 1940, 1971, 1999,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class Penicillin(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'botorch', 'Penicillin.npy'), allow_pickle=True
        )
        self.in_data = data[:, :7]
        self.out_data = data[:, 7:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                11,  28,  32,  42,  80,  84,  85,  88, 117, 130, 140, 146, 150, 170,
                172, 231, 234, 244, 250, 253, 293, 296, 302, 308, 334, 349, 362, 369,
                394, 396, 408, 420, 434, 446, 456, 468, 482,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class Marine(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'marine', 'marine.npy'), allow_pickle=True
        )
        self.in_data = data[:, :6]
        self.out_data = data[:, 6:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                4,  11,  18,  35,  51,  55,  67,  74,  94,  98, 107, 120, 127, 139, 151,
                163, 179, 189, 218, 225, 230, 243, 250, 259, 307, 315, 322, 328, 342, 343,
                352, 362, 371, 374, 378, 387, 403, 405, 411, 427, 447, 475, 482, 491,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class Marine2K(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'marine', 'marine2k.npy'), allow_pickle=True
        )
        self.in_data = data[:, :6]
        self.out_data = data[:, 6:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                19,  111,  123,  156,  208,  236,  287,  331,  340,  419,  435,  522,
                555,  591, 611,  631,  643,  650,  708,  756,  787,  839,  907,  940, 960,
                1018, 1040, 1071, 1083, 1099, 1158, 1231, 1266, 1347, 1356, 1373, 1395, 1470, 1483,
                1550, 1555, 1581, 1601, 1659, 1666, 1697, 1731, 1777, 1795, 1875, 1882,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class Lactose(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'Lactose.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                3,   9,  11,  15,  23,  26,  27,  33,  34,  35,  43,  47,  50,  54,  55,
                59,  63,  66, 67,  74,  75,  79,  81,  87,  91,  95,  99, 103, 106, 107, 109,
                110, 115, 121, 122, 123, 129, 130, 134, 135, 139, 143, 147, 151, 154, 155,
                158, 159, 163, 171, 175, 182, 183, 187, 191, 193, 194, 195, 201, 202, 203,
                207, 209, 211, 215, 218, 219, 225, 226, 227, 230, 231, 233, 235, 239, 241, 245, 246, 247, 
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class CarSideImpact(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'botorch', 'CarSideImpact.npy'), allow_pickle=True
        )
        self.in_data = data[:, :7]
        self.out_data = data[:, 7:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                1,   3,   6,   7,  14,  18,  19,  23,  26,  27,  34,  36,  38,  41,  43,
                46,  48,  51,  52,  58,  60,  67,  74,  78,  79,  81,  82,  84,  89,  94,
                102, 107, 114, 115, 122, 126, 130, 132, 133, 138, 140, 143, 144, 145, 148,
                152, 154, 156, 158, 162, 163, 166, 175, 177, 182, 187, 196, 200, 205, 210,
                214, 217, 218, 219, 226, 228, 234, 242, 243, 248, 250, 255, 257, 259, 266,
                270, 275, 279, 282, 287, 290, 294, 298, 300, 302, 303, 308, 310, 315, 316,
                320, 321, 322, 327, 329, 330, 331, 338, 342, 346, 347, 356, 359, 360, 365,
                370, 371, 372, 373, 374, 376, 385, 387, 394, 398, 399, 402, 403, 406, 407,
                410, 414, 415, 418, 428, 431, 432, 448, 450, 454, 455, 457, 459, 462, 473,
                474, 475, 480, 482, 486, 487, 491, 494, 495, 498, 499,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class WeldedBeam(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'botorch', 'WeldedBeam.npy'), allow_pickle=True
        )
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                59, 99, 123, 207, 211, 234, 247, 291, 307, 323, 337, 339,
                367, 370, 403, 451,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []

class PK2K(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'jes_chem', 'PK2K.npy'), allow_pickle=True
        )
        self.in_data = data[:, :3]
        self.out_data = data[:, 3:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                112,  154,  252,  394,  470,  556,  600,  760,  934, 1027, 1204, 1250,
                1280, 1331, 1456, 1499, 1508, 1542, 1696, 1730, 1748, 1810, 1932,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []
