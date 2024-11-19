from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class Model(ABC):
    """
    Defines the abstract model class.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add_sample(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def predict(self, test_X):
        pass


class ModelList(Model):
    """
    Defines the abstract model list class.
    """

    def __init__(self) -> None:
        super().__init__()


class UncertaintyPredictiveModel(Model):
    """
    Defines the abstract uncertainty predictive model class, which gives mean and variance
    estimates as predictions.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, test_X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        pass


class GPModel(UncertaintyPredictiveModel):
    """
    Defines the abstract Gaussian process model (GP) class.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_lengthscale_and_var(self):
        """
        Get the lengthscale and variance of the GP model.
        """
        raise NotImplementedError

    def get_kernel_type(self):
        """
        Get the kernel type of the GP model.
        """
        # TODO: Define an enum for kernel type.
        raise NotImplementedError
