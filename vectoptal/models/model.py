from abc import ABC, abstractmethod


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
    def __init__(self) -> None:
        super().__init__()

class GPModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def get_lengthscale_and_var(self):
        raise NotImplementedError

    def get_kernel_type(self):
        # TODO: Define an enum for kernel type.
        raise NotImplementedError
