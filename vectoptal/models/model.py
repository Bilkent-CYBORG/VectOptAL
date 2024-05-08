from abc import ABC, abstractmethod


class Model(ABC):
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
