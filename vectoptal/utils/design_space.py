from abc import ABC, abstractmethod

import numpy as np


class DesignSpace(ABC):
    def __init__(self) -> None:
        pass

class DiscreteDesignSpace(DesignSpace):
    def __init__(self, X) -> None:
        super().__init__()

        self.cardinality = len(X)
