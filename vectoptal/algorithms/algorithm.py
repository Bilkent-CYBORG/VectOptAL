from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run_one_step(self):
        pass


class PALAlgorithm(Algorithm, ABC):
    """
    (\\epsilon, \\delta)-PAC modelling.
    """

    def __init__(self, epsilon, delta) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.delta = delta

        # TODO: Maybe make sample_count an ABC property. Will every child have it?

    @abstractmethod
    def run_one_step(self):
        pass
