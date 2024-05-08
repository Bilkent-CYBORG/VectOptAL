from abc import ABC, abstractmethod
from typing import Any, Tuple

from vectoptal.models import Model

import torch
import numpy as np


class AcquisitionStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self):
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class SumVarianceAcquisition(AcquisitionStrategy):
    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        _, variances = self.model.predict(x)
        return np.sum(np.diagonal(variances, axis1=-2, axis2=-1), axis=-1)

def optimize_acqf_discrete(
    acq: AcquisitionStrategy,
    q: int,
    choices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_list, acq_value_list = [], []

    chosen = 0
    while chosen < q:
        with torch.no_grad():
            acq_values = acq(choices)

        best_idx = np.argmax(acq_values)
        candidate_list.append(choices[best_idx])
        acq_value_list.append(acq_values[best_idx])

        choices = np.concatenate(
            [choices[:best_idx], choices[best_idx+1:]]
        )

        chosen += 1
    
    return np.stack(candidate_list, axis=-2), np.array(acq_value_list)
