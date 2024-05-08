import os
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from vectoptal.confidence_region import (
    RectangularConfidenceRegion, EllipsoidalConfidenceRegion,
)
from vectoptal.models import Model


class DesignSpace(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def update(self, model: Model):
        pass

class DiscreteDesignSpace(DesignSpace):
    def __init__(self, points, objective_dim, confidence_type='hyperrectangle') -> None:
        super().__init__()

        if confidence_type == 'hyperrectangle':
            confidence_cls = RectangularConfidenceRegion
        elif confidence_type == 'hyperellipsoid':
            confidence_cls = EllipsoidalConfidenceRegion
        else:
            raise NotImplementedError

        self.cardinality = len(points)

        self.points = points
        self.confidence_regions = []
        for _ in range(len(points)):
            self.confidence_regions.append(confidence_cls(objective_dim))

    def update(self, model: Model, scale: np.ndarray, indices_to_update: Optional[list]=None):
        if indices_to_update is None:
            indices_to_update = list(range(self.cardinality))

        mus, covs = model.predict(self.points[indices_to_update])
        for pt_i, mu, cov in zip(indices_to_update, mus, covs):
            self.confidence_regions[pt_i].update(mu, cov, scale)

class AdaptivelyDiscretizedDesignSpace(DesignSpace):
    def __init__(self) -> None:
        super().__init__()

    def update(self, model: Model):
        pass
