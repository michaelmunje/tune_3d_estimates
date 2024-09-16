from abc import ABC, abstractmethod
from typing import List, Tuple
from utils import Sample, Bbox2d, Bbox3d

class BBoxEstimation(ABC):
    @abstractmethod
    def estimate(self, samples: List[Sample]) -> Tuple[List[Bbox2d], List[Bbox3d]]:
        pass