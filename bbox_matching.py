from abc import ABC, abstractmethod
from typing import List, Dict
from utils import Bbox2d, Bbox3d

class BBoxMatching(ABC):
    @abstractmethod
    def matching(self, 
                 bbox2d_list: List[Bbox2d], 
                 bbox3d_list: List[Bbox3d], 
                 bbox2d_labels: List[Bbox2d], 
                 bbox3d_labels: List[Bbox3d], 
                 tracking_ids: List[int]) -> Dict[int, int]:
        pass