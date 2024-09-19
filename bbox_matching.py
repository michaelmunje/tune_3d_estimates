from abc import ABC, abstractmethod
from typing import List, Dict
from utils import Location, Bbox3d
import json

class BBoxMatching(ABC):
    @abstractmethod
    def matching(self, 
                 location_estimates: List[Location], 
                 bbox2d_labels: List[Location], 
                 bbox3d_labels: List[Bbox3d]) -> Dict[int, int]:
        pass
    
class ManualBBoxMatching(BBoxMatching):
    def __init__(self, matching_filepath: str):
        self.matching_filepath = matching_filepath # json dict of form {tracking_id: coda_tracking_id}
        self.matching_dict = {}
        with open(self.matching_filepath, 'r') as f:
            self.matching_dict = json.load(f)

    def matching(self, 
                 location_estimates: List[Location], 
                 bbox2d_labels: List[Location], 
                 bbox3d_labels: List[Bbox3d]) -> Dict[int, int]:
        return self.matching_dict