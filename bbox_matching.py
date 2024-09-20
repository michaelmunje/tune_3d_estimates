from abc import ABC, abstractmethod
from typing import List, Dict
from structures import LocationWith2DBBox, LocationWith3dBBox
import json

class BBoxMatching(ABC):
    @abstractmethod
    def matching(self, 
                 location_estimates: List[LocationWith2DBBox], 
                 bbox3d_labels: List[LocationWith3dBBox]) -> Dict[str, str]:
        pass
    
class ManualBBoxMatching(BBoxMatching):
    def __init__(self, matching_filepath: str):
        self.matching_filepath = matching_filepath # json dict of form {tracking_id: coda_tracking_id}
        self.matching_dict = {}
        with open(self.matching_filepath, 'r') as f:
            self.matching_dict = json.load(f)

    def matching(self, 
                 location_estimates: List[LocationWith2DBBox], 
                 bbox3d_labels: List[LocationWith3dBBox]) -> Dict[str, str]:
        return self.matching_dict