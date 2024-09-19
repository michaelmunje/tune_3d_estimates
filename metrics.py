from typing import List, Dict
from utils import LocationWith2DBBox, compute_2d_iou

def compute_bbox_errors(self, bbox2d_list: List[LocationWith2DBBox], 
                        bbox2d_labels: List[LocationWith2DBBox],
                        matched_ids: Dict[int, int]) -> List[float]:
    """
    Compute the errors between estimated and ground truth bounding boxes.

    Args:
        bbox2d_list (List[Location]): List of estimated 2D bounding boxes.
        bbox2d_labels (List[Location]): List of ground truth 2D bounding boxes.
        matched_ids (Dict[int, int]): Dictionary mapping estimated bbox ids to ground truth ids.

    Returns:
        Tuple[List[float], List[float]]: Lists of 2D and 3D errors for matched bounding boxes.
    """
    errors_2d = []

    for estimated_id, label_id in matched_ids.items():
        # Compute 2D IoU
        bbox2d_estimated = bbox2d_list[estimated_id]
        bbox2d_label = bbox2d_labels[label_id]
        iou_2d = compute_2d_iou(bbox2d_estimated, bbox2d_label)
        errors_2d.append(1 - iou_2d)  # Error is 1 - IoU

    return errors_2d