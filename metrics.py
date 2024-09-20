from typing import List, Dict
from structures import LocationWith2DBBox, compute_2d_iou
from structures import Trajectory, get_bev_pose_wrt_initial_pose
import numpy as np

def compute_bbox_errors(bbox2d_list: List[LocationWith2DBBox], 
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

def average_displacement_error(traj: Trajectory, reference_frame_traj: Trajectory) -> float:
    # check each corresponding timestep
    total_error = 0.0
    n_timesteps_in_both = 0
    for timestep in traj.corresponding_timesteps:
        if timestep in reference_frame_traj.corresponding_timesteps:
            total_error += np.linalg.norm(traj.get_pose_at_timestep(timestep).get_position_np() - reference_frame_traj.get_pose_at_timestep(timestep).get_position_np())
            n_timesteps_in_both += 1
    if n_timesteps_in_both == 0:
        return 0.0
    return total_error / n_timesteps_in_both

def final_displacement_error(traj: Trajectory, reference_frame_traj: Trajectory) -> float:
    return np.linalg.norm(traj.bev_poses[-1].get_position_np() - reference_frame_traj.bev_poses[-1].get_position_np())

def angular_displacement_error(traj: Trajectory, reference_frame_traj: Trajectory) -> float:
    total_error = 0.0
    n_timesteps_in_both = 0
    for timestep in traj.corresponding_timesteps:
        if timestep in reference_frame_traj.corresponding_timesteps:
            traj_yaw = traj.get_pose_at_timestep(timestep).yaw
            ref_yaw = reference_frame_traj.get_pose_at_timestep(timestep).yaw
            yaw_diff = np.abs(traj_yaw - ref_yaw)
            total_error += yaw_diff
            n_timesteps_in_both += 1
    if n_timesteps_in_both == 0:
        return 0.0
    return total_error / n_timesteps_in_both

def heading_deviation_error(traj: Trajectory, reference_frame_traj: Trajectory) -> float:
    total_error = 0.0
    n_timesteps_in_both = 0
    for timestep in traj.corresponding_timesteps:
        if timestep in reference_frame_traj.corresponding_timesteps:
            traj_yaw = traj.get_pose_at_timestep(timestep).yaw
            ref_yaw = reference_frame_traj.get_pose_at_timestep(timestep).yaw
            yaw_diff = np.abs(traj_yaw - ref_yaw)
            total_error += yaw_diff
            n_timesteps_in_both += 1
    if n_timesteps_in_both == 0:
        return 0.0
    return total_error / n_timesteps_in_both

