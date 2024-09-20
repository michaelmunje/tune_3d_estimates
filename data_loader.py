import json
import os
import sys
from typing import List, Tuple, Dict, Callable

import matplotlib.pyplot as plt

from structures import Sample, LocationWith3dBBox
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from structures import BEVPose, Trajectory, LocationWith2DBBox

def load_samples(frame_sequences: List[List[int]], frame_idx_to_img_fp: Callable[[int], str], frame_idx_to_json_fp: Callable[[int], str], lart_folder: str) -> List[List[Sample]]:
    samples_sequences = []
    for frame_indices in frame_sequences:
        samples = []
        for frame_idx in frame_indices:
            image_path = frame_idx_to_img_fp(frame_idx)
            json_path = frame_idx_to_json_fp(frame_idx)
            
            assert os.path.exists(image_path), f'Image path {image_path} does not exist'
            assert os.path.exists(json_path), f'JSON path {json_path} does not exist'

            with open(json_path, 'r') as file:
                data = json.load(file)
            objects = data['3dbbox']
            
            objects = [obj for obj in objects if obj['classId'] == 'Pedestrian']
            sample = Sample(rgb_image_filepath=image_path, objects=objects, lart_folder=lart_folder)
            samples.append(sample)
        samples_sequences.append(samples)
    return samples_sequences

def extract_labels(samples: List[Sample]) -> List[List[LocationWith3dBBox]]:
    all_labels: List[List[LocationWith3dBBox]] = []
    for sample in samples:
        labels: List[LocationWith3dBBox] = []
        objects = sample.objects
        for obj in objects:
            labels.append(obj.bbox3d)
        all_labels.append(labels)
    return all_labels

def update_estimates_with_new_timesteps(timesteps_added: List[int], samples: List[Sample], 
                                        loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]],
                                        tracking_id: str):
    for timestep in timesteps_added:
        for sample in samples:
            sample_filepath = sample.rgb_image_filepath
            sample_idx = sample.get_sample_idx()
            if sample_idx == timestep:
                loc_estimates_by_sample[sample_filepath][tracking_id] = LocationWith2DBBox(x=-1.0, y=-1.0, w=-1.0, h=-1.0, 
                                                                                            cX=-1.0, cY=-1.0, cZ=-1.0, 
                                                                                            tracking_id=tracking_id, sample_filepath=sample_filepath)

def update_labels_with_new_timesteps(timesteps_added: List[int], samples: List[Sample], 
                                     loc_labels_by_sample: Dict[str, Dict[str, LocationWith3dBBox]],
                                     coda_tracking_id: str):
    for timestep in timesteps_added:
        for sample in samples:
            sample_filepath = sample.rgb_image_filepath
            sample_idx = sample.get_sample_idx()
            if sample_idx == timestep:
                loc_labels_by_sample[sample_filepath][coda_tracking_id] = LocationWith3dBBox(cX=0.0, cY=0.0, cZ=0.0, 
                                                                                                    w = 0.0, h=0.0, l=0.0, 
                                                                                                    r=0.0, p=0.0, y=0.0,
                                                                                                    sample_filepath=sample_filepath,
                                                                                                    tracking_id=coda_tracking_id)

def get_estimates_and_labels_per_sample(samples_sequences: List[List[Sample]],
                                        estimator_fn, matcher_fn) -> Tuple[Dict[str, Dict[str, LocationWith2DBBox]], Dict[str, Dict[str, LocationWith3dBBox]], Dict[str, str]]:
    # let's first group by sample
    loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]] = {} # key is sample filepath, value is a dict tracking_id -> bev_estimate
    loc_labels_by_sample: Dict[str, Dict[str, LocationWith3dBBox]] = {} # key is sample filepath, value is a dict tracking_id -> bev_label
    matched_ids: Dict[str, str] = {} # tracking id from 2d bbox -> tracking id from 3d bbox
    
    # populate estimates and labels for each sample for each tracking id
    for samples in samples_sequences:
        location_estimates: List[List[LocationWith2DBBox]] = estimator_fn(samples)
        location_labels: List[List[LocationWith3dBBox]] = extract_labels(samples)
        
        # Little hacky, but we only need to do this once
        matched_ids: Dict[str, str] = matcher_fn(location_estimates, location_labels)

        for loc_estimates_for_sample in location_estimates:
            sample_filepath = loc_estimates_for_sample[0].sample_filepath
            if sample_filepath not in loc_estimates_by_sample:
                loc_estimates_by_sample[sample_filepath] = {}
            for loc_estimate in loc_estimates_for_sample:
                if loc_estimate.tracking_id in matched_ids:
                    loc_estimates_by_sample[sample_filepath][loc_estimate.tracking_id] = loc_estimate
            
        for sample in samples:
            sample_filepath = sample.rgb_image_filepath
            if sample_filepath not in loc_labels_by_sample:
                loc_labels_by_sample[sample_filepath] = {}
            for obj in sample.objects:
                loc_labels_by_sample[sample_filepath][obj.id] = obj.bbox3d
                
    return loc_estimates_by_sample, loc_labels_by_sample, matched_ids

def get_object_estimate_trajectories(samples_sequences: List[List[Sample]], 
                                     loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]], 
                                     matched_ids: Dict[str, str],
                                     robot_trajectories: List[Trajectory]) -> List[Dict[str, Dict[str, Trajectory]]]:
    # for each sample sequence, create a trajectory
    seq_object_estimate_trajectories: List[Dict[str, Trajectory]] = [] # (for each sample seq), key is track_id, value is trajectory
    
    for seq_idx in range(len(samples_sequences)):
        samples = samples_sequences[seq_idx]
        possible_timesteps = robot_trajectories[seq_idx].corresponding_timesteps
        tracking_id_to_bev: Dict[str, List[Tuple[int, BEVPose]]] = {}
        for sample in samples:
            sample_filepath = sample.rgb_image_filepath
            sample_idx = sample.get_sample_idx()
            loc_estimates = loc_estimates_by_sample[sample_filepath]
            for tracking_id in loc_estimates.keys():
                if tracking_id in matched_ids:
                    if tracking_id not in tracking_id_to_bev:
                        tracking_id_to_bev[tracking_id] = []
                    tracking_id_to_bev[tracking_id].append((sample_idx, loc_estimates[tracking_id].get_bev()))
        trajectory_by_sample_filepath: Dict[str, Trajectory] = {}
        for tracking_id in tracking_id_to_bev:
            indices_and_bev_poses = tracking_id_to_bev[tracking_id]
            bev_poses = [bev_pose for _, bev_pose in indices_and_bev_poses]
            indices = [idx for idx, _ in indices_and_bev_poses]
            estimate_trajectory = Trajectory(bev_poses, indices, possible_timesteps, id=tracking_id, localize=False, initial_yaw_estimation=True)
            trajectory_by_sample_filepath[tracking_id] = estimate_trajectory
        seq_object_estimate_trajectories.append(trajectory_by_sample_filepath)
    return seq_object_estimate_trajectories

def get_object_label_trajectories(samples_sequences: List[List[Sample]], 
                                  loc_labels_by_sample: Dict[str, Dict[str, LocationWith3dBBox]], 
                                  matched_ids: Dict[str, str],
                                  robot_trajectories: List[Trajectory]) -> List[Dict[str, Dict[str, Trajectory]]]:
    # for each sample sequence, create a trajectory
    seq_object_label_trajectories: List[Dict[str, Trajectory]] = [] # (for each sample seq), key is track_id, value is trajectory
    matched_ids_inv = {v: k for k, v in matched_ids.items()}
    
    for seq_idx in range(len(samples_sequences)):
        samples = samples_sequences[seq_idx]
        possible_timesteps = robot_trajectories[seq_idx].corresponding_timesteps
        tracking_id_to_bev: Dict[str, List[Tuple[int, BEVPose]]] = {}
        for sample in samples:
            sample_filepath = sample.rgb_image_filepath
            sample_idx = sample.get_sample_idx()
            loc_labels = loc_labels_by_sample[sample_filepath]
            for tracking_id in loc_labels.keys():
                if tracking_id in matched_ids_inv:
                    if tracking_id not in tracking_id_to_bev:
                        tracking_id_to_bev[tracking_id] = []
                    tracking_id_to_bev[tracking_id].append((sample_idx, loc_labels[tracking_id].get_bev()))
        trajectory_by_sample_filepath: Dict[str, Trajectory] = {}
        for tracking_id in tracking_id_to_bev:
            indices_and_bev_poses = tracking_id_to_bev[tracking_id]
            bev_poses = [bev_pose for _, bev_pose in indices_and_bev_poses]
            indices = [idx for idx, _ in indices_and_bev_poses]
            label_trajectory = Trajectory(bev_poses, indices, possible_timesteps, id=tracking_id, localize=False, initial_yaw_estimation=True)
            trajectory_by_sample_filepath[tracking_id] = label_trajectory
        seq_object_label_trajectories.append(trajectory_by_sample_filepath)
    return seq_object_label_trajectories

def get_track_id_colors(samples_sequences: List[List[Sample]], loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]], matched_ids: Dict[str, str]) -> List[Dict[str, Tuple[float, float, float]]]:
    # track id to color
    # Use a predefined color palette from matplotlib (e.g., 'tab10', 'Set1', 'Set2')
    palette = plt.get_cmap('tab10').colors  # 'tab10' has 10 distinct colors
    num_colors = len(palette)

    # Dictionary to store colors for each tracking ID
    training_id_colors_each_seq: List[Dict[str, Tuple[float, float, float]]] = []
    for samples in samples_sequences:
        tracking_id_colors: Dict[str, Tuple[float, float, float]] = {}
        for sample in samples:
            obj_estimates = loc_estimates_by_sample[sample.rgb_image_filepath]
            for tracking_id in obj_estimates.keys():
                if tracking_id in matched_ids and tracking_id not in tracking_id_colors:
                    # Assign a color to each tracking ID if not already assigned
                    # Use colors from the palette by cycling through them
                    tracking_id_colors[tracking_id] = palette[len(tracking_id_colors) % num_colors]
        training_id_colors_each_seq.append(tracking_id_colors)
    return training_id_colors_each_seq