import yaml
import torch
import cv2
import numpy as np
import json
import os
import sys
import argparse
from typing import List, Tuple, Dict
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

import bbox_and_bev_estimation
import bbox_matching 
from utils import Sample, LocationWith3dBBox, quaternion_to_yaw, transform_trajectory_to_initial_pose
import open3d as o3d
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from utils import get_camera_matrix, BEVPose, Trajectory, LocationWith2DBBox
import math


class ObjectLocalizationEvaluator:
    def __init__(self, config_filepath: str, debug_mode: bool = False):
        assert os.path.exists(config_filepath), f'Config filepath {config_filepath} does not exist'
        with open(config_filepath, "r") as file:
            config: Dict = yaml.safe_load(file)

        self.use_gamma_correction: bool = config["use_gamma_correction"]
        self.use_gaussian_blur: bool = config["use_gaussian_blur"]
        self.use_undistort_correction: bool = config["use_undistort_correction"]
        self.metric_3d_model: str = config["metric_3d_model"]

        bbox_and_bev_estimation_method = getattr(bbox_and_bev_estimation, config["bbox_and_bev_estimation"])
        self.bbox_and_bev_estimation = bbox_and_bev_estimation_method(config)
        bbox_matching_method = getattr(bbox_matching, config["bbox_matching"])
        self.bbox_matching = bbox_matching_method(**config["bbox_matching_config"])
        
        self.lart_folder: str = config["lart_folder"]
        self.debug_mode: bool = debug_mode
        
        self.sample_save_folder: str = config["sample_save_folder"]
        if not os.path.exists(self.sample_save_folder):
            os.makedirs(self.sample_save_folder)

        self.camera_matrix_np: np.ndarray = self.bbox_and_bev_estimation.camera_matrix_np
        self.distortion_coeffs_np: np.ndarray = self.bbox_and_bev_estimation.distortion_coeffs_np
        self.extrinsics_np: np.ndarray = self.bbox_and_bev_estimation.extrinsics_np
        
        self.grid_x_min: float = config["x_min"]
        self.grid_x_max: float = config["x_max"]
        self.grid_y_min: float = config["y_min"]
        self.grid_y_max: float = config["y_max"]
        self.grid_z_min: float = config["z_min"]
        self.grid_z_max: float = config["z_max"]
        self.grid_limits: List[float] = [
            self.grid_x_min,
            self.grid_x_max,
            self.grid_y_min,
            self.grid_y_max,
            self.grid_z_min,
            self.grid_z_max,
        ]
        self.traj_idx: int = config["traj_idx"]
        
        self.frame_intervals_start_indices: List[int] = config["frame_intervals_start_indices"]
        self.frame_interval_lengths: int = config["frame_interval_lengths"]
        self.frame_interval_sample_stride: int = config["frame_interval_sample_stride"]
        self.frame_sequences: List[List[int]] = []

        self.smooth_robot_trajectory: bool = config["smooth_robot_trajectory"]
        self.smooth_estimate_trajectories: bool = config["smooth_estimate_trajectories"]
        self.interpolate_between_trajectory: bool = config["interpolate_between_trajectory"]
        self.smooth_label_trajectories: bool = config["smooth_label_trajectories"]
        
        for start_idx in self.frame_intervals_start_indices:
            self.frame_sequences.append(list(range(start_idx, start_idx + self.frame_interval_lengths, self.frame_interval_sample_stride)))
            
        # check if the filepath exists, if not try 1 lower
        for frame_seq in self.frame_sequences:
            for i, frame_idx in enumerate(frame_seq):
                frame_idx = int(frame_idx)
                while not os.path.exists(self.frame_idx_to_img_fp(frame_idx)):
                    print(f'Frame {frame_idx} does not exist. Trying 1 lower.')
                    frame_idx -= 1
                    assert frame_idx >= 0, 'Frame index cannot be negative.'
                frame_seq[i] = frame_idx
        
        assert type(self.traj_idx) == int, 'Trajectory index must be an integer. Otherwise, you\'ll need to modify this dict below.' 
        
        self.poses_filepath: str = f'coda-devkit/data/poses/dense_global/{self.traj_idx}.txt'
        self.poses: np.ndarray = np.loadtxt(self.poses_filepath)
        
        self.frame_idx_to_bev_pose: Dict[int, BEVPose] = {}
        for idx, pose in enumerate(self.poses):
            x, y = pose[1], pose[2]
            yaw = quaternion_to_yaw(pose[4], pose[5], pose[6], pose[7])
            self.frame_idx_to_bev_pose[int(idx)] = BEVPose(x, y, yaw)

        # for each frame sequence, create a trajectory for poses
        self.trajectories: List[Trajectory] = []
        for frame_seq in self.frame_sequences:
            bevs = [self.frame_idx_to_bev_pose[frame_idx] for frame_idx in frame_seq]
            trajectory = Trajectory(bevs, frame_seq, frame_seq, id='robot', localize=True)
            trajectory.kalman_smooth()
            self.trajectories.append(trajectory)
            

    def frame_idx_to_img_fp(self, frame_idx: int) -> str:
        return f'coda-devkit/data/2d_rect/cam0/{self.traj_idx}/2d_rect_cam0_{self.traj_idx}_{frame_idx}.png'
    
    def frame_idx_to_json_fp(self, frame_idx: int) -> str:
        return f'coda-devkit/data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'
    
    def extract_labels(self, samples: List[Sample]) -> List[List[LocationWith3dBBox]]:
        all_labels: List[List[LocationWith3dBBox]] = []
        for sample in samples:
            labels: List[LocationWith3dBBox] = []
            objects = sample.objects
            for obj in objects:
                labels.append(obj.bbox3d)
            all_labels.append(labels)
        return all_labels
    
    def get_estimates_and_labels_per_sample(self, samples_sequences: List[List[Sample]]) -> Tuple[Dict[str, Dict[str, LocationWith2DBBox]], Dict[str, Dict[str, LocationWith3dBBox]], Dict[str, str]]:
        # let's first group by sample
        loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]] = {} # key is sample filepath, value is a dict tracking_id -> bev_estimate
        loc_labels_by_sample: Dict[str, Dict[str, LocationWith3dBBox]] = {} # key is sample filepath, value is a dict tracking_id -> bev_label
        matched_ids: Dict[str, str] = {} # tracking id from 2d bbox -> tracking id from 3d bbox
        
        # populate estimates and labels for each sample for each tracking id
        for samples in samples_sequences:
            location_estimates: List[List[LocationWith2DBBox]] = self.bbox_and_bev_estimation.estimate(samples)
            location_labels: List[List[LocationWith3dBBox]] = self.extract_labels(samples)
            
            # Little hacky, but we only need to do this once
            matched_ids: Dict[str, str] = self.bbox_matching.matching(location_estimates, location_labels)

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
    
    def get_object_estimate_trajectories(self, samples_sequences: List[List[Sample]], loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]], matched_ids: Dict[str, str]) -> List[Dict[str, Dict[str, Trajectory]]]:
        # for each sample sequence, create a trajectory
        seq_object_estimate_trajectories: List[Dict[str, Trajectory]] = [] # (for each sample seq), key is track_id, value is trajectory
        
        for seq_idx in range(len(samples_sequences)):
            samples = samples_sequences[seq_idx]
            possible_timesteps = self.trajectories[seq_idx].possible_timesteps
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
                estimate_trajectory = Trajectory(bev_poses, indices, possible_timesteps, id=tracking_id, localize=False)
                estimate_trajectory.estimate_yaws()
                
                timesteps_before_interpolation = [ts for ts in estimate_trajectory.corresponding_timesteps]
                # timesteps in trajectory
                if self.interpolate_between_trajectory: 
                    estimate_trajectory.interpolate_all_missing_poses()
                if self.smooth_estimate_trajectories:
                    estimate_trajectory.kalman_smooth()
                    
                timesteps_after_interpolation_and_smoothing = estimate_trajectory.corresponding_timesteps
                
                # now let's see if we need to add entries to loc_estimates_by_sample
                # timesteps that were added
                timesteps_added = [ts for ts in timesteps_after_interpolation_and_smoothing if ts not in timesteps_before_interpolation]
                for timestep in timesteps_added:
                    for sample in samples:
                        sample_filepath = sample.rgb_image_filepath
                        sample_idx = sample.get_sample_idx()
                        if sample_idx == timestep:
                            loc_estimates_by_sample[sample_filepath][tracking_id] = LocationWith2DBBox(x=-1.0, y=-1.0, w=-1.0, h=-1.0, 
                                                                                                       cX=-1.0, cY=-1.0, cZ=-1.0, 
                                                                                                       tracking_id=tracking_id, sample_filepath=sample_filepath)
                trajectory_by_sample_filepath[tracking_id] = estimate_trajectory
            seq_object_estimate_trajectories.append(trajectory_by_sample_filepath)
        return seq_object_estimate_trajectories
    
    def get_object_label_trajectories(self, samples_sequences: List[List[Sample]], loc_labels_by_sample: Dict[str, Dict[str, LocationWith3dBBox]], matched_ids: Dict[str, str]) -> List[Dict[str, Dict[str, Trajectory]]]:
        # for each sample sequence, create a trajectory
        seq_object_label_trajectories: List[Dict[str, Trajectory]] = [] # (for each sample seq), key is track_id, value is trajectory
        matched_ids_inv = {v: k for k, v in matched_ids.items()}
        
        for seq_idx in range(len(samples_sequences)):
            samples = samples_sequences[seq_idx]
            possible_timesteps = self.trajectories[seq_idx].possible_timesteps
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
                label_trajectory = Trajectory(bev_poses, indices, possible_timesteps, id=tracking_id, localize=False)
                label_trajectory.estimate_yaws()
                timesteps_before_interpolation = [ts for ts in label_trajectory.corresponding_timesteps]
                if self.interpolate_between_trajectory:
                    label_trajectory.interpolate_all_missing_poses()
                if self.smooth_label_trajectories:
                    label_trajectory.kalman_smooth()
                timesteps_after_interpolation_and_smoothing = label_trajectory.corresponding_timesteps
                
                # timesteps that were added
                timesteps_added = [ts for ts in timesteps_after_interpolation_and_smoothing if ts not in timesteps_before_interpolation]
                for timestep in timesteps_added:
                    for sample in samples:
                        sample_filepath = sample.rgb_image_filepath
                        sample_idx = sample.get_sample_idx()
                        if sample_idx == timestep:
                            loc_labels_by_sample[sample_filepath][tracking_id] = LocationWith3dBBox(cX=0.0, cY=0.0, cZ=0.0, 
                                                                                                    w = 0.0, h=0.0, l=0.0, 
                                                                                                    r=0.0, p=0.0, y=0.0,
                                                                                                    sample_filepath=sample_filepath,
                                                                                                    tracking_id=tracking_id)
                trajectory_by_sample_filepath[tracking_id] = label_trajectory
            seq_object_label_trajectories.append(trajectory_by_sample_filepath)
        return seq_object_label_trajectories

    def get_track_id_colors(self, samples_sequences: List[List[Sample]], loc_estimates_by_sample: Dict[str, Dict[str, LocationWith2DBBox]], matched_ids: Dict[str, str]) -> List[Dict[str, Tuple[float, float, float]]]:
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

    def evaluate_sequences(self):
        samples_sequences: List[List[Sample]] = self.load_samples()
        loc_estimates_by_sample, loc_labels_by_sample, matched_ids = self.get_estimates_and_labels_per_sample(samples_sequences)
        object_estimate_trajectories_seq = self.get_object_estimate_trajectories(samples_sequences, loc_estimates_by_sample, matched_ids)
        object_label_trajectories_seq = self.get_object_label_trajectories(samples_sequences, loc_labels_by_sample, matched_ids)
        
        assert len(object_estimate_trajectories_seq) == len(object_label_trajectories_seq), 'Number of estimate and label trajectories do not match'
        assert len(samples_sequences) == len(object_estimate_trajectories_seq), 'Number of sequences and trajectories do not match'
        assert len(object_estimate_trajectories_seq) == len(self.trajectories), 'Number of sequences and trajectories do not match'
        
        for i in range(len(self.trajectories)):
            for tracking_id in object_estimate_trajectories_seq[i]:
                assert type(object_estimate_trajectories_seq[i][tracking_id]) == Trajectory, 'Object estimate trajectory must be of type Trajectory'
                transform_trajectory_to_initial_pose(object_estimate_trajectories_seq[i][tracking_id], self.trajectories[i])
                object_estimate_trajectories_seq[i][tracking_id].estimate_yaws()
                if self.smooth_estimate_trajectories:
                    object_estimate_trajectories_seq[i][tracking_id].kalman_smooth()
                    object_estimate_trajectories_seq[i][tracking_id].estimate_yaws()
            for coda_tracking_id in object_label_trajectories_seq[i]:
                assert type(object_label_trajectories_seq[i][coda_tracking_id]) == Trajectory, 'Object label trajectory must be of type Trajectory'
                transform_trajectory_to_initial_pose(object_label_trajectories_seq[i][coda_tracking_id], self.trajectories[i])
                object_label_trajectories_seq[i][coda_tracking_id].estimate_yaws()
                if self.smooth_label_trajectories:
                    object_label_trajectories_seq[i][coda_tracking_id].kalman_smooth()
                    object_label_trajectories_seq[i][coda_tracking_id].estimate_yaws()
        
        training_id_colors_each_seq = self.get_track_id_colors(samples_sequences, loc_estimates_by_sample, matched_ids)
            
        assert len(training_id_colors_each_seq) == len(samples_sequences), 'Number of sequences and colors do not match'
        assert len(self.trajectories) == len(samples_sequences), 'Number of sequences and trajectories do not match'
        
        assert isinstance(object_estimate_trajectories_seq, list), 'Object estimate trajectories must be a list'
        assert isinstance(object_label_trajectories_seq, list), 'Object label trajectories must be a list'
        
        for seq_idx in range(len(samples_sequences)):
            samples = samples_sequences[seq_idx]
            trajectory = self.trajectories[seq_idx]
            object_estimate_trajectories = object_estimate_trajectories_seq[seq_idx]
            object_label_trajectories = object_label_trajectories_seq[seq_idx]
            assert len(samples) == len(trajectory), 'Number of samples and trajectory do not match'
            track_id_to_color = training_id_colors_each_seq[seq_idx]

            seq_bev_filepaths = []
            for i in range(len(samples)):
                sample_filepath = samples[i].rgb_image_filepath
                bev_out_filepath = f'{self.sample_save_folder}/BEV_visualization_{sample_filepath.split("/")[-1]}'
                current_sample_filepath = sample_filepath
                current_robot_pose = trajectory.bev_poses[i]
                self.save_bev_visualization(current_robot_pose, 
                                            object_estimate_trajectories, 
                                            object_label_trajectories,
                                            loc_estimates_by_sample, loc_labels_by_sample, # these are just needed for bbox visuals
                                            current_sample_filepath, 
                                            matched_ids, fp=bev_out_filepath, tracking_id_colors=track_id_to_color)
                seq_bev_filepaths.append(bev_out_filepath)
            # create gif from all the bevs
            self.save_img_bev_gif(seq_bev_filepaths)
            
    def save_img_bev_gif(self, bev_filepaths: List[str]):
        frame_duration = 500
        seq_bev_basename = os.path.basename(bev_filepaths[0]).replace('.png', '')
        seq_bev_out = os.path.join(self.sample_save_folder, f'{seq_bev_basename}.gif')
                    
        # Open the initial image and convert it to RGB mode to avoid random palette application
        initial_img = Image.open(bev_filepaths[0]).convert('RGB')

        # Convert other images to RGB mode as well
        other_imgs = [Image.open(fp).convert('RGB') for fp in bev_filepaths[1:]]

        # Quantize the first image with the ADAPTIVE palette
        initial_img_quantized = initial_img.quantize(method=Image.MEDIANCUT)

        # Quantize all other images using the same palette from the first image
        other_imgs_quantized = [img.quantize(palette=initial_img_quantized) for img in other_imgs]

        # Save GIF with quantized images
        initial_img_quantized.save(
            seq_bev_out,
            save_all=True,
            append_images=other_imgs_quantized,
            duration=frame_duration,
            loop=0,
            optimize=False
        )
    
    def visualize_2d_bbox_on_image(self, image, x, y, w, h, color):
        color = tuple(int(c * 255) for c in color)
        # change to bgr
        color = color[::-1]
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 6)
    
    def save_bev_visualization(self, current_robot_pose,
                               object_estimate_trajectories, object_label_trajectories,
                               all_loc_estimates, all_loc_labels,
                               current_sample_filepath, matched_ids, fp, 
                               include_2d_bbox_estimate_visualization: bool = True,
                               include_3d_bbox_label_visualization: bool = False,
                               tracking_id_colors: Dict[int, Tuple[float, float, float]] = None):
        
        # tracking_id -> bev_estimate
        loc_estimates = all_loc_estimates[current_sample_filepath]
        loc_labels = all_loc_labels[current_sample_filepath]
        
        image = cv2.imread(current_sample_filepath)
        current_sample_idx = int(current_sample_filepath.split('_')[-1].split('.')[0])

        corresponding_estimates_and_labels = []
        for tracking_id, loc_estimate in loc_estimates.items():
            if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
                bev_estimate = object_estimate_trajectories[tracking_id].get_pose_at_timestep(current_sample_idx)
                bev_label = object_label_trajectories[matched_ids[tracking_id]].get_pose_at_timestep(current_sample_idx)
                color = tracking_id_colors[tracking_id]
                corresponding_estimates_and_labels.append((bev_estimate, bev_label, color))
            
            if include_2d_bbox_estimate_visualization:
                # visualize the 2d bbox on the image
                for tracking_id, loc_estimate in loc_estimates.items():
                    if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
                        if loc_estimate.w == 0 or loc_estimate.h == 0 or loc_estimate.x == -1 or loc_estimate.y == -1:
                            continue
                        x, y, w, h = loc_estimate.x, loc_estimate.y, loc_estimate.w, loc_estimate.h
                        x, y = int(x), int(y)
                        w, h = int(w), int(h) 
                        color = tracking_id_colors[tracking_id]
                        self.visualize_2d_bbox_on_image(image, x, y, w, h, color)
            
            if include_3d_bbox_label_visualization:
                raise NotImplementedError
                # # visualize the 3d bbox on the image
                # for loc_estimate in loc_estimates:
                #     loc_estimate.visualize_3d_bbox()
                
            # let's also plot bevs
            self.visualize_bev_plot(current_robot_pose, corresponding_estimates_and_labels)
            
            # save side by side with original image and current plot (use temporary file)
            temp_file = f'{fp}_temp.png'
            plt.savefig(temp_file)
            plt.close()
            img = cv2.imread(temp_file)
            os.remove(temp_file)
            
            # make plot same height as image but keep aspect ratio
            new_height = image.shape[0]
            new_width = int(new_height * img.shape[1] / img.shape[0])
            img = cv2.resize(img, (new_width, new_height))
            
            # add original image to the left
            img = cv2.hconcat([image, img])
            
            cv2.imwrite(fp, img)

    def get_endpoint_diff(self, start, yaw, length):
        # Calculate the direction vector (dx, dy)
        dx = np.cos(yaw)
        dy = np.sin(yaw)
        
        # Normalize the direction vector to ensure its Euclidean length is 1
        norm = np.sqrt(dx**2 + dy**2)
        dx /= norm
        dy /= norm
        
        return dx, dy

    def visualize_bev_plot(self, current_robot_pose, corresponding_estimates_and_labels_and_colors):
        # each estimate (and its corresponding label) should have a different color
        # now let's plot them
        fig, ax = plt.subplots()
        fixed_arrow_length = 0.5
        for bev_estimate, bev_label, color in corresponding_estimates_and_labels_and_colors:
            ax.plot(bev_estimate.y, bev_estimate.x, '*', color=color, markersize=10)
            ax.plot(bev_label.y, bev_label.x, 'o', color=color, markersize=8)
            
            if bev_estimate.yaw is not None:
                bev_estimate_pos = bev_estimate.get_position_np()
                end_bev_estimate = self.get_endpoint_diff(bev_estimate_pos, bev_estimate.yaw, fixed_arrow_length)
                
                # Draw arrow from the current bev_estimate in the flipped direction (away from previous)
                ax.arrow(bev_estimate_pos[1], bev_estimate_pos[0], end_bev_estimate[1], end_bev_estimate[0], 
                        width=0.1, fc=color, ec=color)
                
            if bev_label.yaw is not None:
                bev_label_pos = bev_label.get_position_np()
                end_bev_label = self.get_endpoint_diff(bev_label_pos, bev_label.yaw, fixed_arrow_length)

                # Draw arrow from the current bev_label in the flipped direction (away from previous)
                # compute dist between the points
                # assert np.linalg.norm(np.array(bev_label_pos) - np.array([end_bev_label])) <= fixed_arrow_length, f'bev_label_pos: {bev_label_pos}, end_bev_label_pos: {end_bev_label}'
                ax.arrow(bev_label_pos[1], bev_label_pos[0], end_bev_label[1], end_bev_label[0], 
                        width=0.1, fc=color, ec=color)
        # also plot the trajectory
        ax.plot(current_robot_pose.y, current_robot_pose.x, 's', color='black', markersize=5)
    
        current_robot_position = current_robot_pose.get_position_np()
        end_robot_pose = self.get_endpoint_diff(current_robot_position, current_robot_pose.yaw, fixed_arrow_length)

        ax.arrow(current_robot_position[1], current_robot_position[0], end_robot_pose[1], end_robot_pose[0], 
                width=0.1, fc='black', ec='black')
            
        # expected grid limits
        ax.set_xlim(self.grid_y_min, self.grid_y_max)
        ax.set_ylim(self.grid_x_min, self.grid_x_max)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.grid(True)
        
        # Custom legend for markers
        estimated_legend = mlines.Line2D([], [], color='black', marker='*', linestyle='None', label='Estimate', markersize=10, markerfacecolor='none')
        pseudo_gt_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Pseudo-Ground Truth', markersize=8, markerfacecolor='none')
        # add legend for robot position
        robot_legend = mlines.Line2D([], [], color='black', marker='s', linestyle='None', label='Robot', markersize=5)
        ax.legend(handles=[estimated_legend, pseudo_gt_legend, robot_legend], loc='upper left')
        
        ax.set_title(f'BEV Visualization: {self.bbox_and_bev_estimation.__class__.__name__}')
        ax.set_ylabel('X (forward)')
        ax.set_xlabel('Y (left)')
        ax.set_aspect('equal', 'box')
        plt.show()
    
    def load_samples(self):
        samples_sequences = []
        for frame_indices in self.frame_sequences:
            samples = []
            for frame_idx in frame_indices:
                image_path = self.frame_idx_to_img_fp(frame_idx)
                json_path = self.frame_idx_to_json_fp(frame_idx)
                
                assert os.path.exists(image_path), f'Image path {image_path} does not exist'
                assert os.path.exists(json_path), f'JSON path {json_path} does not exist'

                with open(json_path, 'r') as file:
                    data = json.load(file)
                objects = data['3dbbox']
                
                objects = [obj for obj in objects if obj['classId'] == 'Pedestrian']
                sample = Sample(rgb_image_filepath=image_path, objects=objects, lart_folder=self.lart_folder)
                samples.append(sample)
            samples_sequences.append(samples)
        return samples_sequences
    
    # this will randomly select samples from the dataset
    # and save them to a folder
    def save_selected_samples(self, samples: List[Sample], n_samples: int = 35):
        np.random.seed(42)
        min_samples = min(n_samples, len(samples))
        random_samples = np.random.choice(samples, min_samples)
        # create the folder if it doesn't exist
        if not os.path.exists(self.sample_save_folder):
            os.makedirs(self.sample_save_folder)
        # save img and 2d bbox on img
        for sample in random_samples:
            img = sample.get_img()
            for obj in sample.objects:
                bbox2d = obj.bbox3d.get_bbox2d(self.camera_matrix_np, self.distortion_coeffs_np, self.extrinsics_np)
                bbox2d.tracking_id = obj.id
                x, y, w, h = bbox2d.x, bbox2d.y, bbox2d.w, bbox2d.h
                if w == 0 or h == 0 or x == -1 or y == -1:
                    continue
                cv2.rectangle(img, (y, x), (y+h, x+w), (0, 255, 0), 2)
            print(f'Saving {sample.rgb_image_filepath.split("/")[-1]}')
            cv2.imwrite(f'{self.sample_save_folder}/{sample.rgb_image_filepath.split("/")[-1]}', img)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Evaluate object localization.')
    args.add_argument('--config', type=str, help='Path to configuration file.')
    args.add_argument('--debug', action='store_true', help='Enable debug mode.')
    config_filepath = args.parse_args().config
    debug = args.parse_args().debug
    evaluator = ObjectLocalizationEvaluator(config_filepath, debug)
    evaluator.evaluate_sequences()