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
from utils import Sample, Location, Bbox3d, compute_2d_iou, compute_centroid_error
import open3d as o3d
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from utils import get_camera_matrix
import math

def quaternion_to_yaw(qw, qx, qy, qz):
    # Yaw calculation from quaternion
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

class ObjectLocalizationEvaluator:
    def __init__(self, config_filepath, debug=False):
        # Load the configuration file
        if config_filepath:
            with open(config_filepath, "r") as file:
                config = yaml.safe_load(file)

        self.use_gamma_correction = config["use_gamma_correction"]
        self.use_gaussian_blur = config["use_gaussian_blur"]
        self.use_undistort_correction = config["use_undistort_correction"]
        self.metric_3d_model = config["metric_3d_model"]
        bbox_and_bev_estimation_method = getattr(bbox_and_bev_estimation, config["bbox_and_bev_estimation"])
        self.bbox_and_bev_estimation = bbox_and_bev_estimation_method(config)
        bbox_matching_method = getattr(bbox_matching, config["bbox_matching"])
        self.bbox_matching = bbox_matching_method(**config["bbox_matching_config"])
        self.lart_folder = config["lart_folder"]
        
        self.sample_save_folder = config["sample_save_folder"]
        self.camera_matrix_np = self.bbox_and_bev_estimation.camera_matrix_np
        self.distortion_coeffs_np = self.bbox_and_bev_estimation.distortion_coeffs_np
        self.extrinsics_np = self.bbox_and_bev_estimation.extrinsics_np
        
        self.grid_x_min, self.grid_x_max = config["x_min"], config["x_max"]
        self.grid_y_min, self.grid_y_max = config["y_min"], config["y_max"]
        self.grid_z_min, self.grid_z_max = config["z_min"], config["z_max"]
        self.grid_limits = [
            self.grid_x_min,
            self.grid_x_max,
            self.grid_y_min,
            self.grid_y_max,
            self.grid_z_min,
            self.grid_z_max,
        ]
        self.traj_idx = config["traj_idx"]
        
        self.frame_intervals_start_indices = config["frame_intervals_start_indices"]
        self.frame_interval_lengths = config["frame_interval_lengths"]
        self.frame_interval_sample_stride = config["frame_interval_sample_stride"]
        self.frame_sequences = []
        
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
                frame_seq[i] = str(frame_idx)
        
        assert type(self.traj_idx) == int, 'Trajectory index must be an integer. Otherwise, you\'ll need to modify this dict below.' 
        
        # get corresponding 
        # row idx is frame/sample idx
        # cols: timestamp x y z qw qx qy qz
        # x y z is translation
        # qw qx qy qz is quaternion
        # example of row:
        # 1675697952.451304  -0.000504   0.000465   0.000038   0.998855   0.000319   0.027345  -0.039263
        self.poses_filepath = f'coda-devkit/data/poses/dense_global/{self.traj_idx}.txt'
        self.poses = np.loadtxt(self.poses_filepath)
        # let's map idx to all these things!
        self.frame_idx_to_pose = {
            idx: {'timestep': pose[0], 
                  'x': pose[1], 'y': pose[2], 'z': pose[3], 
                  'qw': pose[4], 'qx': pose[5], 'qy': pose[6], 'qz': pose[7]} 
            for idx, pose in enumerate(self.poses)
        }
        
    #  transform bev coordinates w.r.t. current pose to w.r.t initial pose
    def get_bev_coords_wrt_initial_pose(self, bev_coords: np.array, current_pose_frame_idx: int, initial_pose_frame_idx: int):
        # get the transformation matrix
        original_bev_coords = bev_coords
        current_pose = self.frame_idx_to_pose[current_pose_frame_idx]
        initial_pose = self.frame_idx_to_pose[initial_pose_frame_idx]
        
        # if current_pose and initial_pose are the same, then let's make sure our result is (0, 0)
        same_pose = current_pose_frame_idx == initial_pose_frame_idx
        
        # reference code:

        #     # transform coordinates from current frame to first frame
        #     (curr_x, curr_y), current_yaw = odometry_data[timestep]
        #     (initial_x, initial_y), initial_yaw = odometry_data[0]
            
        #     delta_yaw = (current_yaw - initial_yaw)
            
        #     delta_x = curr_x - initial_x
        #     delta_y = curr_y - initial_y
            
        #     R = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)], 
        #                   [np.sin(delta_yaw), np.cos(delta_yaw)]])
            
        #     R2 = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], 
        #                   [-np.sin(initial_yaw), np.cos(initial_yaw)]])
            
        #     T = np.array([delta_x, delta_y])

        #     coords = np.array(coords)
        #     coords = R @ coords[:2] + (R2 @ T)
        #     coords = [coords[0], coords[1], -joint_coords[1]]
            
        curr_x, curr_y, curr_z = current_pose['x'], current_pose['y'], current_pose['z']
        curr_qw, curr_qx, curr_qy, curr_qz = current_pose['qw'], current_pose['qx'], current_pose['qy'], current_pose['qz']
        curr_yaw = quaternion_to_yaw(curr_qw, curr_qx, curr_qy, curr_qz)
        
        initial_x, initial_y, initial_z = initial_pose['x'], initial_pose['y'], initial_pose['z']
        initial_qw, initial_qx, initial_qy, initial_qz = initial_pose['qw'], initial_pose['qx'], initial_pose['qy'], initial_pose['qz']
        initial_yaw = quaternion_to_yaw(initial_qw, initial_qx, initial_qy, initial_qz)
        
        delta_yaw = curr_yaw - initial_yaw
        
        if same_pose:
            # delta yaw should be close to 0
            assert np.abs(delta_yaw) < 1e-5, f'Delta yaw is not close to 0 when it should be: {delta_yaw}'
        
        delta_x = curr_x - initial_x
        delta_y = curr_y - initial_y
        
        R = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)], 
                      [np.sin(delta_yaw), np.cos(delta_yaw)]])
        
        R2 = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], 
                      [-np.sin(initial_yaw), np.cos(initial_yaw)]])
        
        T = np.array([delta_x, delta_y])
        
        bev_coords = np.array(original_bev_coords)
        bev_coords = R @ bev_coords[:2] + (R2 @ T)
        
        # check for nans
        if np.isnan(bev_coords).any():
            print('Debug info, found nans in bev_coords')
            print(f'original_bev_coords: {original_bev_coords}')
            print(f'bev_coords: {bev_coords}')
            print(f'current_pose: {current_pose}')
            print(f'initial_pose: {initial_pose}')
            print(f'delta_yaw: {delta_yaw}')
            print(f'delta_x: {delta_x}')
            print(f'delta_y: {delta_y}')
            print(f'R: {R}')
            print(f'R2: {R2}')
            print(f'T: {T}')
            
        assert not np.isnan(bev_coords).any(), f'Nans in bev_coords: {bev_coords}'
        
        return bev_coords
        
    def frame_idx_to_img_fp(self, frame_idx: int) -> str:
        return f'coda-devkit/data/2d_rect/cam0/{self.traj_idx}/2d_rect_cam0_{self.traj_idx}_{frame_idx}.png'
    
    def frame_idx_to_json_fp(self, frame_idx: int) -> str:
        return f'coda-devkit/data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'

    def compute_bbox_errors(self, bbox2d_list: List[Location], 
                            bbox2d_labels: List[Location],
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
    
    def extract_labels(self, samples: List[Sample]) -> Tuple[List[Bbox3d], List[Bbox3d], List[Bbox3d], List[int]]:
        bbox3d_labels = []
        bbox2d_labels = []
        bev_labels = []
        tracking_ids = []
        
        for sample in samples:
            objects = sample.objects
            for obj in objects:
                bbox3d_labels.append(obj.bbox3d)
                bbox2d_labels.append(obj.bbox3d.get_bbox2d(self.camera_matrix_np, self.distortion_coeffs_np, self.extrinsics_np))
                bev_labels.append(obj.bbox3d.get_bev_location())
                tracking_ids.append(obj.id)
        
        return bbox3d_labels, bbox2d_labels, bev_labels, tracking_ids

    def evaluate_sequences(self):
        samples_sequences = self.load_samples()
        mse = 0.0
        
        # let's first group by sample
        loc_estimates_by_sample = {} # key is sample filepath, value is a dict tracking_id -> bev_estimate
        loc_labels_by_sample = {} # key is sample filepath, value is a dict tracking_id -> bev_label
        matched_ids = None
        
        # populate estimates and labels for each sample for each tracking id
        for samples in samples_sequences:
            location_estimates = self.bbox_and_bev_estimation.estimate(samples)
            bbox3d_labels, bbox2d_labels, bev_labels, tracking_ids = self.extract_labels(samples)
            
            # Little hacky, but we only need to do this once
            matched_ids = self.bbox_matching.matching(location_estimates, bbox2d_labels, bbox3d_labels)

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
                    
        # correct all of them to be w.r.t. the initial pose (in corresponding sequence)
        for samples in samples_sequences:
            initial_frame_idx = samples[0].get_sample_idx()
            for sample in samples:
                sample_filepath = sample.rgb_image_filepath
                current_sample_idx = sample.get_sample_idx()
                for tracking_id in loc_estimates_by_sample[sample_filepath]:
                    bev_coords = loc_estimates_by_sample[sample_filepath][tracking_id].get_bev_location()
                    updated_bev = self.get_bev_coords_wrt_initial_pose(
                        bev_coords=bev_coords, 
                        current_pose_frame_idx=current_sample_idx, 
                        initial_pose_frame_idx=initial_frame_idx
                    )
                    loc_estimates_by_sample[sample_filepath][tracking_id].cX = updated_bev[0]
                    loc_estimates_by_sample[sample_filepath][tracking_id].cY = updated_bev[1]
                    
                    label_tracking_id = matched_ids[tracking_id]
                    if label_tracking_id in loc_labels_by_sample[sample_filepath]:
                        bev_coords = loc_labels_by_sample[sample_filepath][label_tracking_id].get_bev_location()
                        updated_bev = self.get_bev_coords_wrt_initial_pose(
                            bev_coords=bev_coords, 
                            current_pose_frame_idx=current_sample_idx, 
                            initial_pose_frame_idx=initial_frame_idx
                        )
                        loc_labels_by_sample[sample_filepath][label_tracking_id].cX = updated_bev[0]
                        loc_labels_by_sample[sample_filepath][label_tracking_id].cY = updated_bev[1]
                    
        # track id to color
        
        # Use a predefined color palette from matplotlib (e.g., 'tab10', 'Set1', 'Set2')
        palette = plt.get_cmap('tab10').colors  # 'tab10' has 10 distinct colors
        num_colors = len(palette)

        # Dictionary to store colors for each tracking ID
        training_id_colors_each_seq: List[Dict] = []
        for samples in samples_sequences:
            tracking_id_colors = {}
            for sample in samples:
                obj_estimates = loc_estimates_by_sample[sample.rgb_image_filepath]
                for tracking_id in obj_estimates.keys():
                    if tracking_id in matched_ids and tracking_id not in tracking_id_colors:
                        # Assign a color to each tracking ID if not already assigned
                        # Use colors from the palette by cycling through them
                        tracking_id_colors[tracking_id] = palette[len(tracking_id_colors) % num_colors]
            training_id_colors_each_seq.append(tracking_id_colors)
            
        if not os.path.exists(self.sample_save_folder):
            os.makedirs(self.sample_save_folder)
            
        sample_position_history_by_seq = []
        for samples in samples_sequences:
            sample_position_history = []
            initial_pose_idx = samples[0].get_sample_idx()
            initial_pose = self.frame_idx_to_pose[initial_pose_idx]
            initial_position = np.array([initial_pose['x'], initial_pose['y']])
            initial_yaw = quaternion_to_yaw(initial_pose['qw'], initial_pose['qx'], initial_pose['qy'], initial_pose['qz'])
            for sample in samples:
                sample_idx = sample.get_sample_idx()
                pose = self.frame_idx_to_pose[sample_idx]
                # update pose so w.r.t. initial pose
                localized_position = self.localize_position_wrt_initial_pose(
                    future_position=np.array([pose['x'], pose['y']]), 
                    initial_position=initial_position, 
                    initial_yaw=initial_yaw
                )
                sample_position_history.append(localized_position)
            sample_position_history_by_seq.append(sample_position_history)
        # print(sample_pose_history_by_seq[0])
            
        assert len(training_id_colors_each_seq) == len(samples_sequences), 'Number of sequences and colors do not match'
        assert len(sample_position_history_by_seq) == len(samples_sequences), 'Number of sequences and pose histories do not match'
            
        for samples, sample_pose_history, track_id_to_color in zip(samples_sequences, sample_position_history_by_seq, training_id_colors_each_seq):
            prev_sample_filepath = None
            prev_robot_pose = None
            seq_bev_filepaths = []
            for i in range(len(samples)):
                sample_filepath = samples[i].rgb_image_filepath
                errors_over_samples = []
                    # loc_estimates = loc_estimates_by_sample[sample_filepath]
                    # loc_labels = loc_labels_by_sample[sample_filepath]
                    # errors_bev = self.compute_bev_errors(loc_estimates, loc_labels, matched_ids)
                    # errors_over_samples.extend(errors_bev)

                # Process and analyze the errors
                # mean_error_bev = np.mean(errors_over_samples)
                # mse += mean_error_bev / len(samples_sequences)
                bev_out_filepath = f'{self.sample_save_folder}/BEV_visualization_{sample_filepath.split("/")[-1]}'
                current_sample_filepath = sample_filepath
                current_robot_pose = sample_pose_history[i]
                self.save_bev_pair_visualization(current_robot_pose, prev_robot_pose, 
                                                 loc_estimates_by_sample, loc_labels_by_sample, 
                                                 current_sample_filepath, prev_sample_filepath, 
                                                 matched_ids, fp=bev_out_filepath, tracking_id_colors=track_id_to_color)
                seq_bev_filepaths.append(bev_out_filepath)
                prev_sample_filepath = sample_filepath
                prev_robot_pose = current_robot_pose
            # create gif from all the bevs
            
            frame_duration = 500
            seq_bev_basename = os.path.basename(seq_bev_filepaths[0]).replace('.png', '')
            seq_bev_out = os.path.join(self.sample_save_folder, f'{seq_bev_basename}.gif')
        
            # initial_img = Image.open(seq_bev_filepaths[0])
            # other_imgs = [Image.open(fp) for fp in seq_bev_filepaths[1:]]
            
            # initial_img.save(
            #     seq_bev_out,
            #     save_all=True,
            #     optimize=False,
            #     append_images=other_imgs,
            #     duration=frame_duration,
            #     loop=0
            # )
                        
            # Open the initial image and convert it to RGB mode to avoid random palette application
            initial_img = Image.open(seq_bev_filepaths[0]).convert('RGB')

            # Convert other images to RGB mode as well
            other_imgs = [Image.open(fp).convert('RGB') for fp in seq_bev_filepaths[1:]]

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

    def evaluate_frames_independently(self):
        samples_sequences = self.load_samples()
        
        for samples in samples_sequences:

            # get ground truth 2D, 3D bounding boxes, and tracking ids from data
            bbox3d_labels, bbox2d_labels, bev_labels, tracking_ids = self.extract_labels(samples)

            # estimate 2D bbox estimates and bev estimates
            location_estimates = self.bbox_and_bev_estimation.estimate(samples)

            # Match estimated and ground truth bounding boxes
            # matched_ids is a dictionary where:
            # - keys are the estimated bounding box ids (indices in the estimated bbox list)
            # - values are the corresponding ground truth ids
            matched_ids = self.bbox_matching.matching(location_estimates, bbox2d_labels, bbox3d_labels)

            # Compute errors between estimated and ground truth bounding boxes
            # matched_ids is a dictionary where:
            # - keys are the estimated bounding box ids (indices in the estimated bbox list)
            # - values are the corresponding ground truth ids
            
            # let's first group by sample
            loc_estimates_by_sample = {} # key is sample filepath, value is a dict tracking_id -> bev_estimate
            for loc_estimates_for_sample in location_estimates:
                sample_filepath = loc_estimates_for_sample[0].sample_filepath
                if sample_filepath not in loc_estimates_by_sample:
                    loc_estimates_by_sample[sample_filepath] = {}
                for loc_estimate in loc_estimates_for_sample:
                    loc_estimates_by_sample[sample_filepath][loc_estimate.tracking_id] = loc_estimate
                
            loc_labels_by_sample = {} # key is sample filepath, value is a dict tracking_id -> bev_label
            for sample in samples:
                sample_filepath = sample.rgb_image_filepath
                if sample_filepath not in loc_labels_by_sample:
                    loc_labels_by_sample[sample_filepath] = {}
                for obj in sample.objects:
                    loc_labels_by_sample[sample_filepath][obj.id] = obj.bbox3d
                
            errors_over_samples = []
            
            for sample_filepath in loc_estimates_by_sample:
                loc_estimates = loc_estimates_by_sample[sample_filepath]
                loc_labels = loc_labels_by_sample[sample_filepath]
                errors_bev = self.compute_bev_errors(loc_estimates, loc_labels, matched_ids)
                errors_over_samples.extend(errors_bev)

            # Process and analyze the errors
            mean_error_bev = np.mean(errors_over_samples)
            # mean_error_3d = np.mean(errors_3d)
            
            if not os.path.exists(self.sample_save_folder):
                os.makedirs(self.sample_save_folder)
            
            # let's also save a plot of the bevs
            for sample_filepath in loc_estimates_by_sample:
                bev_out_filepath = f'{self.sample_save_folder}/BEV_visualization_{sample_filepath.split("/")[-1]}'
                self.save_bev_visualization(loc_estimates_by_sample[sample_filepath], 
                                            loc_labels_by_sample[sample_filepath], 
                                            matched_ids, fp=bev_out_filepath)
            
            return mean_error_bev
    
    def visualize_2d_bbox_on_image(self, image, x, y, w, h, color):
        color = tuple(int(c * 255) for c in color)
        # change to bgr
        color = color[::-1]
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 6)
    
    def save_bev_pair_visualization(self, current_robot_pose, prev_robot_pose,
                               all_loc_estimates, all_loc_labels,
                               current_sample_filepath, prev_sample_filepath, matched_ids, fp, 
                               include_2d_bbox_estimate_visualization: bool = True,
                               include_3d_bbox_label_visualization: bool = False,
                               tracking_id_colors: Dict[int, Tuple[float, float, float]] = None):
        
        # tracking_id -> bev_estimate
        loc_estimates = all_loc_estimates[current_sample_filepath]
        loc_labels = all_loc_labels[current_sample_filepath]
        
        prev_loc_estimates = all_loc_estimates[prev_sample_filepath] if prev_sample_filepath else None
        prev_loc_labels = all_loc_labels[prev_sample_filepath] if prev_sample_filepath else None
        image = cv2.imread(current_sample_filepath)

        corresponding_estimates_and_labels = []
        for tracking_id, loc_estimate in loc_estimates.items():
            if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
                bev_estimate = loc_estimate.get_bev_location()
                bev_label = loc_labels[matched_ids[tracking_id]].get_bev_location()
                color = tracking_id_colors[tracking_id]
                if prev_loc_estimates is not None and tracking_id in prev_loc_estimates and matched_ids[tracking_id] in prev_loc_labels:
                    prev_bev_estimate = prev_loc_estimates[tracking_id].get_bev_location() if prev_loc_estimates else None
                    prev_bev_label = prev_loc_labels[matched_ids[tracking_id]].get_bev_location() if prev_loc_labels else None
                else:
                    prev_bev_estimate = None
                    prev_bev_label = None
                corresponding_estimates_and_labels.append((bev_estimate, bev_label, prev_bev_estimate, prev_bev_label, color))
            
            if include_2d_bbox_estimate_visualization:
                # visualize the 2d bbox on the image
                for tracking_id, loc_estimate in loc_estimates.items():
                    if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
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
            self.visualize_bev_pair(current_robot_pose, prev_robot_pose, corresponding_estimates_and_labels)
            
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
            
    def yaw_rotmat(self, yaw: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
            
    def localize_position_wrt_initial_pose(self, future_position, initial_position, initial_yaw):
        rotmat = self.yaw_rotmat(initial_yaw)
        if future_position.shape[-1] == 2:
            rotmat = rotmat[:2, :2]
        elif future_position.shape[-1] == 3:
            pass
        else:
            raise ValueError

        return (future_position - initial_position).dot(rotmat)

    def visualize_bev_pair(self, current_robot_pose, prev_robot_pose, corresponding_estimates_and_labels_and_colors):
        # each estimate (and its corresponding label) should have a different color
        # now let's plot them
        fig, ax = plt.subplots()
        for bev_estimate, bev_label, prev_bev_estimate, prev_bev_label, color in corresponding_estimates_and_labels_and_colors:
            if prev_bev_label is None or prev_bev_estimate is None:
                ax.plot(bev_estimate[1], bev_estimate[0], '*', color=color, markersize=10)
                ax.plot(bev_label[1], bev_label[0], 'o', color=color, markersize=8)
            else:
                fixed_arrow_length = 1.0
                ax.plot(bev_estimate[1], bev_estimate[0], '*', color=color, markersize=10)
                ax.plot(bev_label[1], bev_label[0], 'o', color=color, markersize=8)
                    
                # Calculate direction vector for bev_estimate (pointing away from the previous point)
                vec_estimate = np.array([bev_estimate[1] - prev_bev_estimate[1], bev_estimate[0] - prev_bev_estimate[0]])
                norm_estimate = np.linalg.norm(vec_estimate)
                if norm_estimate != 0:
                    vec_estimate = vec_estimate / norm_estimate * fixed_arrow_length
                
                # Calculate direction vector for bev_label (pointing away from the previous point)
                vec_label = np.array([bev_label[1] - prev_bev_label[1], bev_label[0] - prev_bev_label[0]])
                norm_label = np.linalg.norm(vec_label)
                if norm_label != 0:
                    vec_label = vec_label / norm_label * fixed_arrow_length

                # Draw arrow from the current bev_estimate in the flipped direction (away from previous)
                ax.arrow(bev_estimate[1], bev_estimate[0], vec_estimate[0], vec_estimate[1], 
                        width=0.1, fc=color, ec=color)

                # Draw arrow from the current bev_label in the flipped direction (away from previous)
                ax.arrow(bev_label[1], bev_label[0], vec_label[0], vec_label[1], 
                        width=0.1, fc=color, ec=color)
        # also plot the trajectory
        ax.plot(current_robot_pose[1], current_robot_pose[0], 's', color='black', markersize=5)
        if prev_robot_pose is not None:
            fixed_arrow_length = 1.0
            vec_estimate = np.array([current_robot_pose[1] - prev_robot_pose[1], current_robot_pose[0] - prev_robot_pose[0]])
            norm_estimate = np.linalg.norm(vec_estimate)
            if norm_estimate != 0:
                vec_estimate = vec_estimate / norm_estimate * fixed_arrow_length
            ax.arrow(current_robot_pose[1], current_robot_pose[0], vec_estimate[0], vec_estimate[1], 
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
    
    def save_bev_visualization(self, loc_estimates, loc_labels, matched_ids, fp, 
                               include_2d_bbox_estimate_visualization: bool = True,
                               include_3d_bbox_label_visualization: bool = False):
        first_loc_estimate = next(iter(loc_estimates.values()))
        sample_filepath = first_loc_estimate.sample_filepath
        # load the image
        image = cv2.imread(sample_filepath)
                
        # Use a predefined color palette from matplotlib (e.g., 'tab10', 'Set1', 'Set2')
        palette = plt.get_cmap('tab10').colors  # 'tab10' has 10 distinct colors
        num_colors = len(palette)

        # Dictionary to store colors for each tracking ID
        tracking_id_colors = {}

        corresponding_estimates_and_labels = []
        for tracking_id, loc_estimate in loc_estimates.items():
            if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
                bev_estimate = loc_estimate.get_bev_location()
                bev_label = loc_labels[matched_ids[tracking_id]].get_bev_location()
                
                # Assign a color to each tracking ID if not already assigned
                if tracking_id not in tracking_id_colors:
                    # Use colors from the palette by cycling through them
                    tracking_id_colors[tracking_id] = palette[len(tracking_id_colors) % num_colors]
            
                color = tracking_id_colors[tracking_id]
                corresponding_estimates_and_labels.append((bev_estimate, bev_label, color))
        
        if include_2d_bbox_estimate_visualization:
            # visualize the 2d bbox on the image
            for tracking_id, loc_estimate in loc_estimates.items():
                if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
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
        self.visualize_bev(corresponding_estimates_and_labels)
        
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
        
    def visualize_bev(self, corresponding_estimates_and_labels_and_colors):
        # each estimate (and its corresponding label) should have a different color
    
        # now let's plot them
        fig, ax = plt.subplots()
        for bev_estimate, bev_label, color in corresponding_estimates_and_labels_and_colors:
            ax.plot(bev_estimate[1], bev_estimate[0], '*', color=color, markersize=10)
            ax.plot(bev_label[1], bev_label[0], 'o', color=color, markersize=8)
            
        # expected grid limits
        ax.set_xlim(self.grid_y_min, self.grid_y_max)
        ax.set_ylim(self.grid_x_min, self.grid_x_max)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.grid(True)
        
        # Custom legend for markers
        estimated_legend = mlines.Line2D([], [], color='black', marker='*', linestyle='None', label='Estimated', markersize=10)
        pseudo_gt_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Pseudo-Ground Truth', markersize=8)
        ax.legend(handles=[estimated_legend, pseudo_gt_legend])
        
        ax.set_title(f'BEV Visualization: {self.bbox_and_bev_estimation.__class__.__name__}')
        ax.set_ylabel('X (forward)')
        ax.set_xlabel('Y (left)')
        ax.set_aspect('equal', 'box')
        plt.show()
        
    
    def compute_bev_errors(self, location_estimates, location_labels, matched_ids):
        errors = []
        for estimated_id, label_id in matched_ids.items():
            if estimated_id not in location_estimates or label_id not in location_labels:
                continue
            bev_estimate = location_estimates[estimated_id].get_bev_location()
            bev_label = location_labels[label_id].get_bev_location()
            error = np.linalg.norm(bev_estimate - bev_label)
            errors.append(error)
        return errors
    
    def visualize_point_cloud(self):
        samples = self.load_samples()
        
        for sample in samples:
            self.bbox_and_bev_estimation.visualize_point_cloud(sample)
    
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