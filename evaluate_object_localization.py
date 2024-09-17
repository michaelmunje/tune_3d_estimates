import yaml
import torch
import cv2
import numpy as np
import json
import os
import sys
import argparse
from typing import List, Tuple, Dict

import bbox_and_bev_estimation
from bbox_and_bev_estimation import LartBBoxAndBevEstimation
import bbox_matching 
from metric_depth import get_depth_estimate, get_3d_points
from utils import Sample, Location, Bbox3d, compute_2d_iou, compute_centroid_error
import open3d as o3d

sys.path.append(os.getcwd())
from utils import get_camera_matrix

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
        
        assert type(self.traj_idx) == int, 'Trajectory index must be an integer. Otherwise, you\'ll need to modify this dict below.' 
        self.lart_tracking_id_to_coda_id = {}
        with open('corresponding_trackings.json', 'r') as file:
            data = json.load(file)
            self.lart_tracking_id_to_coda_id = data

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

    def evaluate(self):
        frame_indices = self.get_viable_samples()
        samples = self.load_samples(frame_indices)

        # get ground truth 2D, 3D bounding boxes, and tracking ids from data
        bbox3d_labels, bbox2d_labels, bev_labels, tracking_ids = self.extract_labels(samples)

        # estimate 2D bbox estimates and bev estimates
        location_estimates = self.bbox_and_bev_estimation.estimate(samples)

        # Match estimated and ground truth bounding boxes
        # matched_ids is a dictionary where:
        # - keys are the estimated bounding box ids (indices in the estimated bbox list)
        # - values are the corresponding ground truth ids
        matched_ids = self.bbox_matching.matching(location_estimates, bbox2d_labels, bbox3d_labels)
        
        bev_labels_with_keys = {}
        for tracking_id, bev_label in zip(tracking_ids, bev_labels):
            bev_labels_with_keys[tracking_id] = bev_label
            
        bev_estimates_with_keys = {}
        for loc_estimate in location_estimates:
            tracking_id = loc_estimate.tracking_id
            bev_estimates_with_keys[tracking_id] = np.array([loc_estimate.cX, loc_estimate.cY])

        # Compute errors between estimated and ground truth bounding boxes
        # matched_ids is a dictionary where:
        # - keys are the estimated bounding box ids (indices in the estimated bbox list)
        # - values are the corresponding ground truth ids
        
        # errors_2d = self.compute_bbox_errors(bbox2d_list, bbox2d_labels, matched_ids)
        errors_bev = self.compute_bev_errors(bev_estimates_with_keys, bev_labels_with_keys, matched_ids)

        # Process and analyze the errors
        mean_error_bev = np.mean(errors_bev)
        # mean_error_3d = np.mean(errors_3d)
        
        # let's also save a plot of the bevs
        self.save_bev_visualization(bev_estimates_with_keys, bev_labels_with_keys, matched_ids, fp='data/bev_visualization.png')
        
        return mean_error_bev
    
    def save_bev_visualization(self, bev_estimates, bev_labels, matched_ids, fp):
        # let's also plot bevs
        self.visualize_bevs(bev_estimates, bev_labels, matched_ids)
        plt.savefig(fp)
        
    def visualize_bevs(self, bev_estimates, bev_labels, matched_ids):
        plt.figure()
        plt.scatter(bev_estimates.values(), bev_labels.values(), c='blue', label='Estimated BEV')
        plt.scatter(bev_labels.values(), bev_labels.values(), c='red', label='Ground Truth BEV')
        plt.xlabel('Estimated BEV')
        plt.ylabel('Ground Truth BEV')
        
        plt.xlim(self.grid_x_min, self.grid_x_max)
        plt.ylim(self.grid_y_min, self.grid_y_max)
        plt.axis('equal')
        plt.grid(True)
        
        plt.title('BEV Estimates and Ground Truth')
        plt.legend()
        plt.show()
    
    def compute_bev_errors(self, bev_estimates, bev_labels, matched_ids):
        errors = []
        for estimated_id, label_id in matched_ids.items():
            bev_estimate = bev_estimates[estimated_id]
            bev_label = bev_labels[label_id]
            error = np.linalg.norm(bev_estimate - bev_label)
            errors.append(error)
        return errors
    
    def visualize_point_cloud(self):
        frame_indices = self.get_viable_samples()
        samples = self.load_samples(frame_indices)
        
        for sample in samples:
            self.bbox_and_bev_estimation.visualize_point_cloud(sample)
    
    def get_viable_samples(self) -> List[str]:
        # need to make sure they have both a .png and .json file
        viable_frame_indices = []
        target_folder = f'data/2d_rect/cam0/{self.traj_idx}'
        # iterate over imgs in folder that correspond to our expected filename
        for filename in os.listdir(target_folder):
            if filename.endswith('.png'):
                frame_idx = filename.split('_')[-1].split('.')[0]
                json_path = f'data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'
                if os.path.exists(json_path):
                    viable_frame_indices.append(frame_idx)
        return viable_frame_indices

    def load_samples(self, frame_indices):
        samples = []
        for frame_idx in frame_indices:
            image_path = f'data/2d_rect/cam0/{self.traj_idx}/2d_rect_cam0_{self.traj_idx}_{frame_idx}.png'
            json_path = f'data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'
            
            with open(json_path, 'r') as file:
                data = json.load(file)
                objects = data['3dbbox']
                
            objects = [obj for obj in objects if obj['classId'] == 'Pedestrian']
            sample = Sample(rgb_image_filepath=image_path, objects=objects, lart_folder=self.lart_folder)
            samples.append(sample)
            
        return samples
    
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
    evaluator.evaluate()