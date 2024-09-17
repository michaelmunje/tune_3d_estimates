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
import bbox_matching 
from utils import Sample, Location, Bbox3d, compute_2d_iou, compute_centroid_error
import open3d as o3d
from matplotlib import pyplot as plt

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
        
        assert type(self.traj_idx) == int, 'Trajectory index must be an integer. Otherwise, you\'ll need to modify this dict below.' 

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
    
    def visualize_2d_bbox_on_image(self, image, x, y, w, h):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    def save_bev_visualization(self, loc_estimates, loc_labels, matched_ids, fp, 
                               include_2d_bbox_estimate_visualization: bool = True,
                               include_3d_bbox_label_visualization: bool = False):
        first_loc_estimate = next(iter(loc_estimates.values()))
        sample_filepath = first_loc_estimate.sample_filepath
        # load the image
        image = cv2.imread(sample_filepath)
        
        if include_2d_bbox_estimate_visualization:
            # visualize the 2d bbox on the image
            for tracking_id, loc_estimate in loc_estimates.items():
                if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
                    x, y, w, h = loc_estimate.x, loc_estimate.y, loc_estimate.w, loc_estimate.h
                    x, y = int(x), int(y)
                    w, h = int(w), int(h) 
                    self.visualize_2d_bbox_on_image(image, x, y, w, h)
        
        if include_3d_bbox_label_visualization:
            raise NotImplementedError
            # # visualize the 3d bbox on the image
            # for loc_estimate in loc_estimates:
            #     loc_estimate.visualize_3d_bbox()
            
        # let's also plot bevs
        self.visualize_bev(loc_estimates, loc_labels, matched_ids)
        
        # save side by side with original image and current plot (use temporary file)
        temp_file = f'{fp}_temp.png'
        plt.savefig(temp_file)
        img = cv2.imread(temp_file)
        os.remove(temp_file)
        
        # make plot same height as image but keep aspect ratio
        new_height = image.shape[0]
        new_width = int(new_height * img.shape[1] / img.shape[0])
        img = cv2.resize(img, (new_width, new_height))
        
        # add original image to the left
        img = cv2.hconcat([image, img])
        
        cv2.imwrite(fp, img)
        
    def visualize_bev(self, loc_estimates, loc_labels, matched_ids):
        # each estimate (and its corresponding label) should have a different color
        
        corresponding_estimates_and_labels = []
        for tracking_id, loc_estimate in loc_estimates.items():
            if tracking_id in matched_ids and matched_ids[tracking_id] in loc_labels:
                bev_estimate = loc_estimate.get_bev_location()
                bev_label = loc_labels[matched_ids[tracking_id]].get_bev_location()
                corresponding_estimates_and_labels.append((bev_estimate, bev_label))
        
        # now let's plot them
        fig, ax = plt.subplots()
        for bev_estimate, bev_label in corresponding_estimates_and_labels:
            ax.plot(bev_estimate[1], bev_estimate[0], 'bo', label='Estimated')
            ax.plot(bev_label[1], bev_label[0], 'ro', label='Ground Truth')
            
        # expected grid limits
        ax.set_xlim(self.grid_y_min, self.grid_y_max)
        ax.set_ylim(self.grid_x_min, self.grid_x_max)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.grid(True)
        ax.legend()
        ax.set_title('Bev Visualization')
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
        frame_indices = self.get_viable_samples()
        samples = self.load_samples(frame_indices)
        
        for sample in samples:
            self.bbox_and_bev_estimation.visualize_point_cloud(sample)
    
    def get_viable_samples(self) -> List[str]:
        # need to make sure they have both a .png and .json file
        viable_frame_indices = []
        target_folder = f'coda-devkit/data/2d_rect/cam0/{self.traj_idx}'
        # iterate over imgs in folder that correspond to our expected filename
        for filename in os.listdir(target_folder):
            if filename.endswith('.png'):
                frame_idx = filename.split('_')[-1].split('.')[0]
                json_path = f'coda-devkit/data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'
                if os.path.exists(json_path):
                    viable_frame_indices.append(frame_idx)
        return viable_frame_indices

    def load_samples(self, frame_indices):
        samples = []
        for frame_idx in frame_indices:
            image_path = f'coda-devkit/data/2d_rect/cam0/{self.traj_idx}/2d_rect_cam0_{self.traj_idx}_{frame_idx}.png'
            json_path = f'coda-devkit/data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'
            
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
    evaluator.evaluate()