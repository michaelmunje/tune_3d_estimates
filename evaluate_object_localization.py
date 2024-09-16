import yaml
import torch
import cv2
import numpy as np
import json
import os
import sys
import argparse
from typing import List, Tuple, Dict

import bbox_estimation
import bbox_matching 
from metric_depth import get_depth_estimate, get_3d_points
from utils import Sample, Bbox2d, Bbox3d, compute_2d_iou, compute_centroid_error
import open3d as o3d

sys.path.append(os.getcwd())
from utils import get_camera_matrix

class ObjectLocalizationEvaluator:
    def __init__(self, config_filepath, model):
        # Load the configuration file
        if config_filepath:
            with open(config_filepath, "r") as file:
                config = yaml.safe_load(file)

        self.use_gamma_correction = config["use_gamma_correction"]
        self.use_gaussian_blur = config["use_gaussian_blur"]
        self.use_undistort_correction = config["use_undistort_correction"]
        self.metric_3d_model = config["metric_3d_model"]
        bbox_estimation_method = getattr(bbox_estimation, config["bbox_estimation"])
        self.bbox_estimation = bbox_estimation_method(**config["bbox_estimation_config"])
        bbox_matching_method = getattr(bbox_matching, config["bbox_matching"])
        self.bbox_matching = bbox_matching_method(**config["bbox_matching_config"])
        
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
        
        self.input_width = config["input_width"]
        self.input_height = config["input_height"]

        self.distortion_coeffs = np.array(config["distortion_coeffs"])
        self.translation_vector = torch.tensor(config["translation_vector"])

        self.intrinsics = config["intrinsics"]
        self.camera_matrix = get_camera_matrix(self.intrinsics)
        self.inv_camera_matrix = torch.inverse(self.camera_matrix)
        self.inv_camera_matrix = (
            self.inv_camera_matrix.cuda()
            if torch.cuda.is_available()
            else self.inv_camera_matrix
        )

        # Assuming no rotation, identity rotation matrix (3x3)
        rotation_matrix = torch.eye(3)

        # Combine rotation and translation into a 4x4 extrinsic matrix
        extrinsics = torch.eye(4)
        extrinsics[:3, :3] = rotation_matrix
        extrinsics[:3, 3] = self.translation_vector
        self.extrinsics = extrinsics.cuda() if torch.cuda.is_available() else extrinsics

        if "vit" in self.metric_3d_model:
            self.metric_3d_input_size = (616, 1064)  # for vit model
        else:
            self.metric_3d_input_size = (544, 1216)  # for convnext model

        h, w = self.input_height, self.input_width
        self.input_size = (h, w)
        scale = min(
            self.metric_3d_input_size[0] / self.input_height,
            self.metric_3d_input_size[1] / self.input_width,
        )
        self.scaled_intrinsics = [
            self.intrinsics[0] * scale,
            self.intrinsics[1] * scale,
            self.intrinsics[2] * scale,
            self.intrinsics[3] * scale,
        ]
        self.padding = [123.675, 116.28, 103.53]
        self.pad_h = self.metric_3d_input_size[0] - h
        self.pad_w = self.metric_3d_input_size[1] - w
        self.pad_h = max(self.pad_h, 0)
        self.pad_w = max(self.pad_w, 0)
        self.pad_h_half = self.pad_h // 2
        self.pad_w_half = self.pad_w // 2
        
        # make sure our pad info make sure for the input size...
        # pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half
        assert self.pad_h_half >= 0, 'Pad height half must be greater than or equal to 0.'
        assert self.pad_h - self.pad_h_half >= 0, 'Pad height must be greater than or equal to pad height half.'
        assert self.pad_w_half >= 0, 'Pad width half must be greater than or equal to 0.'
        assert self.pad_w - self.pad_w_half >= 0, 'Pad width must be greater than or equal to pad width half.'
        
        self.pad_info = [
            self.pad_h_half,
            self.pad_h - self.pad_h_half,
            self.pad_w_half,
            self.pad_w - self.pad_w_half,
        ]

        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        
        self.sample_save_folder = 'data/selected_samples'
        
        self.camera_matrix_np = np.array(self.camera_matrix).astype(np.float64)
        self.distortion_coeffs_np = np.array(self.distortion_coeffs).astype(np.float64)
        self.extrinsics_np = self.extrinsics.cpu().numpy().astype(np.float64)
        
        assert type(self.traj_idx) == int, 'Trajectory index must be an integer. Otherwise, you\'ll need to modify this dict below.' 
        self.lart_tracking_id_to_coda_id = {}
        with open('corresponding_trackings.json', 'r') as file:
            data = json.load(file)
            self.lart_tracking_id_to_coda_id = data

        # Load the metric depth estimation model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "yvanyin/metric3d",
            self.metric_3d_model,
            pretrain=True,
            map_location=torch.device("cpu"),
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model)

    def compute_bbox_errors(self, bbox2d_list: List[Bbox2d], 
                            bbox3d_list: List[Bbox3d], 
                            bbox2d_labels: List[Bbox2d],
                            bbox3d_labels: List[Bbox3d],
                            matched_ids: Dict[int, int]) -> Tuple[List[float], List[float]]:
        """
        Compute the errors between estimated and ground truth bounding boxes.

        Args:
            bbox2d_list (List[Bbox2d]): List of estimated 2D bounding boxes.
            bbox3d_list (List[Bbox3d]): List of estimated 3D bounding boxes.
            bbox2d_labels (List[Bbox2d]): List of ground truth 2D bounding boxes.
            bbox3d_labels (List[Bbox3d]): List of ground truth 3D bounding boxes.
            matched_ids (Dict[int, int]): Dictionary mapping estimated bbox ids to ground truth ids.

        Returns:
            Tuple[List[float], List[float]]: Lists of 2D and 3D errors for matched bounding boxes.
        """
        errors_2d = []
        errors_3d = []

        for estimated_id, label_id in matched_ids.items():
            # Compute 2D IoU
            bbox2d_estimated = bbox2d_list[estimated_id]
            bbox2d_label = bbox2d_labels[label_id]
            iou_2d = compute_2d_iou(bbox2d_estimated, bbox2d_label)
            errors_2d.append(1 - iou_2d)  # Error is 1 - IoU

            # Compute 3D centroid 
            bbox3d_estimated = bbox3d_list[estimated_id]
            bbox3d_label = bbox3d_labels[label_id]
            centroid_error_3d = compute_centroid_error(bbox3d_estimated, bbox3d_label)
            errors_3d.append(centroid_error_3d)

        return errors_2d, errors_3d
    
    def extract_labels(self):
        # TODO: Implement extract_labels method
        # This method should extract 2D and 3D bounding boxes, and tracking IDs from the samples
        # It should return:
        # - bbox2d_labels: List of ground truth 2D bounding boxes
        # - bbox3d_labels: List of ground truth 3D bounding boxes
        # - tracking_ids: List of tracking IDs for the ground truth objects
        pass

    def evaluate(self):
        frame_indices = self.get_viable_samples()
        samples = self.load_samples(frame_indices)

        # estimate 2D and 3D bounding boxes 
        bbox2d_list, bbox3d_list = self.bbox_estimation.estimate(samples)

        # get ground truth 2D, 3D bounding boxes, and tracking ids from data
        bbox2d_labels, bbox3d_labels, tracking_ids = self.extract_labels(samples)

        # Match estimated and ground truth bounding boxes
        # matched_ids is a dictionary where:
        # - keys are the estimated bounding box ids (indices in the estimated bbox list)
        # - values are the corresponding ground truth ids
        matched_ids = self.bbox_matching.matching(bbox2d_list, bbox3d_list, bbox2d_labels, bbox3d_labels, tracking_ids)

        # Compute errors between estimaed and ground truth bounding boxes
        errors_2d, errors_3d = self.compute_bbox_errors(bbox2d_list, bbox3d_list, bbox2d_labels, bbox3d_labels, matched_ids)

        # Process and analyze the errors
        mean_error_2d = np.mean(errors_2d)
        mean_error_3d = np.mean(errors_3d)
        
        return mean_error_2d, mean_error_3d
    
    def visualize_point_cloud(self):
        frame_indices = self.get_viable_samples()
        samples = self.load_samples(frame_indices)
        
        for sample in samples:
            # let's get the 3d point cloud and visualize it :)
            frame = sample.get_img()
            depth = get_depth_estimate(frame, self.scaled_intrinsics, self.model, self.mean, self.std,
                                        self.pad_info, self.input_size, self.padding, self.pad_h, self.pad_w, self.pad_h_half, self.pad_w_half)
            points_3d = get_3d_points(depth, self.inv_camera_matrix, self.extrinsics)

            # Create a PointCloud object
            point_cloud = o3d.geometry.PointCloud()

            # swap the channels
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame.ndim == 3:  # H x W x 3
                frame = frame.reshape(-1, 3)

            # Normalize frame values if necessary (assuming frame has values 0-255)
            frame_normalized = frame / 255.0

            # Ensure the number of colors matches the number of points
            assert (
                points_3d.shape[0] == frame_normalized.shape[0]
            ), "Points and colors must match in size."

            # Assign points to the point cloud
            point_cloud.points = o3d.utility.Vector3dVector(
                points_3d.cpu().numpy()
            )
            point_cloud.colors = o3d.utility.Vector3dVector(
                frame_normalized
            )

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )

            # occupancy grid reference points
            reference_pts = [
                o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1, origin=[self.grid_x_min, self.grid_y_min, 0]
                ),
                o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1, origin=[self.grid_x_min, self.grid_y_max, 0]
                ),
                o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1, origin=[self.grid_x_max, self.grid_y_min, 0]
                ),
                o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1, origin=[self.grid_x_max, self.grid_y_max, 0]
                ),
            ]

            # To visualize the point cloud
            o3d.visualization.draw_geometries(
                [point_cloud, axis] + reference_pts
            )
        
        # save selected samples
        self.save_selected_samples(samples)
    
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
                
            # filter samples for debug :)
            objects = [obj for obj in objects if obj['classId'] == 'Pedestrian']
            # objects = [obj for obj in objects if obj['instanceId'] == 'Pedestrian:241']
            
            sample = Sample(rgb_image_filepath=image_path, objects=objects)
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
    config_filepath = args.parse_args().config
    evaluator = ObjectLocalizationEvaluator(config_filepath)
    evaluator.evaluate()