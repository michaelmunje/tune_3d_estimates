from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from structures import Sample, LocationWith2DBBox, LocationWith3dBBox
import torch
import numpy as np
from structures import get_camera_matrix
from metric_depth import get_depth_estimate, get_3d_points, get_3d_estimation_in_bbox
import open3d as o3d
import cv2
import os
import tqdm

class BBoxAndBevEstimation(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.input_width = config["input_width"]
        self.input_height = config["input_height"]
        h, w = self.input_height, self.input_width
        self.input_size = (h, w)
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
        
        self.camera_matrix_np = np.array(self.camera_matrix).astype(np.float64)
        self.distortion_coeffs_np = np.array(self.distortion_coeffs).astype(np.float64)
        self.extrinsics_np = self.extrinsics.cpu().numpy().astype(np.float64)
                
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
        
    @abstractmethod
    def estimate(self, samples: List[Sample]) -> List[List[LocationWith2DBBox]]:
        pass
    
class LartBBoxAndBevEstimation(BBoxAndBevEstimation):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lart_folder = config['lart_folder']
        
    def estimate(self, samples: List[Sample]) -> List[List[LocationWith2DBBox]]:
        all_estimated_locations: List[List[LocationWith2DBBox]] = []
        # get lart data
        for sample in samples:
            lart_data = sample.get_lart_data()
            estimated_locations: List[LocationWith2DBBox] = []
            # get the bbox2d and bev location from the lart data
            for tracking_id in lart_data:
                estimate = lart_data[tracking_id]

                bbox2d = estimate['bbox']
                joint_coords = estimate['avg_joint_coordinates']
                
                x, y, w, h = bbox2d
                
                if w == 0 or h == 0:
                    continue
                
                estimated_location = LocationWith2DBBox(x, y, w, h, joint_coords[2], -joint_coords[0], -joint_coords[1], tracking_id, sample.rgb_image_filepath)
                estimated_locations.append(estimated_location)
            all_estimated_locations.append(estimated_locations)
        return all_estimated_locations
    
# here we use monocular depth estimation to get the bev location
# and we get the 2d bbox from lart data
class MDEBBoxAndBevEstimation(BBoxAndBevEstimation):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.metric_3d_model = config["metric_3d_model"]

        if "vit" in self.metric_3d_model:
            self.metric_3d_input_size = (616, 1064)  # for vit model
        else:
            self.metric_3d_input_size = (544, 1216)  # for convnext model

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
        h, w = self.input_size
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

        # Load the metric depth estimation model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module = torch.hub.load(
            "yvanyin/metric3d",
            self.metric_3d_model,
            pretrain=True,
            map_location=torch.device("cpu"),
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        
    def estimate(self, samples: List[Sample]) -> List[List[LocationWith2DBBox]]:
        all_estimated_locations: List[List[LocationWith2DBBox]] = []
        
        cache_dir = 'depth_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        print("Running MDEBBoxAndBevEstimation")

        for sample in tqdm.tqdm(samples):
            estimated_locations: List[LocationWith2DBBox] = []
            
            frame_filepath = sample.rgb_image_filepath
            file_basename = os.path.basename(frame_filepath)
            file_basename = file_basename.split('.')[0]
            cache_filepath = f"{cache_dir}/depth_estimate_{file_basename}.npy"
            if not os.path.exists(cache_filepath):
                depth = get_depth_estimate(sample.get_img(), self.scaled_intrinsics, self.model, self.mean, self.std,
                                            self.pad_info, self.input_size, self.padding, self.pad_h, self.pad_w, self.pad_h_half, self.pad_w_half)
                np.save(cache_filepath, depth.cpu().numpy())
            else:
                depth = np.load(cache_filepath)
                depth = torch.from_numpy(depth).to(self.device)
            # get the bev location from the depth estimation
            points_3d = get_3d_points(depth, self.inv_camera_matrix, self.extrinsics)
            
            lart_data = sample.get_lart_data()
            for tracking_id in lart_data:
                estimate = lart_data[tracking_id]
                
                bbox2d = estimate['bbox']
                
                x, y, w, h = bbox2d
                
                if w == 0 or h == 0:
                    continue
                
                avg_3d_location = get_3d_estimation_in_bbox(frame=sample.get_img(), 
                                                            point_cloud=points_3d, 
                                                            bbox=bbox2d)
                avg_3d_location = avg_3d_location.cpu().numpy()
                estimated_location = LocationWith2DBBox(x, y, w, h, avg_3d_location[0], avg_3d_location[1], avg_3d_location[2], tracking_id, sample.rgb_image_filepath)
                estimated_locations.append(estimated_location)
            all_estimated_locations.append(estimated_locations)
        return all_estimated_locations
    
    def visualize_point_cloud(self, samples: List[Sample]):
        cache_dir = 'depth_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        for sample in samples:
            # let's get the 3d point cloud and visualize it :)
            frame = sample.get_img()
            # check cache for depth estimate
            frame_filepath = sample.rgb_image_filepath
            file_basename = os.path.basename(frame_filepath)
            file_basename = file_basename.split('.')[0]
            cache_filepath = f"{cache_dir}/depth_estimate_{file_basename}.npy"
            if not os.path.exists(cache_filepath):
                depth = get_depth_estimate(frame, self.scaled_intrinsics, self.model, self.mean, self.std,
                                            self.pad_info, self.input_size, self.padding, self.pad_h, self.pad_w, self.pad_h_half, self.pad_w_half)
                np.save(cache_filepath, depth.cpu().numpy())
            else:
                depth = np.load(cache_filepath)
                depth = torch.from_numpy(depth).to(self.device)
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