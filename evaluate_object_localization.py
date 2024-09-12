
import yaml
import torch
import cv2
import numpy as np
import json
import os
import sys
import argparse
from typing import List, Tuple
import open3d as o3d

sys.path.append(os.getcwd())
from helpers.geometry import projectPointsWithDist, get_pointsinfov_mask, get_3dbbox_corners

# @torch.jit.script
def model_inference(model: torch._dynamo.eval_frame.OptimizedModule, rgb: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
    return pred_depth

@torch.jit.script
def postprocess_depth(pred_depth: torch.Tensor, pad_info: List[int], input_size: Tuple[int, int], intrinsic: List[float]) -> torch.Tensor:
    # Unpadding
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]:pred_depth.shape[0] - pad_info[1], pad_info[2]:pred_depth.shape[1] - pad_info[3]]

    # Upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], input_size, mode='bilinear').squeeze()

    # De-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of the canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # Now the depth is metric

    # Clamp the depth values to a maximum of 300 meters
    pred_depth = torch.clamp(pred_depth, 0, 300)

    return pred_depth

# @torch.jit.script
def frame_np_to_torch_with_pad_info(frame, mean: torch.Tensor, std: torch.Tensor, 
                                    padding: List[float], pad_h: int, pad_w: int, pad_h_half: int, pad_w_half: int) -> torch.Tensor:
    rgb = frame[:, :, ::-1] # not sure if necessary -- need to check
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :]
    rgb = rgb.cuda() if torch.cuda.is_available() else rgb
    return rgb

def get_depth_estimate(frames: List[np.array], intrinsics: List[float], model: torch._dynamo.eval_frame.OptimizedModule,
                       mean: torch.Tensor, std: torch.Tensor,  
                       pad_info: List[float], input_size: Tuple[int, int],
                       padding: List[float], pad_h: int, pad_w: int, pad_h_half: int, pad_w_half: int) -> torch.Tensor:
    batch_rgb = []
    
    if type(frames) != list:
        frames = [frames]

    for frame in frames:
        rgb = frame_np_to_torch_with_pad_info(frame, mean, std, padding, pad_h, pad_w, pad_h_half, pad_w_half)
        batch_rgb.append(rgb)
        
    batch_rgb = torch.cat(batch_rgb, dim=0)

    # pred_depths = [model_inference(model, rgb) for rgb in batch_rgb]
    
    pred_depths = model_inference(model, batch_rgb)

    processed_depths = []
    for i in range(len(frames)):
        pred_depth = postprocess_depth(pred_depths[i], pad_info, input_size, intrinsics)
        processed_depths.append(pred_depth)
        
    avg_depth = torch.stack(processed_depths).mean(dim=0)

    return avg_depth

@torch.jit.script
def swap_camera_axes(points_3d: torch.Tensor) -> torch.Tensor:
    """
    Swaps the axes of the 3D points so that:
    - X-axis becomes depth distance
    - Z-axis becomes height
    - Y-axis remains unchanged

    Parameters:
    - points_3d (torch.Tensor): 3D points of shape (N, 3).

    Returns:
    - torch.Tensor: 3D points with swapped axes of shape (N, 3).
    """
    
    # Swap the axes
    # swapped_points = torch.stack([points_3d[:, 2],  # Z (height) becomes X (depth)
    #                               points_3d[:, 1],  # X becomes Y
    #                               points_3d[:, 0]], # Y becomes Z
    #                              dim=1)
    
    swapped_points = torch.stack([points_3d[:, 2],  # Z (height) becomes X (depth)
                                  points_3d[:, 0],  # X becomes Y
                                  points_3d[:, 1]], # Y becomes Z
                                 dim=1)
    
    return swapped_points

@torch.jit.script
def get_3d_points(depth: torch.Tensor, 
                  inv_camera_matrix: torch.Tensor, 
                  extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Transforms depth map into 3D points in world coordinates.

    Parameters:
    - depth (torch.Tensor): Depth map of shape (H, W).
    - inv_camera_matrix (torch.Tensor): Inverse camera intrinsic matrix of shape (3, 3).
    - extrinsics (torch.Tensor): Extrinsic matrix of shape (4, 4) for transforming from camera to world coordinates.

    Returns:
    - torch.Tensor: 3D points in world coordinates of shape (H * W, 3).
    """
    
    # Get height and width of the depth map
    # depth = depth.T
    H, W = depth.shape
    # depth = torch.ones_like(depth)

    # Create a meshgrid for pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
    
    # Flatten x, y and depth
    x = x.flatten()
    y = y.flatten()
    depth_flat = depth.flatten()
    
    # Create homogeneous pixel coordinates
    homogeneous_pixel_coords = torch.stack((x, y, torch.ones_like(x)), dim=1).float()

    # Transform pixel coordinates to camera coordinates
    camera_coords = inv_camera_matrix @ homogeneous_pixel_coords.T
    camera_coords = camera_coords.T  # shape (H*W, 3)
    
    # Reverse the sign of the depth values if depth is inversely related to distance
    # Comment out the line below if depth directly represents distance
    depth_flat = depth_flat
    
    # Multiply by the depth to get 3D points in camera coordinates
    camera_coords *= depth_flat.unsqueeze(1)
    
    M = torch.eye(3, device=depth.device)
    M[0, 0] = -1.0
    M[1, 1] = -1.0
    camera_coords = torch.matmul(M, camera_coords.T).T

    # Convert camera coordinates to homogeneous coordinates (H*W, 4)
    camera_coords_homogeneous = torch.cat([camera_coords, torch.ones((camera_coords.shape[0], 1), device=depth.device)], dim=1)

    # world points without extrinsics
    world_coords = camera_coords_homogeneous[:, :3]
    world_coords = swap_camera_axes(world_coords)
    world_coords += extrinsics[:3, 3].to(depth.device)

    # Transform 3D points from camera to world coordinates using the extrinsic matrix
    # world_coords_homogeneous = (extrinsics @ camera_coords_homogeneous.T).T
    
    # Discard the homogeneous coordinate to get the final 3D points in world coordinates
    # world_coords = camera_coords_homogeneous[:, :3]

    return world_coords

def get_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return camera_matrix

def get_inverse_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    camera_matrix = get_camera_matrix(intrinsics)
    inv_camera_matrix = torch.inverse(camera_matrix)
    return inv_camera_matrix

# save 2d bounding box and also 3d location
class Bbox2d:
    def __init__(self, x: float, y: float, w: float, h: float,
                 cX: float, cY: float, cZ: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cX = cX
        self.cY = cY
        self.cZ = cZ

class Bbox3d:
    def __init__(self, cX: float, cY: float, cZ: float, 
                 w: float, h: float, l: float, 
                 r:float, p:float, y: float):
        self.cX = cX
        self.cY = cY
        self.cZ = cZ
        self.h = h
        self.w = w
        self.l = l
        self.r = r
        self.p = p
        self.y = y
    
    def to_dict(self):
        return {
            'cX': self.cX,
            'cY': self.cY,
            'cZ': self.cZ,
            'w': self.w,
            'h': self.h,
            'l': self.l,
            'r': self.r,
            'p': self.p,
            'y': self.y
        }
        
    # requires projecting the 3D bounding box to 2D
    def get_bbox2d(self, instrinsics_mat: np.array, distortion_coeffs: np.array, extrinsics: np.array):
        
        assert type(instrinsics_mat) == np.ndarray
        assert type(distortion_coeffs) == np.ndarray
        assert type(extrinsics) == np.ndarray
        
        # get corner points based on center points and dimensions
        x_min, x_max = self.cX - self.w/2, self.cX + self.w/2
        y_min, y_max = self.cY - self.l/2, self.cY + self.l/2
        z_min, z_max = self.cZ - self.h/2, self.cZ + self.h/2
        
        pc_np = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max]
        ])
        
        ext_homo_mat = extrinsics

        np.set_printoptions(suppress=True)
        #Load projection, rectification, distortion camera matrices

        K   = instrinsics_mat
        d   = distortion_coeffs
        
        ext_homo_mat = np.array([[-3.2457400e-02, -9.9947312e-01, -5.0000000e-08,  8.0000000e-02],
                    [-1.4228826e-01,  4.6207900e-03, -9.8981448e-01, -1.0000000e-01],
                    [ 9.8929296e-01, -3.2126800e-02, -1.4236327e-01, -3.0000000e-02],
                    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

        image_points = projectPointsWithDist(pc_np[:, :3].astype(np.float64), ext_homo_mat[:3, :3], 
            ext_homo_mat[:3, 3], K, d)

        valid_points_mask = get_pointsinfov_mask(
            (ext_homo_mat[:3, :3] @ pc_np[:, :3].T).T + ext_homo_mat[:3, 3])
        
        assert len(image_points) == len(pc_np)
        
        image_points = image_points[valid_points_mask]
        
        if len(image_points) == 0:
            return Bbox2d(x=-1, y=-1, w=0, h=0,
                  cX=self.cX, cY=self.cY, cZ=self.cZ)
        
        y_max = np.max(image_points[:, 0]).astype(int)
        y_min = np.min(image_points[:, 0]).astype(int)
        x_max = np.max(image_points[:, 1]).astype(int)
        x_min = np.min(image_points[:, 1]).astype(int)
        
        x, y = x_min, y_min
        w, h = x_max - x_min, y_max - y_min
        
        return Bbox2d(x=x, y=y, w=w, h=h,
                      cX=self.cX, cY=self.cY, cZ=self.cZ)

class Entity:
    def __init__(self, data: dict):
        self.name = data['classId']
        self.id = data['instanceId']
        self.bbox3d = Bbox3d(
            data['cX'],
            data['cY'],
            data['cZ'],
            data['w'],
            data['h'],
            data['l'],
            data['r'],
            data['p'],
            data['y']
        )

class Sample:
    def __init__(self, rgb_image_filepath: str, objects: dict):
        self.rgb_image_filepath: str = rgb_image_filepath
        self.objects: List[Entity] = [Entity(obj) for obj in objects]
        # also create mapping from instanceId to Entity
        self.instance_id_to_entity = {obj.id: obj for obj in self.objects}

    def get_img(self):
        return cv2.imread(self.rgb_image_filepath)
    
class Estimate:
    def __init__(self, x, y, z, lart_tracking_id, lart_bounding_box):
        self.x = x
        self.y = y
        self.z = z
        self.lart_tracking_id = lart_tracking_id
        self.lart_bounding_box = lart_bounding_box

class ObjectLocalizationEvaluator:
    def __init__(self, config_filepath):
        # Load the configuration file
        if config_filepath:
            with open(config_filepath, "r") as file:
                config = yaml.safe_load(file)

        self.use_gamma_correction = config["use_gamma_correction"]
        self.use_gaussian_blur = config["use_gaussian_blur"]
        self.use_undistort_correction = config["use_undistort_correction"]
        self.metric_3d_model = config["metric_3d_model"]
        
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

        self.intrinsics = config["intrinsics"]
        self.inv_camera_matrix = get_inverse_camera_matrix(self.intrinsics)
        self.inv_camera_matrix = (
            self.inv_camera_matrix.cuda()
            if torch.cuda.is_available()
            else self.inv_camera_matrix
        )
        self.input_width = config["input_width"]
        self.input_height = config["input_height"]

        self.camera_matrix = np.array(
            [
                [self.intrinsics[0], 0, self.intrinsics[2]],  # fx, 0, cx
                [0, self.intrinsics[1], self.intrinsics[3]],  # 0, fy, cy
                [0, 0, 1],
            ]
        )

        self.distortion_coeffs = np.array(config["distortion_coeffs"])
        self.translation_vector = torch.tensor(config["translation_vector"])

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

        # Load the model
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

    def evaluate(self):
        frame_indices = self.get_viable_samples()
        samples = self.load_samples(frame_indices)
        
        def get_3d_estimation_in_bbox(frame, point_cloud, bbox, inner_bbox=True):
            # need some way to select the points that are in a bbox
            bbox_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            # dummy example
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            if inner_bbox: # then get inside region
                h = h // 2
                w = w // 2
                x = x + h // 2
                y = y + w // 2
            bbox_mask[x:x+w, y:y+h] = 1
            return point_cloud[bbox_mask.flatten() == 1].mean(dim=0)
        
        # ... do some evaluation here
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

        
        # we will need to use lart tracking id to coda id mapping
        
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