import cv2
import torch
import numpy as np
from typing import List, Tuple
from geometry import projectPointsWithDist, get_pointsinfov_mask, get_3dbbox_corners
import os
import pickle
import math

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
        
def localize_position_wrt_initial_pose(future_position, initial_position, initial_yaw):
    rotmat = yaw_rotmat(initial_yaw)
    if future_position.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif future_position.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (future_position - initial_position).dot(rotmat)

class BEVPose:
    def __init__(self, x: float, y: float, yaw: float):
        self.x = x
        self.y = y
        self.yaw = yaw

    # minus
    def __sub__(self, other):
        delta_yaw = self.yaw - other.yaw
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        return BEVPose(self.x - other.x, self.y - other.y, delta_yaw)
    
    # plus
    def __add__(self, other):
        delta_yaw = self.yaw + other.yaw
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        return BEVPose(self.x + other.x, self.y + other.y, delta_yaw)

    def get_position_np(self):
        return np.array([self.x, self.y])
    
    def __repr__(self):
        return f"BEVPose(x={self.x}, y={self.y}, yaw={self.yaw})"

#  transform bev coordinates w.r.t. current pose to w.r.t initial pose
def get_bev_pose_wrt_initial_pose(bev_pose: BEVPose, bev_reference_pose: BEVPose, bev_target_pose: BEVPose) -> BEVPose:
    initial_yaw = bev_target_pose.yaw
    delta_bev_pose = bev_reference_pose - bev_target_pose
    delta_x, delta_y, delta_yaw = delta_bev_pose.x, delta_bev_pose.y, delta_bev_pose.yaw
    
    R = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)], 
                    [np.sin(delta_yaw), np.cos(delta_yaw)]])
    
    R2 = np.array([[np.cos(initial_yaw), np.sin(initial_yaw)], 
                    [-np.sin(initial_yaw), np.cos(initial_yaw)]])
    
    T = np.array([delta_x, delta_y])
    
    bev_position = bev_pose.get_position_np()
    bev_coords = R @ bev_position + (R2 @ T)
    transformed_yaw = bev_pose.yaw + delta_yaw if bev_pose.yaw is not None else None

    assert not np.isnan(bev_coords).any(), f'Nans in bev_coords: {bev_coords}'
    
    return BEVPose(bev_coords[0], bev_coords[1], transformed_yaw)

class Trajectory:
    def __init__(self, bev_poses: List[BEVPose], corresponding_timesteps: List[int], possible_timesteps: List[int], id: str, localize: bool):
        assert len(bev_poses) == len(corresponding_timesteps), 'Number of bev poses and corresponding timesteps do not match'
        assert len(corresponding_timesteps) <= len(possible_timesteps), 'Number of corresponding timesteps should be less than or equal to possible timesteps'
        self.bev_poses = bev_poses
        if localize:
            initial_position = self.bev_poses[0].get_position_np()
            initial_yaw = self.bev_poses[0].yaw
            for i in range(len(self.bev_poses)):
                self.bev_poses[i].x, self.bev_poses[i].y = localize_position_wrt_initial_pose(
                    self.bev_poses[i].get_position_np(), 
                    initial_position, 
                    initial_yaw
                )
                self.bev_poses[i].yaw = self.bev_poses[i].yaw - initial_yaw
                self.bev_poses[i].yaw = self.bev_poses[i].yaw if self.bev_poses[i].yaw < np.pi else self.bev_poses[i].yaw - 2 * np.pi

        assert len(corresponding_timesteps) == len(self.bev_poses), 'Number of timesteps and bev poses do not match'
        self.corresponding_timesteps = corresponding_timesteps
        self.possible_timesteps = possible_timesteps
        self.id = id # can be robot, track id, coda id, etc.

    def get_timestep(self, idx: int):
        return self.corresponding_timesteps[idx]
    
    def get_timestep_idx(self, timestep: int):
        assert timestep in self.corresponding_timesteps, f'Timestep {timestep} not found in corresponding timesteps'
        return self.corresponding_timesteps.index(timestep)
    
    def get_pose_at_timestep(self, timestep: int):
        return self.bev_poses[self.get_timestep_idx(timestep)]
    
    def get_discontinuities(self) -> List[Tuple[int, int]]:
        # only care about them between first seen timestep and last seen timestep
        discontinuities: List[Tuple[int, int]] = []
        prev_timestep = self.corresponding_timesteps[0]
        for i in range(1, len(self.corresponding_timesteps)):
            next_timestep = self.corresponding_timesteps[i]
            prev_timestep_idx = self.possible_timesteps.index(prev_timestep)
            next_timestep_idx = self.possible_timesteps.index(next_timestep)
            if next_timestep_idx - prev_timestep_idx > 1:
                discontinuities.append((prev_timestep, next_timestep))
            prev_timestep = next_timestep
        return discontinuities
    
    def has_discontinuities(self) -> bool:
        return len(self.get_discontinuities()) > 0
    
    def interpolate_all_missing_poses(self):
        while self.has_discontinuities():
            discontinuities = self.get_discontinuities()
            # get first discontinuity
            prev_filled_timestep, next_filled_timestep = discontinuities[0]
            self.interpolate_missing_pose(prev_filled_timestep, next_filled_timestep)
    
    def interpolate_missing_pose(self, prev_timestep: int, next_timestep: int):
        prev_idx = self.get_timestep_idx(prev_timestep)
        target_idx = prev_idx + 1
        
        assert next_timestep - prev_timestep > 1, 'Two consecutive timesteps should have at least 1 frame gap'

        next_pose = self.get_pose_at_timestep(next_timestep)
        prev_pose = self.get_pose_at_timestep(prev_timestep)
        self.bev_poses.insert(target_idx, BEVPose(
            (next_pose.x + prev_pose.x) / 2,
            (next_pose.y + prev_pose.y) / 2,
            (next_pose.yaw + prev_pose.yaw) / 2
        ))
        # correct yaw to make sure valid range
        self.bev_poses[target_idx].yaw = self.bev_poses[target_idx].yaw if self.bev_poses[target_idx].yaw < np.pi else self.bev_poses[target_idx].yaw - 2 * np.pi
        
        # interpolate in the middle (or close to the middle) of the two timesteps
        # find middle timestep from possible timesteps
        next_timestep_idx = self.possible_timesteps.index(next_timestep)
        prev_timestep_idx = self.possible_timesteps.index(prev_timestep)
        new_timestep_idx = (next_timestep_idx + prev_timestep_idx) // 2
        new_timestep = self.possible_timesteps[new_timestep_idx]
        self.corresponding_timesteps.insert(target_idx, new_timestep)

        
    def kalman_smooth(self, speed_guess: float = 0.325, process_noise: float = 1.0, measurement_noise: float = 2.0):
        """
        Smooth the trajectory using a Extended Kalman filter.
        
        Parameters:
        - speed_guess: A rough guess of the speed between poses.
        - process_noise: The process noise covariance, representing how much uncertainty there is in the dynamics.
        - measurement_noise: The measurement noise covariance, representing how noisy the position measurements are.
        
        Returns:
        - A new Trajectory object with smoothed BEVPose positions.
        """
        # Initialize state vector: [x, y, vx, vy]
        n = len(self.bev_poses)
        smoothed_poses = []

        # State [x, y, vx, vy] (positions and velocities)
        state = np.array([self.bev_poses[0].x, self.bev_poses[0].y, self.bev_poses[0].yaw, speed_guess])
        
        # State covariance matrix
        state_cov = np.eye(4)

        # Process noise
        Q = process_noise * np.eye(4)

        # Measurement matrix (we measure x and y positions only)
        H = np.array([[1, 0, 0, 0],  # we measure x
                      [0, 1, 0, 0]]) # we measure y
        
        # Measurement noise covariance matrix
        R = measurement_noise * np.eye(2)

        # Identity matrix for updating
        I = np.eye(4)

        # Time step (assuming constant time step between poses)
        dt = 0.25  # You may need to adjust this based on your data

        # Process model matrix (position updates with velocity)
        A = np.array([[1, 0, 1, 0],  # x' = x + vx
                      [0, 1, 0, 1],  # y' = y + vy
                      [0, 0, 1, 0],  # vx' = vx
                      [0, 0, 0, 1]]) # vy' = vy

        # Process noise (assuming noise in the velocity component)
        Q = process_noise * np.eye(4)

        for i in range(n):
            # Get the current measurement (position)
            z = np.array([self.bev_poses[i].x, self.bev_poses[i].y])

            # Prediction step (non-linear)
            x, y, yaw, v = state
            state_pred = np.array([
                x + v * np.cos(yaw) * dt,
                y + v * np.sin(yaw) * dt,
                yaw,
                v
            ])

            # Jacobian of the state transition function
            F = np.array([
                [1, 0, -v * np.sin(yaw) * dt, np.cos(yaw) * dt],
                [0, 1,  v * np.cos(yaw) * dt, np.sin(yaw) * dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # Covariance prediction
            state_cov = F @ state_cov @ F.T + Q

            # Kalman gain calculation
            S = H @ state_cov @ H.T + R  # residual covariance
            K = state_cov @ H.T @ np.linalg.inv(S)  # Kalman gain

            # Update step
            y = z - H @ state_pred  # measurement residual
            state = state_pred + K @ y  # state update
            state_cov = (I - K @ H) @ state_cov  # covariance update

            # Store the smoothed BEVPose
            smoothed_poses.append(BEVPose(state[0], state[1], state[2]))

        self.bev_poses = smoothed_poses

    def estimate_yaws(self):
        if len(self.bev_poses) == 1:
            self.bev_poses[0].yaw = None
            return

        # use the difference between consecutive poses to estimate the yaw
        for i in range(len(self.bev_poses) - 1):
            next_position = self.bev_poses[i + 1].get_position_np()
            current_position = self.bev_poses[i].get_position_np()
            displacement = next_position - current_position
            self.bev_poses[i].yaw = np.arctan2(displacement[1], displacement[0])
        self.bev_poses[-1].yaw = self.bev_poses[-2].yaw

    def __repr__(self):
        return f"Trajectory(poses={self.bev_poses})"
    
    def __len__(self):
        return len(self.bev_poses)
    
def transform_trajectory_to_initial_pose(traj: Trajectory, reference_frame_traj: Trajectory):
    initial_reference_pose = reference_frame_traj.bev_poses[0]
    for i in range(len(traj.bev_poses)):
        current_timestep = traj.corresponding_timesteps[i]
        reference_pose = reference_frame_traj.get_pose_at_timestep(current_timestep)
        traj.bev_poses[i] = get_bev_pose_wrt_initial_pose(
            traj.bev_poses[i], 
            reference_pose, 
            initial_reference_pose
        )

def get_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return camera_matrix

def get_inverse_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    camera_matrix = get_camera_matrix(intrinsics)
    inv_camera_matrix = torch.inverse(camera_matrix)
    return inv_camera_matrix

def quaternion_to_yaw(qw, qx, qy, qz):
    # Yaw calculation from quaternion
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw

class LocationWith2DBBox:
    def __init__(self, x: float, y: float, w: float, h: float,
                 cX: float, cY: float, cZ: float,
                 tracking_id: str,
                 sample_filepath: str):
        
        # 2b bounding box params
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
        # 3d location params
        self.cX = cX
        self.cY = cY
        self.cZ = cZ

        self.tracking_id = tracking_id
        self.sample_filepath = sample_filepath
        
    def get_bev(self):
        return BEVPose(self.cX, self.cY, yaw=None)

class LocationWith3dBBox:
    def __init__(self, cX: float, cY: float, cZ: float, 
                 w: float, h: float, l: float, 
                 r:float, p:float, y: float,
                 sample_filepath: str,
                 tracking_id: str):
        self.h = h
        self.w = w
        self.l = l
        self.r = r
        self.p = p
        self.y = y
        
        self.cX = cX
        self.cY = cY
        self.cZ = cZ
        
        self.sample_filepath = sample_filepath
        self.tracking_id: str = tracking_id

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

    def get_bev(self):
        return BEVPose(self.cX, self.cY, yaw=None)
        
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
            return LocationWith2DBBox(x=-1, y=-1, w=0, h=0,
                  cX=self.cX, cY=self.cY, cZ=self.cZ, tracking_id=self.tracking_id, sample_filepath=self.sample_filepath)
        
        y_max = np.max(image_points[:, 0]).astype(int)
        y_min = np.min(image_points[:, 0]).astype(int)
        x_max = np.max(image_points[:, 1]).astype(int)
        x_min = np.min(image_points[:, 1]).astype(int)
        
        x, y = x_min, y_min
        w, h = x_max - x_min, y_max - y_min
        
        return LocationWith2DBBox(x=x, y=y, w=w, h=h, cX=self.cX, cY=self.cY, cZ=self.cZ, tracking_id=self.tracking_id, sample_filepath=self.sample_filepath)
    
def compute_2d_iou(bbox1: LocationWith2DBBox, bbox2: LocationWith2DBBox) -> float:
    """Compute the Intersection over Union (IoU) for 2D bounding boxes."""
    x1 = max(bbox1.x, bbox2.x)
    y1 = max(bbox1.y, bbox2.y)
    x2 = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
    y2 = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1.w * bbox1.h
    bbox2_area = bbox2.w * bbox2.h

    # Avoid division by zero
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou
    
def compute_centroid_error(bbox1: LocationWith3dBBox, bbox2: LocationWith3dBBox) -> float:
    """Compute the error between the centroids of 3D bounding boxes."""
    return np.sqrt(
        (bbox1.cX - bbox2.cX) ** 2 +
        (bbox1.cY - bbox2.cY) ** 2 +
        (bbox1.cZ - bbox2.cZ) ** 2
    )

class Entity:
    def __init__(self, data: dict, filepath: str):
        self.name = data['classId']
        self.id = data['instanceId']
        self.bbox3d = LocationWith3dBBox(
            data['cX'],
            data['cY'],
            data['cZ'],
            data['w'],
            data['h'],
            data['l'],
            data['r'],
            data['p'],
            data['y'],
            filepath,
            data['instanceId']
        )
        self.filepath = filepath

class Sample:
    def __init__(self, rgb_image_filepath: str, objects: dict, lart_folder: str):
        self.rgb_image_filepath: str = rgb_image_filepath
        self.objects: List[Entity] = [Entity(obj, rgb_image_filepath) for obj in objects]
        # also create mapping from instanceId to Entity
        self.instance_id_to_entity = {obj.id: obj for obj in self.objects}
        self.lart_folder = lart_folder
        
    def get_sample_idx(self):
        return int(self.rgb_image_filepath.split('/')[-1].replace('.png', '').split('_')[-1])

    def get_img(self):
        return cv2.imread(self.rgb_image_filepath)
    
    def get_lart_data(self):
        rgb_image_filename = self.rgb_image_filepath.split('/')[-1]
        lart_filename = rgb_image_filename.replace('.png', '.pkl')
        lart_filepath = os.path.join(self.lart_folder, lart_filename)
        with open(lart_filepath, 'rb') as f:
            lart_data = pickle.load(f)
        # lart data is a dictionary
        # key is the tracking id
        return lart_data
    
class Estimate:
    def __init__(self, x, y, z, lart_tracking_id, lart_bounding_box):
        self.x = x
        self.y = y
        self.z = z
        self.lart_tracking_id = lart_tracking_id
        self.lart_bounding_box = lart_bounding_box