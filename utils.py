import cv2
import torch
import numpy as np
from typing import List, Tuple
from geometry import projectPointsWithDist, get_pointsinfov_mask, get_3dbbox_corners
import os
import pickle

def get_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return camera_matrix

def get_inverse_camera_matrix(intrinsics: List[float]) -> torch.Tensor:
    camera_matrix = get_camera_matrix(intrinsics)
    inv_camera_matrix = torch.inverse(camera_matrix)
    return inv_camera_matrix

# save 2d bounding box and also 3d location
class Location:
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
        
    def get_bev_location(self):
        return np.array([self.cX, self.cY])

class Bbox3d:
    def __init__(self, cX: float, cY: float, cZ: float, 
                 w: float, h: float, l: float, 
                 r:float, p:float, y: float,
                 sample_filepath: str,
                 tracking_id: str):
        self.cX = cX
        self.cY = cY
        self.cZ = cZ
        self.h = h
        self.w = w
        self.l = l
        self.r = r
        self.p = p
        self.y = y
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

    def get_bev_location(self):
        return np.array([self.cX, self.cY])
        
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
            return Location(x=-1, y=-1, w=0, h=0,
                  cX=self.cX, cY=self.cY, cZ=self.cZ, tracking_id=self.tracking_id, sample_filepath=self.sample_filepath)
        
        y_max = np.max(image_points[:, 0]).astype(int)
        y_min = np.min(image_points[:, 0]).astype(int)
        x_max = np.max(image_points[:, 1]).astype(int)
        x_min = np.min(image_points[:, 1]).astype(int)
        
        x, y = x_min, y_min
        w, h = x_max - x_min, y_max - y_min
        
        return Location(x=x, y=y, w=w, h=h, cX=self.cX, cY=self.cY, cZ=self.cZ, tracking_id=self.tracking_id, sample_filepath=self.sample_filepath)
    
def compute_2d_iou(bbox1: Location, bbox2: Location) -> float:
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
    
def compute_centroid_error(bbox1: Bbox3d, bbox2: Bbox3d) -> float:
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
        self.bbox3d = Bbox3d(
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