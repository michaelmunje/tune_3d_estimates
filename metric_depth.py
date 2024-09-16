import numpy as np
import cv2
import torch 
from typing import List, Tuple

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
    swapped_points = torch.stack([points_3d[:, 2],  # Z (height) becomes X (depth)
                                  points_3d[:, 0],  # X becomes Y
                                  points_3d[:, 1]], # Y becomes Z
                                 dim=1)
    
    return swapped_points

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