import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Dict
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.append(os.getcwd())

def save_img_bev_gif(bev_filepaths: List[str], sample_save_folder: str):
    frame_duration = 500
    seq_bev_basename = os.path.basename(bev_filepaths[0]).replace('.png', '')
    seq_bev_out = os.path.join(sample_save_folder, f'{seq_bev_basename}.gif')
                
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

def visualize_2d_bbox_on_image(image, x, y, w, h, color):
    color = tuple(int(c * 255) for c in color)
    # change to bgr
    color = color[::-1]
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 6)

def save_bev_visualization(current_robot_pose,
                            object_estimate_trajectories, object_label_trajectories,
                            all_loc_estimates, all_loc_labels,
                            current_sample_filepath, matched_ids, fp, 
                            tracking_id_colors: Dict[int, Tuple[float, float, float]],
                            grid_x_min: float, grid_x_max: float, grid_y_min: float, grid_y_max: float,
                            estimator_name: str,
                            include_2d_bbox_estimate_visualization: bool = True,
                            include_3d_bbox_label_visualization: bool = False):
    
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
                    visualize_2d_bbox_on_image(image, x, y, w, h, color)
        
        if include_3d_bbox_label_visualization:
            raise NotImplementedError
            # # visualize the 3d bbox on the image
            # for loc_estimate in loc_estimates:
            #     loc_estimate.visualize_3d_bbox()
            
        # let's also plot bevs
        visualize_bev_plot(current_robot_pose, corresponding_estimates_and_labels,
                           grid_x_min, grid_x_max, grid_y_min, grid_y_max, estimator_name)
        
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

def get_endpoint_diff(start, yaw, length):
    # Calculate the direction vector (dx, dy)
    dx = np.cos(yaw)
    dy = np.sin(yaw)
    
    # Normalize the direction vector to ensure its Euclidean length is 1
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm
    
    return dx, dy

def visualize_bev_plot(current_robot_pose, corresponding_estimates_and_labels_and_colors,
                       grid_x_min: float, grid_x_max: float, grid_y_min: float, grid_y_max: float,
                       estimator_name: str):
    # each estimate (and its corresponding label) should have a different color
    # now let's plot them
    fig, ax = plt.subplots()
    fixed_arrow_length = 0.5
    for bev_estimate, bev_label, color in corresponding_estimates_and_labels_and_colors:
        ax.plot(bev_estimate.y, bev_estimate.x, '*', color=color, markersize=10)
        ax.plot(bev_label.y, bev_label.x, 'o', color=color, markersize=8)
        
        if bev_estimate.yaw is not None:
            bev_estimate_pos = bev_estimate.get_position_np()
            end_bev_estimate = get_endpoint_diff(bev_estimate_pos, bev_estimate.yaw, fixed_arrow_length)
            
            # Draw arrow from the current bev_estimate in the flipped direction (away from previous)
            ax.arrow(bev_estimate_pos[1], bev_estimate_pos[0], end_bev_estimate[1], end_bev_estimate[0], 
                    width=0.1, fc=color, ec=color)
            
        if bev_label.yaw is not None:
            bev_label_pos = bev_label.get_position_np()
            end_bev_label = get_endpoint_diff(bev_label_pos, bev_label.yaw, fixed_arrow_length)

            # Draw arrow from the current bev_label in the flipped direction (away from previous)
            # compute dist between the points
            # assert np.linalg.norm(np.array(bev_label_pos) - np.array([end_bev_label])) <= fixed_arrow_length, f'bev_label_pos: {bev_label_pos}, end_bev_label_pos: {end_bev_label}'
            ax.arrow(bev_label_pos[1], bev_label_pos[0], end_bev_label[1], end_bev_label[0], 
                    width=0.1, fc=color, ec=color)
    # also plot the trajectory
    ax.plot(current_robot_pose.y, current_robot_pose.x, 's', color='black', markersize=5)

    current_robot_position = current_robot_pose.get_position_np()
    end_robot_pose = get_endpoint_diff(current_robot_position, current_robot_pose.yaw, fixed_arrow_length)

    ax.arrow(current_robot_position[1], current_robot_position[0], end_robot_pose[1], end_robot_pose[0], 
            width=0.1, fc='black', ec='black')
        
    # expected grid limits
    ax.set_xlim(grid_y_min, grid_y_max)
    ax.set_ylim(grid_x_min, grid_x_max)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.grid(True)
    
    # Custom legend for markers
    estimated_legend = mlines.Line2D([], [], color='black', marker='*', linestyle='None', label='Estimate', markersize=10, markerfacecolor='none')
    pseudo_gt_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Pseudo-Ground Truth', markersize=8, markerfacecolor='none')
    # add legend for robot position
    robot_legend = mlines.Line2D([], [], color='black', marker='s', linestyle='None', label='Robot', markersize=5)
    ax.legend(handles=[estimated_legend, pseudo_gt_legend, robot_legend], loc='upper left')
    
    ax.set_title(f'BEV Visualization: {estimator_name}')
    ax.set_ylabel('X (forward)')
    ax.set_xlabel('Y (left)')
    ax.set_aspect('equal', 'box')
    plt.show()
