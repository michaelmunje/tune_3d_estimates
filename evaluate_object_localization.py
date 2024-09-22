import yaml
import numpy as np
import os
import sys
import argparse
from typing import List, Dict
from matplotlib import pyplot as plt

import bbox_and_bev_estimation
import bbox_matching 
from structures import Sample, quaternion_to_yaw, transform_trajectory_to_initial_pose
from metrics import average_displacement_error, final_displacement_error, angular_displacement_error, heading_deviation_error, trajectory_abs_angle_diff
from structures import BEVPose, Trajectory
from visualize import save_img_bev_gif
from data_loader import load_samples, get_estimates_and_labels_per_sample, get_object_estimate_trajectories, get_object_label_trajectories, get_track_id_colors, update_estimates_with_new_timesteps, update_labels_with_new_timesteps
from visualize import save_bev_visualization
sys.path.append(os.getcwd())


class ObjectLocalizationEvaluator:
    def __init__(self, config_filepath: str, debug_mode: bool = False):
        assert os.path.exists(config_filepath), f'Config filepath {config_filepath} does not exist'
        with open(config_filepath, "r") as file:
            config: Dict = yaml.safe_load(file)
            
        self.config = config
        self.use_gamma_correction: bool = config["use_gamma_correction"]
        self.use_gaussian_blur: bool = config["use_gaussian_blur"]
        self.use_undistort_correction: bool = config["use_undistort_correction"]
        self.metric_3d_model: str = config["metric_3d_model"]

        bbox_and_bev_estimation_method = getattr(bbox_and_bev_estimation, config["bbox_and_bev_estimation"])
        self.bbox_and_bev_estimation = bbox_and_bev_estimation_method(config)
        bbox_matching_method = getattr(bbox_matching, config["bbox_matching"])
        self.bbox_matching = bbox_matching_method(**config["bbox_matching_config"])
        
        self.lart_folder: str = config["lart_folder"]
        self.debug_mode: bool = debug_mode
        
        self.sample_save_folder: str = config["sample_save_folder"]
        if not os.path.exists(self.sample_save_folder):
            os.makedirs(self.sample_save_folder)
        self.save_visualiations: bool = True

        self.camera_matrix_np: np.ndarray = self.bbox_and_bev_estimation.camera_matrix_np
        self.distortion_coeffs_np: np.ndarray = self.bbox_and_bev_estimation.distortion_coeffs_np
        self.extrinsics_np: np.ndarray = self.bbox_and_bev_estimation.extrinsics_np
        
        self.grid_x_min: float = config["x_min"]
        self.grid_x_max: float = config["x_max"]
        self.grid_y_min: float = config["y_min"]
        self.grid_y_max: float = config["y_max"]
        self.grid_z_min: float = config["z_min"]
        self.grid_z_max: float = config["z_max"]
        self.grid_limits: List[float] = [
            self.grid_x_min,
            self.grid_x_max,
            self.grid_y_min,
            self.grid_y_max,
            self.grid_z_min,
            self.grid_z_max,
        ]
        self.traj_idx: int = config["traj_idx"]
        
        self.frame_intervals_start_indices: List[int] = config["frame_intervals_start_indices"]
        self.frame_interval_lengths: int = config["frame_interval_lengths"]
        self.frame_interval_sample_stride: int = config["frame_interval_sample_stride"]
        self.frame_sequences: List[List[int]] = []

        self.smooth_robot_trajectory: bool = config["smooth_robot_trajectory"]
        self.smooth_estimate_trajectories: bool = config["smooth_estimate_trajectories"]
        self.interpolate_between_trajectory: bool = config["interpolate_between_trajectory"]
        self.smooth_label_trajectories: bool = config["smooth_label_trajectories"]
        
        self.random_seed: float = config["random_seed"]
        np.random.seed(self.random_seed)
        
        self.set_ekf_params(randomize=False)     
        
        for start_idx in self.frame_intervals_start_indices:
            self.frame_sequences.append(list(range(start_idx, start_idx + self.frame_interval_lengths, self.frame_interval_sample_stride)))
            
        # check if the filepath exists, if not try 1 lower
        for frame_seq in self.frame_sequences:
            for i, frame_idx in enumerate(frame_seq):
                frame_idx = int(frame_idx)
                while not os.path.exists(self.frame_idx_to_img_fp(frame_idx)):
                    # print(f'Frame {frame_idx} does not exist. Trying 1 lower.')
                    frame_idx -= 1
                    assert frame_idx >= 0, 'Frame index cannot be negative.'
                frame_seq[i] = frame_idx
        
        assert type(self.traj_idx) == int, 'Trajectory index must be an integer. Otherwise, you\'ll need to modify this dict below.' 
        
        self.poses_filepath: str = f'coda-devkit/data/poses/dense_global/{self.traj_idx}.txt'
        self.poses: np.ndarray = np.loadtxt(self.poses_filepath)
        
        self.frame_idx_to_bev_pose: Dict[int, BEVPose] = {}
        for idx, pose in enumerate(self.poses):
            x, y = pose[1], pose[2]
            yaw = quaternion_to_yaw(pose[4], pose[5], pose[6], pose[7])
            self.frame_idx_to_bev_pose[int(idx)] = BEVPose(x, y, yaw)

        # for each frame sequence, create a trajectory for poses
        self.robot_trajectories: List[Trajectory] = []
        for frame_seq in self.frame_sequences:
            bevs = [self.frame_idx_to_bev_pose[frame_idx] for frame_idx in frame_seq]
            trajectory = Trajectory(bevs, frame_seq, frame_seq, id='robot', localize=True, initial_yaw_estimation=False)
            if self.smooth_robot_trajectory:
                trajectory.kalman_smooth(process_noise=self.ekf_noise, measurement_noise=self.ekf_measurement_noise)
            self.robot_trajectories.append(trajectory)
            
    def turn_off_visualizations(self):
        self.save_visualiations = False
            
    def set_ekf_params(self, randomize: bool):
        # Function to generate random EKF parameters
        def get_noise_param(config_param):
            if isinstance(config_param, list):
                return np.random.uniform(config_param[0], config_param[1])
            return config_param
        
        suffix = "_range" if randomize else ""
        
        self.ekf_x_noise = get_noise_param(self.config["ekf_x_noise" + suffix])
        self.ekf_y_noise = get_noise_param(self.config["ekf_y_noise" + suffix])
        self.ekf_yaw_noise = get_noise_param(self.config["ekf_yaw_noise" + suffix])
        self.ekf_speed_noise = get_noise_param(self.config["ekf_speed_noise" + suffix])
        self.ekf_noise = [self.ekf_x_noise, self.ekf_y_noise, self.ekf_yaw_noise, self.ekf_speed_noise]

        self.ekf_x_measurement_noise = get_noise_param(self.config["ekf_x_measurement_noise" + suffix])
        self.ekf_y_measurement_noise = get_noise_param(self.config["ekf_y_measurement_noise" + suffix])
        self.ekf_measurement_noise = [self.ekf_x_measurement_noise, self.ekf_y_measurement_noise]

    def frame_idx_to_img_fp(self, frame_idx: int) -> str:
        return f'coda-devkit/data/2d_rect/cam0/{self.traj_idx}/2d_rect_cam0_{self.traj_idx}_{frame_idx}.png'
    
    def frame_idx_to_json_fp(self, frame_idx: int) -> str:
        return f'coda-devkit/data/3d_bbox/os1/{self.traj_idx}/3d_bbox_os1_{self.traj_idx}_{frame_idx}.json'
    
    def evaluate_sequences(self):
        samples_sequences: List[List[Sample]] = load_samples(self.frame_sequences, 
                                                             self.frame_idx_to_img_fp, 
                                                             self.frame_idx_to_json_fp, 
                                                             self.lart_folder)
        loc_estimates_by_sample, loc_labels_by_sample, matched_ids = get_estimates_and_labels_per_sample(samples_sequences, 
                                                                                                         self.bbox_and_bev_estimation.estimate, 
                                                                                                         self.bbox_matching.matching)
        object_estimate_trajectories_seq = get_object_estimate_trajectories(samples_sequences, loc_estimates_by_sample, matched_ids, self.robot_trajectories)
        object_label_trajectories_seq = get_object_label_trajectories(samples_sequences, loc_labels_by_sample, matched_ids, self.robot_trajectories)
        
        assert len(object_estimate_trajectories_seq) == len(object_label_trajectories_seq), 'Number of estimate and label trajectories do not match'
        assert len(samples_sequences) == len(object_estimate_trajectories_seq), 'Number of sequences and trajectories do not match'
        assert len(object_estimate_trajectories_seq) == len(self.robot_trajectories), 'Number of sequences and trajectories do not match'
        
        for i in range(len(self.robot_trajectories)):
            samples = samples_sequences[i]
            robot_trajectory = self.robot_trajectories[i]
            for tracking_id in object_estimate_trajectories_seq[i]:
                estimate_trajectory = object_estimate_trajectories_seq[i][tracking_id]
                assert type(estimate_trajectory) == Trajectory, 'Object estimate trajectory must be of type Trajectory'
                transform_trajectory_to_initial_pose(estimate_trajectory, robot_trajectory)
                estimate_trajectory.estimate_yaws()
                
                timesteps_before_interpolation = [ts for ts in estimate_trajectory.corresponding_timesteps]
                if self.interpolate_between_trajectory:
                    estimate_trajectory.interpolate_all_missing_poses()
                if self.smooth_estimate_trajectories:
                    estimate_trajectory.kalman_smooth(process_noise=self.ekf_noise, measurement_noise=self.ekf_measurement_noise)
                    # estimate_trajectory.estimate_yaws()
                    
                timesteps_after_interpolation_and_smoothing = estimate_trajectory.corresponding_timesteps
                timesteps_added = [ts for ts in timesteps_after_interpolation_and_smoothing if ts not in timesteps_before_interpolation]
                update_estimates_with_new_timesteps(timesteps_added, samples, loc_estimates_by_sample, tracking_id)
                
                # if self.smooth_estimate_trajectories:
                #     estimate_trajectory.kalman_smooth()
                #     # estimate_trajectory.estimate_yaws()

            for coda_tracking_id in object_label_trajectories_seq[i]:
                label_trajectory = object_label_trajectories_seq[i][coda_tracking_id]
                assert type(label_trajectory) == Trajectory, 'Object label trajectory must be of type Trajectory'
                transform_trajectory_to_initial_pose(label_trajectory, robot_trajectory)
                label_trajectory.estimate_yaws()
                
                timesteps_before_interpolation = [ts for ts in label_trajectory.corresponding_timesteps]
                if self.interpolate_between_trajectory:
                    label_trajectory.interpolate_all_missing_poses()
                    
                if self.smooth_label_trajectories:
                    label_trajectory.kalman_smooth(process_noise=self.ekf_noise, measurement_noise=self.ekf_measurement_noise)
                    # label_trajectory.estimate_yaws()

                timesteps_after_interpolation_and_smoothing = label_trajectory.corresponding_timesteps
                # timesteps that were added
                timesteps_added = [ts for ts in timesteps_after_interpolation_and_smoothing if ts not in timesteps_before_interpolation]
                update_labels_with_new_timesteps(timesteps_added, samples, loc_labels_by_sample, coda_tracking_id)
                
                # if self.smooth_label_trajectories:
                #     label_trajectory.kalman_smooth()
                #     label_trajectory.estimate_yaws()
        
        training_id_colors_each_seq = get_track_id_colors(samples_sequences, loc_estimates_by_sample, matched_ids)
            
        assert len(training_id_colors_each_seq) == len(samples_sequences), 'Number of sequences and colors do not match'
        assert len(self.robot_trajectories) == len(samples_sequences), 'Number of sequences and trajectories do not match'
        
        assert isinstance(object_estimate_trajectories_seq, list), 'Object estimate trajectories must be a list'
        assert isinstance(object_label_trajectories_seq, list), 'Object label trajectories must be a list'
        
        # compute metrics
        average_displacement_errors = 0.0
        final_displacement_errors = 0.0
        angular_displacement_errors = 0.0
        heading_deviation_errors = 0.0
        avg_angular_traj_change_estimates = 0.0
        avg_angular_traj_change_labels = 0.0
        n_trajs = 0
        
        for seq_idx in range(len(samples_sequences)):
            samples = samples_sequences[seq_idx]
            robot_trajectory = self.robot_trajectories[seq_idx]
            object_estimate_trajectories = object_estimate_trajectories_seq[seq_idx]
            object_label_trajectories = object_label_trajectories_seq[seq_idx]
            assert len(samples) == len(robot_trajectory), 'Number of samples and robot_trajectory do not match'
            
            for tracking_id in object_estimate_trajectories:
                coda_tracking_id = matched_ids[tracking_id]
                # if len == 1, skip
                if len(object_estimate_trajectories[tracking_id]) == 1:
                    continue
                average_displacement_errors += average_displacement_error(object_estimate_trajectories[tracking_id], object_label_trajectories[coda_tracking_id])
                final_displacement_errors += final_displacement_error(object_estimate_trajectories[tracking_id], object_label_trajectories[coda_tracking_id])
                angular_displacement_errors += angular_displacement_error(object_estimate_trajectories[tracking_id], object_label_trajectories[coda_tracking_id])
                heading_deviation_errors += heading_deviation_error(object_estimate_trajectories[tracking_id], object_label_trajectories[coda_tracking_id])
                avg_angular_traj_change_estimates += trajectory_abs_angle_diff(object_estimate_trajectories[tracking_id])
                avg_angular_traj_change_labels += trajectory_abs_angle_diff(object_label_trajectories[coda_tracking_id])
                n_trajs += 1
            
            if not self.save_visualiations:
                continue
            
            # visuals
            track_id_to_color = training_id_colors_each_seq[seq_idx]

            seq_bev_filepaths = []
            for i in range(len(samples)):
                sample_filepath = samples[i].rgb_image_filepath
                bev_out_filepath = f'{self.sample_save_folder}/individual_BEVs/{sample_filepath.split("/")[-1]}'
                if not os.path.exists(f'{self.sample_save_folder}/individual_BEVs'):
                    os.makedirs(f'{self.sample_save_folder}/individual_BEVs')
                current_sample_filepath = sample_filepath
                current_robot_pose = robot_trajectory.bev_poses[i]
                save_bev_visualization(current_robot_pose, 
                                            object_estimate_trajectories, 
                                            object_label_trajectories,
                                            loc_estimates_by_sample, loc_labels_by_sample, # these are just needed for bbox visuals
                                            current_sample_filepath, 
                                            matched_ids, fp=bev_out_filepath, tracking_id_colors=track_id_to_color,
                                            grid_x_min=self.grid_x_min, grid_x_max=self.grid_x_max, grid_y_min=self.grid_y_min, grid_y_max=self.grid_y_max,
                                            estimator_name=self.bbox_and_bev_estimation.__class__.__name__)
                seq_bev_filepaths.append(bev_out_filepath)
            # create gif from all the bevs
            save_img_bev_gif(seq_bev_filepaths, self.sample_save_folder)
    
        average_displacement_errors /= n_trajs
        final_displacement_errors /= n_trajs
        angular_displacement_errors /= n_trajs
        heading_deviation_errors /= n_trajs
        avg_angular_traj_change_estimates /= n_trajs
        avg_angular_traj_change_labels /= n_trajs
        print(f'Average Displacement Errors: {average_displacement_errors}')
        print(f'Angular Displacement Errors: {angular_displacement_errors}')
        print(f'Average Angular Trajectory Change Estimates: {avg_angular_traj_change_estimates}')
        # let's save metrics to a file and the relevant params used
        with open(f'{self.sample_save_folder}/metrics.txt', 'w') as file:
            file.write(f'Average Average Displacement Errors: {average_displacement_errors}\n')
            file.write(f'Average Final Displacement Errors: {final_displacement_errors}\n')
            file.write(f'Average Angular Displacement Errors: {angular_displacement_errors}\n')
            file.write(f'Average Heading Deviation Errors: {heading_deviation_errors}\n')
            file.write(f'Average Angular Trajectory Change Estimates: {avg_angular_traj_change_estimates}\n')
            file.write(f'Average Angular Trajectory Change Labels: {avg_angular_traj_change_labels}\n')
            file.write(f'Interpolate between trajectories: {self.interpolate_between_trajectory}\n')
            file.write(f'Smooth estimate trajectories: {self.smooth_estimate_trajectories}\n')
            file.write(f'Smooth label trajectories: {self.smooth_label_trajectories}\n')
            file.write(f'Smooth robot trajectory: {self.smooth_robot_trajectory}\n')

        # return dict of metrics and params
        return {
            'average_displacement_errors': average_displacement_errors,
            'final_displacement_errors': final_displacement_errors,
            'angular_displacement_errors': angular_displacement_errors,
            'heading_deviation_errors': heading_deviation_errors,
            'avg_angular_traj_change_estimates': avg_angular_traj_change_estimates,
            'avg_angular_traj_change_labels': avg_angular_traj_change_labels,
            
            'interpolate_between_trajectory': self.interpolate_between_trajectory,
            'smooth_estimate_trajectories': self.smooth_estimate_trajectories,
            'smooth_label_trajectories': self.smooth_label_trajectories,
            'smooth_robot_trajectory': self.smooth_robot_trajectory,
            
            'ekf_x_noise': self.ekf_x_noise,
            'ekf_y_noise': self.ekf_y_noise,
            'ekf_yaw_noise': self.ekf_yaw_noise,
            'ekf_speed_noise': self.ekf_speed_noise,
            'ekf_x_measurement_noise': self.ekf_x_measurement_noise,
            'ekf_y_measurement_noise': self.ekf_y_measurement_noise
        }
        
def plot_param_vs_metric(results, param_name, metric_name, save_folder, num_iterations=1000, degree=3):
    """Plots EKF parameter vs metric with polynomial regression and saves the plot."""
    param_values = [result['ekf_params'][param_name] for result in results]
    metric_values = [result['metrics'][metric_name] for result in results]

    # Sort param_values and corresponding metric_values for a smooth curve
    sorted_pairs = sorted(zip(param_values, metric_values))
    param_values, metric_values = zip(*sorted_pairs)

    # Scatter plot of the actual points
    plt.figure(figsize=(8, 6))
    plt.scatter(param_values, metric_values, label='Data points')

    # Polynomial fit of the data
    coefficients = np.polyfit(param_values, metric_values, degree)
    poly = np.poly1d(coefficients)
    poly_y = poly(param_values)

    # Plot the polynomial regression line
    plt.plot(param_values, poly_y, color='red', label=f'Polynomial fit (degree={degree})', linewidth=3)
    
    plt.title(f'{param_name} vs {metric_name} n_iter={num_iterations}')
    plt.xlabel(f'{param_name}')
    plt.ylabel(f'{metric_name}')
    plt.grid()
    plt.legend()

    # Save the plot
    plot_filename = os.path.join(save_folder, f'{param_name}_vs_{metric_name}.png')
    plt.savefig(plot_filename)
    plt.close()

def save_top_5_results(results, metric_name, metrics_of_interest, save_folder):
    """Save the top 5 results for a given metric into a file."""
    top_5_results_filepath = os.path.join(save_folder, f'top_5_{metric_name}.txt')
    
    with open(top_5_results_filepath, 'w') as file:
        for idx, result in enumerate(results[:5]):
            metrics = result['metrics']
            ekf_params = result['ekf_params']
            file.write(f"\nRank {idx+1}:\n")
            for metric_name in metrics_of_interest:
                file.write(f"{metric_name}: {metrics[metric_name]}\n")
            file.write(f"EKF Parameters:\n")
            file.write(f"  EKF X Noise: {ekf_params['ekf_x_noise']}\n")
            file.write(f"  EKF Y Noise: {ekf_params['ekf_y_noise']}\n")
            file.write(f"  EKF Yaw Noise: {ekf_params['ekf_yaw_noise']}\n")
            file.write(f"  EKF Speed Noise: {ekf_params['ekf_speed_noise']}\n")
            file.write(f"  EKF X Measurement Noise: {ekf_params['ekf_x_measurement_noise']}\n")
            file.write(f"  EKF Y Measurement Noise: {ekf_params['ekf_y_measurement_noise']}\n")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Evaluate object localization.')
    args.add_argument('--config', type=str, help='Path to configuration file.')
    args.add_argument('--debug', action='store_true', help='Enable debug mode.')
    args.add_argument('--ekf_tune', action='store_true', help='Enable EKF tuning mode.')
    config_filepath = args.parse_args().config
    debug = args.parse_args().debug

    evaluator = ObjectLocalizationEvaluator(config_filepath, debug)
    
    if not args.parse_args().ekf_tune:
        evaluator.evaluate_sequences()
        sys.exit(0)

    # Set the number of iterations you want to run
    num_iterations = 10000  # For example, run 1000 random search iterations
    results = []

    # Weights for the weighted metric based on your preferences
    avg_disp_weight = 1.05  # Most important, prefer values below 0.68
    avg_ang_traj_weight = 1.2  # Values close to 0.10-0.14 are preferred
    ang_disp_weight = 0.2  # Least important

    for i in range(num_iterations):
        print(f'Running iteration {i+1}/{num_iterations}')
        evaluator.turn_off_visualizations()
        # Generate random EKF parameters for this iteration
        evaluator.set_ekf_params(randomize=True)

        # Evaluate and get the metrics dictionary
        metrics = evaluator.evaluate_sequences()

        # Calculate the weighted metric
        weighted_metric = (avg_disp_weight * metrics['average_displacement_errors'] +
                           avg_ang_traj_weight * metrics['avg_angular_traj_change_estimates'] +
                           ang_disp_weight * metrics['angular_displacement_errors'])
        
        metrics['weighted_metric'] = weighted_metric

        # Store the metrics and the corresponding EKF parameters
        results.append({
            'metrics': metrics,
            'ekf_params': {
                'ekf_x_noise': evaluator.ekf_x_noise,
                'ekf_y_noise': evaluator.ekf_y_noise,
                'ekf_yaw_noise': evaluator.ekf_yaw_noise,
                'ekf_speed_noise': evaluator.ekf_speed_noise,
                'ekf_x_measurement_noise': evaluator.ekf_x_measurement_noise,
                'ekf_y_measurement_noise': evaluator.ekf_y_measurement_noise
            }
        })

    # Create the directory for plots and results if it doesn't exist
    save_folder = os.path.join(evaluator.sample_save_folder, 'ekf_tuning_results')
    os.makedirs(save_folder, exist_ok=True)

    # Sort results and save the top 5 results for each metric
    metrics_of_interest = ['average_displacement_errors', 'angular_displacement_errors', 'avg_angular_traj_change_estimates', 'weighted_metric']

    for metric in metrics_of_interest:
        # Sort the results by the current metric
        sorted_results = sorted(results, key=lambda x: x['metrics'][metric])
        
        # Save the top 5 results for the current metric
        save_top_5_results(sorted_results, metric, metrics_of_interest, save_folder)

    # Plot parameter vs metric for all results
    plot_save_folder = os.path.join(save_folder, 'plots')
    os.makedirs(plot_save_folder, exist_ok=True)

    ekf_params = ['ekf_x_noise', 'ekf_y_noise', 'ekf_yaw_noise', 'ekf_speed_noise', 'ekf_x_measurement_noise', 'ekf_y_measurement_noise']

    # Generate and save the plots for all metrics
    for param in ekf_params:
        for metric in metrics_of_interest:
            plot_param_vs_metric(results, param, metric, plot_save_folder, degree=3)

    print(f"Saved all parameter vs. metric plots and top 5 results in {save_folder}")