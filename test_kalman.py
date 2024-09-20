import numpy as np
from utils import BEVPose
import matplotlib.pyplot as plt
from typing import List

label_bev = np.load('eval_kalman/seq_id_2_track_id_95_label_bev.npy')
label_timesteps = np.load('eval_kalman/seq_id_2_track_id_95_label_timesteps.npy')
bev_estimate = np.load('eval_kalman/seq_id_2_track_id_95_bev_estimate.npy')
estimate_timesteps = np.load('eval_kalman/seq_id_2_track_id_95_estimate_timesteps.npy')
robot_poses = np.load('eval_kalman/seq_id_2_robot_poses.npy')
robot_timesteps = np.load('eval_kalman/seq_id_2_robot_timesteps.npy')

def kalman_smooth(bev_poses, speed_guess: float = 0.325, process_noise: List[float] = [1.0], measurement_noise: List[float] = [2.0]):
    # Convert to a bev_poses list
    bev_poses_tmp = []
    for pose in bev_poses:
        x, y = pose
        bev_poses_tmp.append(BEVPose(x, y, None))
    bev_poses = bev_poses_tmp

    def estimate_yaws(bev_poses):
        if len(bev_poses) == 1:
            bev_poses[0].yaw = None
            return bev_poses

        # use the difference between consecutive poses to estimate the yaw
        for i in range(len(bev_poses) - 1):
            next_position = bev_poses[i + 1].get_position_np()
            current_position = bev_poses[i].get_position_np()
            displacement = next_position - current_position
            bev_poses[i].yaw = np.arctan2(displacement[1], displacement[0])
        bev_poses[-1].yaw = bev_poses[-2].yaw
        return bev_poses

    # get initial estimation of yaw by interpolation
    bev_poses = estimate_yaws(bev_poses)

    # Initialize state vector: [x, y, vx, vy]
    n = len(bev_poses)
    smoothed_poses = []

    # State [x, y, vx, vy] (positions and velocities)
    state = np.array([bev_poses[0].x, bev_poses[0].y, bev_poses[0].yaw, speed_guess])
    
    # State covariance matrix
    state_cov = np.eye(4)

    # Process noise
    if len(process_noise) == 1:
        Q = process_noise[0] * np.eye(4)
    elif len(process_noise) == 4:
        Q = np.diag(process_noise)
    else:
        raise ValueError("process_noise must be a list of either 1 or 4 elements")

    # Measurement matrix (we measure x and y positions only)
    H = np.array([[1, 0, 0, 0],  # we measure x
                    [0, 1, 0, 0]]) # we measure y
    
    # Measurement noise covariance matrix
    if len(measurement_noise) == 1:
        R = measurement_noise[0] * np.eye(2)
    elif len(measurement_noise) == 2:
        R = np.diag(measurement_noise)
    else:
        raise ValueError("process_noise must be a list of either 1 or 2 elements")

    # Identity matrix for updating
    I = np.eye(4)

    # Time step (assuming constant time step between poses)
    dt = 0.25  # You may need to adjust this based on your data

    # Forward pass (Kalman filter)
    forward_states = []
    forward_covs = []
    for i in range(n):
        # Get the current measurement (position)
        z = np.array([bev_poses[i].x, bev_poses[i].y])

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

        forward_states.append(state)
        forward_covs.append(state_cov)

    # Backward pass (RTS smoother)
    smoothed_states = [forward_states[-1]]
    smoothed_covs = [forward_covs[-1]]
    
    for i in range(n - 2, -1, -1):
        x, y, yaw, v = forward_states[i]
        F = np.array([
            [1, 0, -v * np.sin(yaw) * dt, np.cos(yaw) * dt],
            [0, 1,  v * np.cos(yaw) * dt, np.sin(yaw) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        P_pred = F @ forward_covs[i] @ F.T + Q
        C = forward_covs[i] @ F.T @ np.linalg.inv(P_pred)
        
        smoothed_state = forward_states[i] + C @ (smoothed_states[0] - F @ forward_states[i])
        smoothed_cov = forward_covs[i] + C @ (smoothed_covs[0] - P_pred) @ C.T
        
        smoothed_states.insert(0, smoothed_state)
        smoothed_covs.insert(0, smoothed_cov)

    # Convert smoothed states to BEVPose objects
    smoothed_poses = [BEVPose(state[0], state[1], state[2]) for state in smoothed_states]

    return smoothed_poses, bev_poses

smoothed_label_poses, label_poses = kalman_smooth(label_bev, process_noise=[0.2, 0.2, 1, 1], measurement_noise=[0.2])
smoothed_estimate_poses, estimate_poses = kalman_smooth(bev_estimate, process_noise=[0.5, 0.5, 1, 1], measurement_noise=[1.5])

def plot_comparison(poses, smoothed_poses, name):
    # Extract x, y, and yaw from original and smoothed poses
    original_x = [pose.x for pose in poses]
    original_y = [pose.y for pose in poses]
    original_yaw = [pose.yaw for pose in poses]

    smoothed_x = [pose.x for pose in smoothed_poses]
    smoothed_y = [pose.y for pose in smoothed_poses]
    smoothed_yaw = [pose.yaw for pose in smoothed_poses]

    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

    # Plot x
    ax1.plot(original_x, label='Original')
    ax1.plot(smoothed_x, label='Smoothed')
    ax1.set_title('X Position')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('X')
    ax1.legend()

    # Plot y
    ax2.plot(original_y, label='Original')
    ax2.plot(smoothed_y, label='Smoothed')
    ax2.set_title('Y Position')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Y')
    ax2.legend()

    # Plot yaw
    ax3.plot(original_yaw, label='Original')
    ax3.plot(smoothed_yaw, label='Smoothed')
    ax3.set_title('Yaw')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Yaw (radians)')
    ax3.legend()

    # Plot pose comparison
    ax4.scatter(original_x, original_y, c='blue', label='Original', alpha=0.5)
    ax4.scatter(smoothed_x, smoothed_y, c='red', label='Smoothed', alpha=0.5)

    # Add arrows to show heading
    arrow_step = max(1, len(original_x) // 20)  # Show fewer arrows for clarity
    ax4.quiver(original_x[::arrow_step], original_y[::arrow_step], 
            np.cos(original_yaw[::arrow_step]), np.sin(original_yaw[::arrow_step]), 
            color='blue', scale=30, width=0.002, alpha=0.5)
    ax4.quiver(smoothed_x[::arrow_step], smoothed_y[::arrow_step], 
            np.cos(smoothed_yaw[::arrow_step]), np.sin(smoothed_yaw[::arrow_step]), 
            color='red', scale=30, width=0.002, alpha=0.5)

    ax4.set_title('Pose Comparison')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.legend()
    ax4.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig('kalman_smoothing_results_' + name + '.png')
    plt.close()

plot_comparison(label_poses, smoothed_label_poses, 'label')
plot_comparison(estimate_poses, smoothed_estimate_poses, 'estimate')    
plot_comparison(label_poses, estimate_poses, 'compare_to_gt_no_kf')
plot_comparison(label_poses, smoothed_estimate_poses, 'compare_to_gt_kf')

