
# Tuning 3D Pedestrian Tracking Estimates

Monocular 3D estimate tuning and evaluation for social navigation scenes.  
This repo provides a small pipeline to (1) load scene frames and 2D/3D annotations, (2) estimate per‑agent positions using a monocular metric depth model, (3) project to a Bird’s‑Eye View (BEV) for tracking, (4) smooth with a simple EKF, and (5) evaluate with standard trajectory and bbox metrics. Utilities are included to visualize LART 2D boxes and to grid‑search/tune EKF hyperparameters.

> **Status:** the code in this archive is a working scaffold; some files include redacted sections (`...`). The Dockerfile and config show the intended environment. The commands below reflect the interface present in the code and should get you running once you point to your data and fill any missing pieces.

---

## Contents

```
tune_3d_estimates-main/
├── Dockerfile
├── build_docker_image.sh
├── run_evaluation.sh
├── tune.sh
├── tuning_cfg.yaml
├── evaluate_object_localization.py
├── bbox_and_bev_estimation.py
├── bbox_matching.py
├── data_loader.py
├── geometry.py
├── metric_depth.py
├── metrics.py
├── save_lart_bounding_box_visuals.py
├── structures.py
├── visualize.py
└── constants.py
```

### Key scripts

- `evaluate_object_localization.py`: main entry point.  
  Args (from code):  
  - `--config PATH` (required): YAML config.  
  - `--ekf_tune` (optional): grid‑search EKF noise params defined in `tuning_cfg.yaml`.  
  - `--debug` (optional): extra checks/plots.
- `save_lart_bounding_box_visuals.py`: renders one image per LART bbox for QA.
- `build_docker_image.sh` / `run_evaluation.sh` / `tune.sh`: convenience wrappers for Docker build/run/tuning.

---

## Data prerequisites

The pipeline expects **images**, **2D bboxes** (LART), and **(optionally) CODa‑style metadata**.

- **Environment variable**  
  Set the dataset root so loaders can resolve files:
  ```bash
  export CODA_ROOT_DIR=/path/to/your/coda_dataset_root
  ```

- **Images & labels**  
  - Images: accessible via deterministic paths (frame index → image file).  
  - 2D bboxes (LART): a per‑sample pickle/JSON mapping `tracking_id → bbox2d`.
  - If you have a CODa devkit, keep it under `coda-devkit/` or update paths in `tuning_cfg.yaml`.  
  - For manual id correspondences during evaluation, `tuning_cfg.yaml` references:
    ```yaml
    bbox_matching_config:
      matching_filepath: coda-devkit/corresponding_trackings.json
    ```

> If your structure differs, adapt the path logic in `data_loader.py` and the functions that map `frame_idx → (image, json)`.

---

## Configuration (`tuning_cfg.yaml`)

Important fields (present in the provided config; fill your values as needed):

```yaml
# Image→BEV bounds (meters)
x_min: 0.0; x_max: 20.0
y_min: -10.0; y_max: 10.0
z_min: 0.0; z_max: 2.0

# Camera
intrinsics: [fx, fy, cx, cy]          # fill with your calibrated intrinsics
distortion_coeffs: [k1, k2, p1, p2, k3]   # optional; used if undistortion enabled
rotation_matrix: [...]                # 3x3, camera→robot (if available)
translation_vector: [tx, ty, tz]      # meters

# Preprocessing flags
use_undistort_correction: true|false
use_gaussian_blur: true|false
use_gamma_correction: true|false

# Metric depth model (downloaded via torch.hub)
metric_3d_model: "metric3d_vit_small" # example; see yvanyin/metric3d hub names

# EKF (and tuning ranges)
ekf_x_noise: 1.5
ekf_y_noise: 2.5
ekf_yaw_noise: 0.3
ekf_speed_noise: 0.02
ekf_x_measurement_noise: 5.0
ekf_y_measurement_noise: 3.0

# Grid search ranges used when --ekf_tune is set
ekf_x_noise_range: [0.5, 4.0]
ekf_y_noise_range: [2.0, 3.0]
ekf_yaw_noise_range: [0.3, 0.7]
ekf_speed_noise_range: [0.01, 0.05]
ekf_x_measurement_noise_range: [4.0, 6.0]
ekf_y_measurement_noise_range: [2.0, 5.0]

# Outputs
save_folder: "estimations_save"
sample_save_folder: "estimations_sample_save"
smooth_estimate_trajectories: true
smooth_label_trajectories: false
smooth_robot_trajectory: false

random_seed: 42
debug_mode: false
```

---

## Recommended setup (Docker, GPU)

The Docker image is configured for CUDA 12.1 and will pre‑download the metric depth model.

### Requirements

- Linux (tested with Ubuntu 22.04 inside the image)
- **NVIDIA GPU** + recent driver (compatible with CUDA 12.1)
- Docker 24+
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
- Internet access (first build/model download)

### Build

```bash
cd tune_3d_estimates-main
./build_docker_image.sh
```

This runs:
```bash
docker build -t tune_3d_estimation_docker_img -f Dockerfile .
```
During the build, `download_depth_model.py --config tuning_cfg.yaml` executes to cache the depth model (see `metric_3d_model`).

### Run (evaluation)

Set your config path (relative to repo root or absolute):
```bash
export CONFIG_FILEPATH=tuning_cfg.yaml
./run_evaluation.sh
```

This will (1) stop any existing container with the same name, (2) run a new container with the repo mounted at `/workspace`, and (3) execute:
```bash
python evaluate_object_localization.py --config $CONFIG_FILEPATH
```

### Run (tuning)

```bash
export CONFIG_FILEPATH=tuning_cfg.yaml
./tune.sh   # runs with: --ekf_tune
```

> The run scripts also map X11 for optional on‑screen plots. If you are headless, that’s fine—the code primarily saves images/GIFs via OpenCV/Matplotlib.

---

## Local (no Docker) setup

If you prefer a Python venv/conda install (GPU strongly recommended):

1. **Python**: 3.9–3.11  
2. **Install deps** (minimal set inferred from imports):

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA/CPU build
pip install numpy opencv-python pillow matplotlib pyyaml tqdm open3d
```

3. **(First‑time)**: download the depth model
```bash
python download_depth_model.py --config tuning_cfg.yaml
```

4. **Run**
```bash
python evaluate_object_localization.py --config tuning_cfg.yaml
# or
python evaluate_object_localization.py --config tuning_cfg.yaml --ekf_tune --debug
```

---

## Outputs

- **Per‑sample visuals** under `estimations_sample_save/` (configurable)  
  - BEV GIFs of estimated vs label trajectories (`visualize.py` / `save_img_bev_gif`)
  - Optional 2D bbox overlays (see `save_lart_bounding_box_visuals.py`)
- **Metrics** (from `metrics.py`):  
  - 2D bbox error (1 − IoU) for matched ids (`compute_bbox_errors`)  
  - Trajectory metrics such as ADE/FDE, heading deviation, absolute angle differences, etc.

> Filenames and exact CSV outputs may depend on the sections elided in this archive; search for `save_*` functions in `visualize.py` and where metrics are aggregated in `evaluate_object_localization.py`.

---

## Troubleshooting

- **Model not found**: ensure `metric_3d_model` in `tuning_cfg.yaml` is a valid [Torch Hub model in `yvanyin/metric3d`].
- **CUDA/driver mismatch**: if Docker fails to see your GPU, verify `nvidia-smi` on the host and that the NVIDIA Container Toolkit is installed.
- **Headless servers**: if X11 errors appear, you can remove the X11 volume/env lines in the `run_evaluation.sh`/`tune.sh` scripts; the pipeline saves files to disk.
- **Paths not found**: verify `CODA_ROOT_DIR` and any path templates in `tuning_cfg.yaml`; adjust `data_loader.py` path logic if your layout differs.
- **Open3D import**: If Open3D wheels don’t match your Python, try `pip install open3d==0.17.*` or similar.

---

## Citation & Acknowledgements

- Metric depth via Torch Hub: `yvanyin/metric3d` (pretrained Metric3D).  
- CODa devkit format & LART bbox utilities as referenced in code.  
- This code was prepared for internal experiments on social navigation scene understanding.

If you use or extend this, please add proper citations for your datasets and pre‑trained models.
