# Use the NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    python3-pip \
    software-properties-common \
    tmux \
    vim \
    unzip


# Install PCL library
RUN apt-get update && apt-get install -y libpcl-dev

# Install additional tools
RUN apt-get update && apt-get install -y \
    tmux \
    vim \
    python-is-python3 \
    ffmpeg \
    wget \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV with CUDA support
RUN apt-get update && apt-get install -y \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libtbb-dev \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*


# Install TorchScript (PyTorch C++ API) and other Python dependencies
RUN pip3 install numpy torch torchvision torchaudio tqdm opencv-python matplotlib Pillow mmengine mmcv imagecorruptions
RUN pip3 install tensorboardX imgaug iopath timm plyfile
RUN pip3 install numpy==1.23.5 opencv-python==4.7.0.72

# Download and install libtorch for CUDA 12.1
RUN curl -LO https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip -d /usr/local && \
    rm libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip

RUN apt-get update && apt-get install -y \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xfixes0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrender1 \
    libxcb1 \
    libfontconfig1 \
    libxext6 \
    x11-apps

# Set up environment variables for GUI
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_INDIRECT=1

# Create the workspace directory inside the container
RUN mkdir -p /workspace/src

RUN pip install open3d

# Set the working directory
WORKDIR /workspace

# Copy over download depth model file
COPY tuning_cfg.yaml tuning_cfg.yaml
COPY download_depth_model.py download_depth_model.py

# run python script to download depth model so we don't have to do it everytime we run the container 
RUN python3 download_depth_model.py --config tuning_cfg.yaml

# Set the default command to source ROS 2 and start a bash shell and run demo.py python script
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && exec bash"]