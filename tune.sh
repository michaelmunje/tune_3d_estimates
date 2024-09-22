#!/bin/bash

# Define the image name and container name
IMAGE_NAME="tune_3d_estimation_docker_img"

# Define the path where the current directory will be mounted inside the container
HOST_WORKSPACE_PATH=$(pwd)
CONTAINER_WORKSPACE_PATH="/workspace"

CONTAINER_NAME="tune_3d_estimation_docker_container"
# Check if the container is already running
if [[ "$(docker ps -q -f name=$CONTAINER_NAME)" ]]; then
    echo "Stopping the already running container..."
    docker stop $CONTAINER_NAME
fi

# Check if a container with the same name exists (but is stopped or exited)
if [[ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]]; then
    echo "Removing the existing container..."
    docker rm $CONTAINER_NAME
fi

CONFIG_FILEPATH="$CONTAINER_WORKSPACE_PATH/tuning_cfg.yaml"

export DISPLAY=$(echo $DISPLAY)
xhost +local:
export LIBGL_ALWAYS_INDIRECT=0

# first check if isolated_network exists
if [[ "$(docker network ls -q -f name=isolated_network)" ]]; then
    echo "isolated_network already exists"
else
    echo "isolated_network does not exist, creating isolated_network..."
    docker network create isolated_network
fi

# Run the Docker container with the current directory and video file mounted inside
echo "Running the Docker container..."

docker run -it --rm \
    --gpus all \
    --name $CONTAINER_NAME \
    --net=isolated_network \
    --ipc=host \
    --pid=host \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e LIBGL_ALWAYS_INDIRECT=0 \
    -e DISPLAY=$DISPLAY \
    -v="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -u root \
    -v "$HOST_WORKSPACE_PATH:$CONTAINER_WORKSPACE_PATH" \
    $IMAGE_NAME /bin/bash -c "python evaluate_object_localization.py --config $CONFIG_FILEPATH --ekf_tune"
    # --env="QT_X11_NO_MITSHM=1" \
    # --env="LIBGL_ALWAYS_INDIRECT=1" \