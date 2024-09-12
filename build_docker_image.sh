#!/bin/bash

# Define the image name
IMAGE_NAME="tune_3d_estimation_docker_img"

# Build the Docker image if it doesn't exist
echo "Building the Docker image..."
docker build -t $IMAGE_NAME -f Dockerfile .