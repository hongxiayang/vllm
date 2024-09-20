#!/usr/bin/bash


# The first optional parameter is the base image, default is rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
# https://compute-artifactory.amd.com/artifactory/rocm-docker-experimental/rocm-plus-docker/framework/release-public/rocm6.0_ubuntu20.04_py3.9_pytorch_release-2.1/
#compute-artifactory.amd.com:5000/rocm-plus-docker/framework/release-public:rocm6.0_ubuntu20.04_py3.9_pytorch_release-2.1
#rocm6.0_ubuntu22.04_py3.10_pytorch_release-2.1
# the second optional parameter is the FA GFX ARCH List to build flashattention, default is gfx90a;gfx942
# the third optional parameter is flashattention branch name, the default is 3d2b6f5
set -ex

if [[ $# -ge 1 ]]; then
   BASE_IMAGE="$1"
else
   #BASE_IMAGE="compute-artifactory.amd.com:5000/rocm-plus-docker/framework/release-public:rocm6.0_ubuntu22.04_py3.10_pytorch_release-2.1"
   #BASE_IMAGE="compute-artifactory.amd.com:5000/rocm-plus-docker/framework/release-public:rocm6.0_ubuntu20.04_py3.9_pytorch_rocm6.0_internal_testing"
   BASE_IMAGE="rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1"
   #BASE_IMAGE="rocm/pytorch-nightly:latest"
   #BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1"
fi

echo "BASE Image is $BASE_IMAGE"

if [[ $# -ge 2 ]]; then
   FA_GFX_ARCH_LIST="$2"
else
   FA_GFX_ARCH_LIST="gfx90a;gfx942"
fi

echo "GFX_ARCH_LIST: $FA_GFX_ARCH_LIST"

if [[ $# -ge 3 ]]; then
   FA_BRANCH_NAME="$3"
else
   FA_BRANCH_NAME="3d2b6f5"
fi

echo "FA_BRANCH_NAME: $FA_BRANCH_NAME"

# DockerImageName="vllm-${BASE_IMAGE}"

DockerImageName="vllm-sdp-test"

echo "Docker Image Name: $DockerImageName"

# docker build --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg FA_GFX_ARCHS="$FA_GFX_ARCH_LIST" --build-arg FA_BRANCH="$FA_BRANCH_NAME"  -f Dockerfile.rocm -t "$DockerImageName" . 

# sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host \
#       --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G  -v ~/dockerx/data/llama-2-7b-chat-hf:/app/model "$DockerImageName"

# -v /data:/data for mi300 to use different models

# test ref attention

docker build --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg BUILD_FA="0"  -f Dockerfile.rocm -t "$DockerImageName" . 

sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 8G  -v ~/dockerx:/dockerx -v ~/dockerx/data/llama-2-7b-chat-hf:/app/model "$DockerImageName"

# -v /data:/data for mi300 to use different models
