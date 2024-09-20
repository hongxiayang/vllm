#!/usr/bin/bash
set -ex

docker commit $1 rocm/pytorch-private:hongxia-vllm
docker push rocm/pytorch-private:hongxia-vllm

