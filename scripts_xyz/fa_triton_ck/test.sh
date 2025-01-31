#!/bin/bash

#DOCKER=rocm/vllm-private:ali1209_yuzho_moe_final_f0468ab
DOCKER=rocmshared/pytorch:vllm_aiter_20250124

#DOCKER=moe_vllm_test

#NAME=hongxia-moe-test
NAME=hongxia-aiter
docker run -it --network=host --device=/dev/kfd --device=/dev/dri \
        --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -v /home:/home -v /home/hongxyan/dockerx:/dockerx -v /data:/data -v /mnt:/mnt \
        --name $NAME $DOCKER

# --shm-size=16G --ulimit memlock=-1 --ulimit stack=67108864 \
