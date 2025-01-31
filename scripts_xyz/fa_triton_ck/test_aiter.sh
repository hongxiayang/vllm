#!/bin/bash

#MODEL="/mnt/md0/pretrained_model/Mixtral-8x7B-Instruct-v0.1"

#MODEL="/data/Meta-Llama-3.1-70B-Instruct-FP8-KV"

MODEL="/data/Mixtral-8x7B-Instruct-v0.1"

#HIP_VISIBLE_DEVICES=4,5,6,7 
#VLLM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL -tp 8 --block-size 16  --quantization fp8 --kv-cache-dtype fp8 --batch-size 128 --input-len 100 --output-len 1000 --num-iters-warmup 5 --num-iters 10

VLLM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0 python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL -tp 8 --block-size 16 --kv-cache-dtype fp8 --batch-size 128 --input-len 100 --output-len 1000 --num-iters-warmup 5 --num-iters 10
