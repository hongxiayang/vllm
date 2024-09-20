#!/usr/bin/bash

#model_path="/data/llama2-70b-chat"
model_path="/dockerx/data/llama-2-7b-chat-hf"
dataset_path="/dockerx/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

torchrun --standalone --nnodes=1 --nproc-per-node=2 /app/vllm/benchmarks/benchmark_throughput.py --dataset "$dataset_path" --model "$model_path" "${positional_args[@]}"
  return $?
