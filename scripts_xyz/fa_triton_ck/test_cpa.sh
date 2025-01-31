#!/bin/bash

set +ex

#MODEL="/data/Mixtral-8x7B-Instruct-v0.1"

MODEL="/data/model/Llama-3.1-70B-Instruct-FP8-KV"

#MODEL="/data/model/Mixtral-8x22B-Instruct-v0.1-FP8-KV"

#MODEL="/data/Meta-Llama-3-70B-Instruct"

#VLLM_USE_TRITON_FLASH_ATTN=0 python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL -tp 1 --block-size 16 --kv-cache-dtype fp8 --batch-size 128 --input-len 100 --output-len 1000 --num-iters-warmup 5 --num-iters 10

#VLLM_USE_AITER=1 


# VLLM_USE_TRITON_FLASH_ATTN=0 python /app/vllm/benchmarks/benchmark_throughput.py --model $MODEL --kv-cache-dtype fp8 \
# --input-len 100 --output-len 100 --quantization fp8  --distributed-executor-backend mp --tensor-parallel-size 1



# VLLM_USE_TRITON_FLASH_ATTN=0 python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL --kv-cache-dtype fp8 --batch-size 256  \
# --input-len 2048 --output-len 2048 --quantization fp8  --distributed-executor-backend mp --tensor-parallel-size 8 \
# --block_size 16 --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10
 
 # --enable-chunked-prefill=False


#VLLM_USE_TRITON_FLASH_ATTN=0 

#core-dumped

# python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL --dtype float16 --kv-cache-dtype fp8 \
#   --gpu-memory-utilization 0.99 --batch-size 256 --input-len 2048 --output-len 2048 --quantization fp8  \
#   --distributed-executor-backend mp --tensor-parallel-size 8 --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
#   --enable-chunked-prefill=False

# 1 got OOM error.
# python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL --dtype float16 --kv-cache-dtype fp8 \
#   --gpu-memory-utilization 0.99 --batch-size 256 --input-len 2048 --output-len 2048 --quantization fp8  \
#   --distributed-executor-backend mp --tensor-parallel-size 1 --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
#   --enable-chunked-prefill=False

# remove 0.99, tp8 still core dump, tp1 is ok, very slow
# python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL --dtype float16 --kv-cache-dtype fp8 \
#   --batch-size 256 --input-len 2048 --output-len 2048 --quantization fp8  \
#   --distributed-executor-backend mp --tensor-parallel-size 8 --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
#   --enable-chunked-prefill=False


# still core dump
# python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL --dtype float16 --kv-cache-dtype fp8 \
#   --batch-size 128 --input-len 128 --output-len 128 --quantization fp8  \
#   --distributed-executor-backend mp --tensor-parallel-size 8 --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
#   --enable-chunked-prefill=False

#(should I cast dtype?)
#VLLM_USE_TRITON_FLASH_ATTN=0  



# python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL --kv-cache-dtype fp8 \
#   --batch-size 128 --input-len 128 --output-len 128 --quantization fp8  \
#   --distributed-executor-backend mp --tensor-parallel-size 8 --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
#   --enable-chunked-prefill=False


if [[ $# -ge 1 ]]; then
   MODEL="$1"
fi

TP=8

for batch_size in 1 2 8 128 256;
do
    for gen_len in 128 2048;
    do
            for input_len in 128 2048;
            do
            # please do NOT change the format of hte following echo. DLM relies on this infor.
            echo "===================== RUNNING $MODEL $input_len $gen_len $batch_size tp=$TP ==================================================="
            python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL \
            --kv-cache-dtype fp8 \
            --batch-size $batch_size --input-len $input_len --output-len $gen_len --quantization fp8  \
            --distributed-executor-backend mp --tensor-parallel-size $TP --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
            --enable-chunked-prefill=False
            done
    done
done