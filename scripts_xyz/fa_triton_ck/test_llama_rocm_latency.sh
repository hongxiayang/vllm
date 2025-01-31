#!/bin/bash

# set -e will stop when error happens

set +ex

MODEL="/data/Meta-Llama-3-70B-Instruct"


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
            echo "===================== RUNNING $MODEL $input_len $gen_len $batch_size tp=$TP ==================================================="
            python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL \
            --kv-cache-dtype fp8 \
            --batch-size $batch_size --input-len $input_len --output-len $gen_len \
            --distributed-executor-backend mp --tensor-parallel-size $TP --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
            --enable-chunked-prefill=False
            done
    done
done

# below is to rerun the failure cases
# TP=8

# for batch_size in 128 256;
# do
#     for gen_len in 128 2048;
#     do
#             for input_len in 128 2048;
#             do
#             echo "===================== RUNNING $MODEL $input_len $gen_len $batch_size tp=$TP ==================================================="
#             python /app/vllm/benchmarks/benchmark_latency.py --model $MODEL \
#             --kv-cache-dtype fp8 \
#             --batch-size $batch_size --input-len $input_len --output-len $gen_len \
#             --distributed-executor-backend mp --tensor-parallel-size $TP --num-iters-warmup 2 --num-iters 5 --num-scheduler-steps 10 \
#             --enable-chunked-prefill=False
#             done
#     done
# done
