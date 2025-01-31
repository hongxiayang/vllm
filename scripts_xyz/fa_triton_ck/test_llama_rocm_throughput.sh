#!/bin/bash

# set -e will stop when error happens

set +ex

MODEL="/data/Meta-Llama-3-70B-Instruct"


if [[ $# -ge 1 ]]; then
   MODEL="$1"
fi

for TP in 1 4 8;
  do
    for output_len in 128 256 2048;
    do
        for input_len in 128 512 2048;
        do
        echo "===================== RUNNING $MODEL $input_len $output_len tp=$TP ==================================================="
        python /app/vllm/benchmarks/benchmark_throughput.py --model $MODEL \
        --kv-cache-dtype fp8_e4m3 --quantization fp8 \
        --input-len $input_len --output-len $output_len \
        --distributed-executor-backend mp --tensor-parallel-size $TP --num-prompts 200 --num-scheduler-steps 10 \
        --enable-chunked-prefill=False
        done
    done
done

# DATASET_PATH="/dockerx/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

# for TP in 1 8;
#     do
#     echo "===================== RUNNING $MODEL shared_GPT tp=$TP ==================================================="
#     python /app/vllm/benchmarks/benchmark_throughput.py --model $MODEL --dataset $DATASET_PATH \
#     --distributed-executor-backend mp --tensor-parallel-size $TP --num-prompts 1000 --num-scheduler-steps 10 \
#     --enable-chunked-prefill=False

#     sleep 2
# done
