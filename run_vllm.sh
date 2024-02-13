#!/bin/bash

MODEL="/data/Llama-2-13B-Chat-fp16/"
if [[ $# -ge 1 ]]; then
   MODEL="$1"
fi

for gen_len in 1 32;
do
	for input_len in 512 1024 2048 3072;
	do
	# please do NOT change the format of hte following echo. DLM relies on this infor.
	echo "===================== RUNNING $MODEL $input_len $gen_len ==================================================="
	python benchmarks/benchmark_latency.py --model $MODEL --input-len $input_len --output-len $gen_len --batch-size 1  --tensor-parallel-size 1 --num-iters 5
	done
done
