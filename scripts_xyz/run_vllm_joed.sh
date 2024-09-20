#!/bin/bash

MODEL="/data/llama-2-7b-chat-hf"
#MODEL="/data/llama2-70b-chat"
if [[ $# -ge 1 ]]; then
   MODEL="$1"
fi

for gen_len in 1024;
do
	for input_len in 1024;
	do
	# please do NOT change the format of hte following echo. DLM relies on this infor.
	echo "===================== RUNNING $MODEL $input_len $gen_len ==================================================="
	python /app/vllm/benchmarks/benchmark_latency.py --enforce-eager --model $MODEL --input-len $input_len --output-len $gen_len --batch-size 8  --tensor-parallel-size 1 --num-iters 10 
	done
done
