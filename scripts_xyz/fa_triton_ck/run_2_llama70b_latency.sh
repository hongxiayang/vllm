#!/bin/bash

set +ex

# the 70b fp8 kv cache model
#VLLM_USE_TRITON_FLASH_ATTN=1 ./test_cpa.sh &> triton_tp8.out
#VLLM_USE_TRITON_FLASH_ATTN=0 ./test_cpa.sh &> ck_fa_tp8.out

# the regular 70b model
VLLM_USE_TRITON_FLASH_ATTN=0 ./test_llama_rocm_latency.sh &> ck_fa_tp8_llama70b_latency.out
egrep "RUNNING|Avg latency" ck_fa_tp8_llama70b_latency.out |grep -v echo > ck_fa_tp8_llama70b_latency_short.out

VLLM_USE_TRITON_FLASH_ATTN=1 ./test_llama_rocm_latency.sh &> triton_tp8_llama70b_latency.out
egrep "RUNNING|Avg latency" triton_tp8_llama70b_latency.out |grep -v echo > triton_tp8_llama70b_latency_short.out 

python3 parse_short_latency_to_csv.py --input-file ck_fa_tp8_llama70b_latency_short.out --output-file ck_fa_tp8_llama70b_latency.csv
python3 parse_short_latency_to_csv.py --input-file triton_tp8_llama70b_latency_short.out --output-file triton_tp8_llama70b_latency.csv


