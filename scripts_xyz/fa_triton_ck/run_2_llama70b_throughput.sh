#!/bin/bash

set +ex

# the 70b fp8 kv cache model
#VLLM_USE_TRITON_FLASH_ATTN=1 ./test_cpa.sh &> triton_tp8.out
#VLLM_USE_TRITON_FLASH_ATTN=0 ./test_cpa.sh &> ck_fa_tp8.out

# the regular 70b model
VLLM_USE_TRITON_FLASH_ATTN=0 ./test_llama_rocm_throughput.sh &> ck_fa_llama70b_fixed_throughput.out
egrep "RUNNING|Throughput:" ck_fa_llama70b_fixed_throughput.out |grep -v echo > ck_fa_llama70b_fixed_throughput_short.out

# VLLM_USE_TRITON_FLASH_ATTN=1 ./test_llama_rocm_throughput.sh &> triton_llama70b_fixed_throughput1.out
# egrep "RUNNING|Throughput:" triton_llama70b_fixed_throughput1.out |grep -v echo > triton_llama70b_fixed_throughput1_short.out 


# VLLM_USE_TRITON_FLASH_ATTN=1 ./test_llama_rocm_throughput.sh &> triton_llama70b_fixed_throughput2.out
# egrep "RUNNING|Throughput:" triton_llama70b_fixed_throughput2.out |grep -v echo > triton_llama70b_fixed_throughput2_short.out 

# python3 parse_short_thru_to_csv.py --input-file ck_fa_llama70b_fixed_throughput_short.out --output-file ck_fa_llama70b_fixed_throughput.csv
# python3 parse_short_thru_to_csv.py --input-file triton_llama70b_fixed_throughput1_short.out --output-file triton_llama70b_fixed_throughput1.csv
