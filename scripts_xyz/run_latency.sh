#!/bin/bash

mkdir -p /dockerx/dataset
export TOKENIZERS_PARALLELISM=false
#export NCCL_WORK_FIFO_DEPTH=262144
#export NCCL_WORK_FIFO_DEPTH=524288 #40%
#export NCCL_WORK_FIFO_DEPTH=786432 # failed at 0%
#export NCCL_WORK_FIFO_DEPTH=589824
#export NCCL_DEBUG=VERSION
#export NCCL_DEBUG=INFO
#export LD_LIBRARY_PATH=/dockerx/rccl/build/release/:$LD_LIBRARY_PATH
#export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
#export HSA_ENABLE_DEBUG=1

main () {
  cd /app
  positional_args=()

  #model_path="/app/model"
  model_path="/data/llama2-70b-chat"
  #model_path="/dockerx/data/llama-2-7b-chat-hf"
  dataset_path="/dockerx/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
  dataset_path_modified=0

  while [[ $# -gt 0 ]]; do
    case $1 in
      "-h"|"--help")
        python3 /app/vllm/benchmarks/benchmark_latency.py --help
        return 0
        ;;
      "--dataset")
        dataset_path="$2"
        dataset_path_modified=1
        shift
        shift
        ;;
      "--model")
        model_path="$2"
        shift
        shift
        ;;
      *)
        positional_args+=("$1")
        shift
        ;;
    esac
  done

  if [ ! -f "$dataset_path" ]; then
    if [[ $dataset_path_modified -lt 1 ]]; then
      cd /dockerx/dataset
      wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
      cd /app
    fi
  fi

  #python3 /app/vllm/benchmarks/benchmark_throughput.py --dataset "$dataset_path" --model "$model_path" "${positional_args[@]}"
  #RAY_DEDUP_LOGS=0 python3 /dockerx/vllm/benchmarks/benchmark_throughput.py --dataset "$dataset_path" --model "$model_path" "${positional_args[@]}" &
  #ROCR_VISIBLE_DEVICES="0,1,2,3" python3 /dockerx/vllm/benchmarks/benchmark_throughput.py --dataset "$dataset_path" --model "$model_path" -tp 4 --enforce-eager &
  #ROCR_VISIBLE_DEVICES="4,5,6,7" python3 /dockerx/vllm/benchmarks/benchmark_throughput.py --dataset "$dataset_path" --model "$model_path" -tp 4 --enforce-eager &

  #RAY_DEDUP_LOGS=0 python3 /app/vllm/benchmarks/benchmark_throughput.py --input-len 64 --output-len 128 --model "$model_path" "${positional_args[@]}"
  python3 /app/vllm/benchmarks/benchmark_latency.py --model "$model_path" "${positional_args[@]}"
  return $?

}

main "$@"
exit $?

