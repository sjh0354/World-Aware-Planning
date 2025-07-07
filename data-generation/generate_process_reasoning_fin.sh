#!/bin/bash

subset_arg=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --visual)
            subset_arg="visual"
            shift
            ;;
        --symbolic)
            subset_arg="symbolic"
            shift
            ;;
        --reference)
            subset_arg="reference"
            shift
            ;;
        --spatial)
            subset_arg="spatial"
            shift
            ;;
        *)
            echo "Unknown: $1"
            echo "Using: $0 (--visual)"
            exit 1
            ;;
    esac
done

if [[ -z "$subset_arg" ]]; then
    echo "Error: Undefined subset"
    echo "Using: $0 (--visual)"
    exit 1
fi

vllm_cmd="/opt/conda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
--served-model-name qwen2vl \
--model downloaded_ckpts/Qwen2.5-VL-72B-Instruct \
--port 8000 \
--limit-mm-per-prompt image=25 \
--tensor-parallel-size 4"

res_folder="/vllm-log"

mkdir -p "${res_folder}"

nohup bash -c "$vllm_cmd" \
> >(tee -a "${res_folder}/run_stdout.log") \
2> >(tee -a "${res_folder}/run_stderr.log") &

vllm_pid=$!
echo "vLLM Process ID: $vllm_pid. Check logs in ${res_folder}/"

sleep 10m

conda activate vllm

python rewrite_step_reasoning.py \
--subset "$subset_arg"