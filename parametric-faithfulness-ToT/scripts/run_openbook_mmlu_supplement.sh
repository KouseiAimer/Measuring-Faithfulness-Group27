#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-sample_select}"
PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/local_hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ENDPOINT="https://huggingface.co"

"$PYTHON" -u unlearn-CoT.py \
  --model_name local_models/Llama-3.2-3B-Instruct --dataset openbook \
  --strategy sentencize --stepwise --method npo_KL --lr 3e-05 --epochs 5 \
  --pos --ff2 --max_samples 70 --n_unlearn 50 --verify_samples 20 --mmlu 10 \
  --reasoning_root final_cot_CoT --result_root final_result_CoT --source_tag cot \
  --split_manifest splits/openbookqa_seed1001_n50_retain20.json

CACHE="final_tree_ToT/openbook/Llama-3.2-3B-Instruct/${MODE}_test_n=70_s=1001.jsonl"
"$PYTHON" -u unlearn-ToT.py \
  --model_name local_models/Llama-3.2-3B-Instruct --dataset openbook --stepwise \
  --reasoning_cache "$CACHE" --method npo_KL --lr 3e-05 --epochs 5 \
  --pos --ff2 --max_samples 70 --n_unlearn 50 --verify_samples 20 --mmlu 10 \
  --result_root final_result_ToT \
  --split_manifest splits/openbookqa_seed1001_n50_retain20.json
