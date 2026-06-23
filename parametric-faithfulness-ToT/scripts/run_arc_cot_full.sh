#!/usr/bin/env bash
set -euo pipefail

PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/local_hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" -u unlearn-CoT.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --dataset arc-challenge \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-05 --epochs 5 --pos --ff2 \
  --max_samples 120 --n_unlearn 100 --verify_samples 20 \
  --cot_max_new_tokens 300 \
  --reasoning_root final_cot_CoT --result_root final_result_CoT \
  --source_tag cot \
  --split_manifest splits/arc_challenge_seed1001_n100_retain20.json
