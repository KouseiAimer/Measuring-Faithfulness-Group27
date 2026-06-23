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

"$PYTHON" -u tot_generation.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --dataset arc-challenge --split validation --mode compare \
  --max_instances 30 --seed 1001 --output_dir final_tree_ToT

REPORT="final_tree_ToT/arc-challenge/Llama-3.2-3B-Instruct/mode_selection_validation_n=30_s=1001.json"
MODE="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["chosen_mode"])' "$REPORT")"
echo "Selected formal ToT mode for ARC-Challenge: $MODE"
bash scripts/run_arc_tot_full.sh "$MODE"
