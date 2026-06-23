#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_openbook_tot_full.sh sample_select|beam_prune}"
if [[ "$MODE" != "sample_select" && "$MODE" != "beam_prune" ]]; then
  echo "Unknown ToT mode: $MODE" >&2
  exit 2
fi

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
  --dataset openbook --split test --mode "$MODE" \
  --max_instances 70 --seed 1001 --output_dir final_tree_ToT

CACHE="final_tree_ToT/openbook/Llama-3.2-3B-Instruct/${MODE}_test_n=70_s=1001.jsonl"

"$PYTHON" -u unlearn-ToT.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --dataset openbook --stepwise \
  --reasoning_cache "$CACHE" \
  --method npo_KL --lr 3e-05 --epochs 5 --pos --ff2 \
  --max_samples 70 --n_unlearn 50 --verify_samples 20 \
  --split_manifest splits/openbookqa_seed1001_n50_retain20.json
