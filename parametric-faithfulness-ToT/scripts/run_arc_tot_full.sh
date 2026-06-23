#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_arc_tot_full.sh sample_select|beam_prune}"
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
  --dataset arc-challenge --split test --mode "$MODE" \
  --max_instances 120 --seed 1001 --output_dir final_tree_ToT

CACHE="final_tree_ToT/arc-challenge/Llama-3.2-3B-Instruct/${MODE}_test_n=120_s=1001.jsonl"

# Generation is light enough to run concurrently; unlearning loads an oracle
# and a trainable model, so wait for headroom if the OpenBookQA arms overlap.
while true; do
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$FREE_MIB" -ge 30000 ]]; then
    break
  fi
  echo "Waiting for GPU memory before ARC ToT unlearning: ${FREE_MIB} MiB free"
  sleep 300
done

"$PYTHON" -u unlearn-ToT.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --dataset arc-challenge --stepwise \
  --reasoning_cache "$CACHE" \
  --method npo_KL --lr 3e-05 --epochs 5 --pos --ff2 \
  --max_samples 120 --n_unlearn 100 --verify_samples 20 \
  --split_manifest splits/arc_challenge_seed1001_n100_retain20.json
