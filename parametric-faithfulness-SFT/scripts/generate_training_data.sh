#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?Usage: generate_training_data.sh drafts|revisions}"
PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
mkdir -p artifacts/data logs

if [[ "$PHASE" == "drafts" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$PYTHON" -u generate_drafts.py \
    --model local_models/Llama-3.2-3B-Instruct \
    --dataset local_datasets/openbookqa \
    --max-samples 2000 --max-new-tokens 128 --batch-size 32 --seed 1001 \
    2>&1 | tee logs/generate_drafts.log
elif [[ "$PHASE" == "revisions" ]]; then
  "$PYTHON" -u revise_drafts.py \
    --model deepseek-v4-pro --max-tokens 4096 --min-words 20 --max-words 80 --workers 8 \
    2>&1 | tee logs/revise_drafts.log
  "$PYTHON" -u revise_drafts.py \
    --model deepseek-v4-pro --max-tokens 4096 --min-words 20 --max-words 80 --workers 4 --retry-errors \
    2>&1 | tee -a logs/revise_drafts.log
else
  echo "Unknown phase: $PHASE" >&2
  exit 2
fi
