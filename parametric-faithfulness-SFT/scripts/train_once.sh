#!/usr/bin/env bash
set -euo pipefail

PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
mkdir -p artifacts/models logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" -u train_lora.py \
  --model local_models/Llama-3.2-3B-Instruct \
  --train-data artifacts/data/sft_train.jsonl \
  --output-dir artifacts/models/deepseek_revision_sft \
  --rank 32 --alpha 64 --dropout 0.05 \
  --max-length 512 --batch-size 2 --gradient-accumulation 16 \
  --learning-rate 1e-4 --epochs 1 --seed 1001 \
  2>&1 | tee logs/train_lora.log

