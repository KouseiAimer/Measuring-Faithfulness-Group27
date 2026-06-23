#!/usr/bin/env bash
set -euo pipefail

PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
mkdir -p artifacts/fur_results logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" -u run_fur.py \
  --arm base --model local_models/Llama-3.2-3B-Instruct \
  --cots artifacts/eval_cots/base_openbookqa_test.jsonl \
  --output artifacts/fur_results/base.jsonl \
  --lr 3e-05 --epochs 5 --beta 0.1 --kl-coeff 1.0 \
  --n-retain-steps 4 --new-cot-tokens 128 --seed 1001 \
  2>&1 | tee logs/run_base_fur.log

