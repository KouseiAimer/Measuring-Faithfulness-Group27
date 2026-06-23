#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?Usage: generate_eval_cots.sh base|sft}"
if [[ "$ARM" != "base" && "$ARM" != "sft" ]]; then
  echo "Unknown arm: $ARM" >&2
  exit 2
fi
PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
mkdir -p artifacts/eval_cots artifacts/splits logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" -u generate_eval_cots.py \
  --arm "$ARM" --dataset local_datasets/openbookqa \
  --manifest artifacts/splits/openbookqa_test_seed1001_n100_retain20.json \
  --n-targets 100 --n-retain 20 --max-new-tokens 128 --seed 1001 \
  2>&1 | tee "logs/generate_eval_cots_${ARM}.log"

