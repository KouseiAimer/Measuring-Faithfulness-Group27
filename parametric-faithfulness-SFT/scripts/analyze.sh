#!/usr/bin/env bash
set -euo pipefail

PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
"$PYTHON" -u analyze_fur.py \
  --base-cots artifacts/eval_cots/base_openbookqa_test.jsonl \
  --sft-cots artifacts/eval_cots/sft_openbookqa_test.jsonl \
  --base-results artifacts/fur_results/base.jsonl \
  --sft-results artifacts/fur_results/sft.jsonl \
  --output-dir artifacts/analysis

