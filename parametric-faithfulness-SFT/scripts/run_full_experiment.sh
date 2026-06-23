#!/usr/bin/env bash
set -euo pipefail

PYTHON="/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python"
cd "$(dirname "$0")/.."
mkdir -p logs artifacts
MASTER="logs/run_full_experiment.master.log"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

record() {
  echo "[$(timestamp)] $*" | tee -a "$MASTER"
}

record "Starting/resuming DeepSeek-revision SFT full experiment."

record "Stage 1/7: generating student training drafts."
bash scripts/generate_training_data.sh drafts

record "Stage 2/7: requesting and filtering DeepSeek revisions."
bash scripts/generate_training_data.sh revisions
SFT_ROWS="$(wc -l < artifacts/data/sft_train.jsonl)"
record "Filtered SFT dataset contains ${SFT_ROWS} records."
if [[ "$SFT_ROWS" -lt 500 ]]; then
  record "Stopping: at least 500 accepted records are required for formal SFT."
  exit 1
fi

MERGED="artifacts/models/deepseek_revision_sft/merged/Llama-3.2-3B-Instruct/config.json"
if [[ -f "$MERGED" ]]; then
  record "Stage 3/7: merged SFT model already exists; skipping training to avoid a second SFT run."
else
  record "Stage 3/7: executing the unique LoRA-SFT training run."
  bash scripts/train_once.sh
fi

record "Stage 4/7: generating independent base test CoTs."
bash scripts/generate_eval_cots.sh base

record "Stage 5/7: generating independent SFT test CoTs."
bash scripts/generate_eval_cots.sh sft

record "Stage 6/7: running complete base FUR."
bash scripts/run_base_fur.sh

record "Stage 7/7: running complete SFT FUR and summarizing."
bash scripts/run_sft_fur.sh
bash scripts/analyze.sh | tee -a "$MASTER"

record "Full experiment finished. Read artifacts/analysis/metrics.md."

