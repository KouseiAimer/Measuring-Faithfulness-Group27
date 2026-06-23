#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${RUN_HF_HOME:-$ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${RUN_HF_DATASETS_CACHE:-$ROOT/.hf_datasets_cache}"
export TRANSFORMERS_CACHE="${RUN_TRANSFORMERS_CACHE:-$ROOT/.hf_cache/transformers}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"
MODEL="${MODEL:-local_models/Llama-3.2-3B-Instruct}"
MODEL_SHORT="${MODEL_SHORT:-LLaMA-3-3B}"
DATASETS="${DATASETS:-arc-challenge openbook}"
MAX_SAMPLES="${MAX_SAMPLES:-250}"
N_UNLEARN="${N_UNLEARN:-250}"
VERIFY_SAMPLES="${VERIFY_SAMPLES:-20}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-05}"
NUM_SHARDS="${NUM_SHARDS:-3}"
SEED="${SEED:-1001}"
STRATEGY="${STRATEGY:-sentencize}"
METHOD="${METHOD:-npo_KL}"
COT_MAX_NEW_TOKENS="${COT_MAX_NEW_TOKENS:-300}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-300}"
RUN_DIR="${RUN_DIR:-paper_runs/l3b_arc_openbook_n${MAX_SAMPLES}_e${EPOCHS}_shards${NUM_SHARDS}}"

mkdir -p "$RUN_DIR" logs

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_unlearn() {
  local logfile="$1"
  shift
  echo "[$(timestamp)] START $logfile"
  "$CONDA_BIN" run --no-capture-output -n faith python -u unlearn.py "$@" > "$logfile" 2>&1
  echo "[$(timestamp)] DONE  $logfile"
}

launch_shards() {
  local dataset="$1"
  local label="${dataset}_l3b_paper"
  local pids=()

  echo "[$(timestamp)] Launching ${label} with ${NUM_SHARDS} shards"
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_log="logs/${label}_shard${shard_idx}.log"
    run_unlearn "$shard_log" \
      --model_name "$MODEL" \
      --local_files_only \
      --dataset "$dataset" \
      --strategy "$STRATEGY" \
      --stepwise \
      --method "$METHOD" \
      --lr "$LR" \
      --pos \
      --ff2 \
      --max_samples "$MAX_SAMPLES" \
      --n_unlearn "$N_UNLEARN" \
      --verify_samples "$VERIFY_SAMPLES" \
      --epochs "$EPOCHS" \
      --cot_max_new_tokens "$COT_MAX_NEW_TOKENS" \
      --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
      --selection_strategy all \
      --num_shards "$NUM_SHARDS" \
      --shard_idx "$shard_idx" &
    pids+=("$!")
    sleep 15
  done

  local failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  if [[ "$failures" -ne 0 ]]; then
    echo "[$(timestamp)] ${label} failed in ${failures} shard(s)" >&2
    return 1
  fi
}

merge_and_analyze() {
  local dataset="$1"
  local out_dir="final_results/${dataset}/${MODEL_SHORT}"
  local base="${METHOD}_${STRATEGY}_s=True_lr=${LR}_rs=${SEED}_n=${MAX_SAMPLES}_pos=True_ff2=True"
  local merged="${out_dir}/${base}_merged.out"
  local summary="${RUN_DIR}/${dataset}_summary.json"

  : > "$merged"
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_file="${out_dir}/${base}_shard=${shard_idx}-of-${NUM_SHARDS}.out"
    if [[ -s "$shard_file" ]]; then
      cat "$shard_file" >> "$merged"
    else
      echo "[$(timestamp)] Missing or empty shard file: $shard_file" >&2
    fi
  done

  "$CONDA_BIN" run --no-capture-output -n faith python analyze_enhanced_results.py \
    --result_file "$merged" \
    --out_json "$summary"
}

echo "[$(timestamp)] LLaMA-3.2-3B paper-parameter run started"
echo "[$(timestamp)] model=$MODEL datasets=$DATASETS max_samples=$MAX_SAMPLES n_unlearn=$N_UNLEARN verify=$VERIFY_SAMPLES epochs=$EPOCHS lr=$LR shards=$NUM_SHARDS pos=true ff2=true"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true

for dataset in $DATASETS; do
  echo "[$(timestamp)] Dataset: $dataset"
  run_unlearn "logs/${dataset}_l3b_cot_generation.log" \
    --model_name "$MODEL" \
    --local_files_only \
    --dataset "$dataset" \
    --strategy "$STRATEGY" \
    --stepwise \
    --method "$METHOD" \
    --lr "$LR" \
    --pos \
    --ff2 \
    --max_samples "$MAX_SAMPLES" \
    --n_unlearn 0 \
    --verify_samples "$VERIFY_SAMPLES" \
    --epochs 1 \
    --cot_max_new_tokens "$COT_MAX_NEW_TOKENS" \
    --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --selection_strategy all

  launch_shards "$dataset"
  merge_and_analyze "$dataset"
done

echo "[$(timestamp)] Paper-parameter run complete"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
