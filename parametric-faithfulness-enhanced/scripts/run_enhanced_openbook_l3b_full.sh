#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/fdu-aidake-cfff/public/liangyanpeng/Measuring-Faithfulness-Group27/parametric-faithfulness-enhanced"
cd "$ROOT"

export HF_ENDPOINT="https://huggingface.co"
export HF_HOME="$ROOT/.hf_cache"
export HF_DATASETS_CACHE="$ROOT/.hf_datasets_cache"
export TRANSFORMERS_CACHE="$ROOT/.hf_cache/transformers"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"
MODEL="local_models/Llama-3.2-3B-Instruct"
DATASET="openbook"
MAX_SAMPLES=100
N_UNLEARN=80
VERIFY_SAMPLES=20
EPOCHS=2
LR="3e-05"
NUM_SHARDS=3
TOP_K=1
SEED=1001
STRATEGY="sentencize"
METHOD="npo_KL"

MODEL_SHORT="LLaMA-3-3B"
OUT_DIR="final_results/${DATASET}/${MODEL_SHORT}"
RUN_DIR="enhanced_runs/openbook_l3b_n${MAX_SAMPLES}_e${EPOCHS}_shards${NUM_SHARDS}"
mkdir -p "$OUT_DIR" "$RUN_DIR" logs

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
  local label="$1"
  shift
  local extra_args=("$@")
  local pids=()

  echo "[$(timestamp)] Launching ${label} with ${NUM_SHARDS} shards"
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_log="logs/${label}_shard${shard_idx}.log"
    run_unlearn "$shard_log" \
      --model_name "$MODEL" \
      --local_files_only \
      --dataset "$DATASET" \
      --strategy "$STRATEGY" \
      --stepwise \
      --method "$METHOD" \
      --lr "$LR" \
      --ff2 \
      --max_samples "$MAX_SAMPLES" \
      --n_unlearn "$N_UNLEARN" \
      --verify_samples "$VERIFY_SAMPLES" \
      --epochs "$EPOCHS" \
      --cot_max_new_tokens 160 \
      --eval_max_new_tokens 120 \
      --num_shards "$NUM_SHARDS" \
      --shard_idx "$shard_idx" \
      "${extra_args[@]}" &
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
  echo "[$(timestamp)] Finished ${label}"
}

merge_and_analyze() {
  local label="$1"
  local suffix="$2"
  local base="${METHOD}_${STRATEGY}_s=True_lr=${LR}_rs=${SEED}_n=${MAX_SAMPLES}_pos=False_ff2=True${suffix}"
  local merged="${OUT_DIR}/${base}_merged.out"
  local summary="${RUN_DIR}/${label}_summary.json"

  : > "$merged"
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_file="${OUT_DIR}/${base}_shard=${shard_idx}-of-${NUM_SHARDS}.out"
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

echo "[$(timestamp)] Enhanced OpenBook LLaMA-3.2-3B run started"
echo "[$(timestamp)] model=$MODEL dataset=$DATASET max_samples=$MAX_SAMPLES n_unlearn=$N_UNLEARN verify=$VERIFY_SAMPLES epochs=$EPOCHS shards=$NUM_SHARDS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true

echo "[$(timestamp)] Stage 1: generate/reuse CoT cache"
run_unlearn "logs/openbook_l3b_cot_generation.log" \
  --model_name "$MODEL" \
  --local_files_only \
  --dataset "$DATASET" \
  --strategy "$STRATEGY" \
  --stepwise \
  --method "$METHOD" \
  --lr "$LR" \
  --ff2 \
  --max_samples "$MAX_SAMPLES" \
  --n_unlearn 0 \
  --verify_samples "$VERIFY_SAMPLES" \
  --epochs 1 \
  --cot_max_new_tokens 160 \
  --eval_max_new_tokens 120 \
  --selection_strategy last \
  --top_k "$TOP_K"

echo "[$(timestamp)] Stage 2: Full FUR"
launch_shards "openbook_l3b_full" --selection_strategy all --top_k "$TOP_K"
merge_and_analyze "full" ""

echo "[$(timestamp)] Stage 3: Selective FUR last-top1"
launch_shards "openbook_l3b_last_top1" --selection_strategy last --top_k "$TOP_K"
merge_and_analyze "last_top1" "_sel=last_k=${TOP_K}"

echo "[$(timestamp)] Stage 4: Selective FUR random-top1"
launch_shards "openbook_l3b_random_top1" --selection_strategy random --top_k "$TOP_K"
merge_and_analyze "random_top1" "_sel=random_k=${TOP_K}"

echo "[$(timestamp)] Enhanced run complete"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
