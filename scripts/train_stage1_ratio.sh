#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-data/pact}"
OUT="${OUT:-outputs/ratio_stage1}"
MODEL_NAME="${MODEL_NAME:-microsoft/deberta-v3-large}"

mkdir -p "$OUT"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

{
  echo "============================================================"
  echo "RATIO Stage-1 training"
  echo "module: ratio_detector.train"
  echo "gpu_id: ${GPU_ID}"
  echo "seed: ${SEED}"
  echo "data_root: ${DATA_ROOT}"
  echo "expected_pact_split: PACT/datasplit/benchmark_grouped"
  echo "model_name: ${MODEL_NAME}"
  echo "output_dir: ${OUT}"
  echo "config: configs/stage1_ratio.json"
  echo "target_result: test_macro_f1=0.5961684203974654"
  echo "objective: CE + LIR regression (published-best config uses w_consistency=0.0)"
  echo "started_at_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "============================================================"
} | tee "$OUT/screen.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m ratio_detector.train \
  --train_data "${DATA_ROOT}/train.jsonl" \
  --val_data "${DATA_ROOT}/val.jsonl" \
  --test_data "${DATA_ROOT}/test.jsonl" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUT" \
  --seed "$SEED" \
  --max_seq_len 512 \
  --batch_size 8 \
  --epochs 10 \
  --lr_backbone 3e-6 \
  --lr_heads 3e-4 \
  --dropout 0.2 \
  --weight_decay 0.05 \
  --adam_eps 1e-6 \
  --warmup_ratio 0.06 \
  --grad_accum_steps 3 \
  --max_grad_norm 0.5 \
  --w_ce 1.0 \
  --w_mae 1.0 \
  --w_mse 0.0 \
  --w_huber 0.0 \
  --w_consistency 0.0 \
  --label_smoothing 0.0 \
  --early_stop_patience 3 \
  --monitor_metric macro_f1 \
  --mixed_precision no \
  --gradient_checkpointing \
  --num_workers 4 \
  --pin_memory \
  --no-use_weighted_sampler \
  --no-use_class_weights \
  --no-use_aux_targets \
  --no-use_continuous_features \
  2>&1 | tee -a "$OUT/screen.log"

echo "finished_at_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/screen.log"
