#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-outputs/ratio_stage1/best_model.pt}"
REWRITE_ROOT="${REWRITE_ROOT:-data/rewrite}"
OUT="${OUT:-outputs/ratio_rcdpo}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

mkdir -p "$OUT"

{
  echo "============================================================"
  echo "RATIO Stage-2 RCDPO training"
  echo "module: ratio_rcdpo.train_stage2"
  echo "gpu_id: ${GPU_ID}"
  echo "seed: ${SEED}"
  echo "stage1_checkpoint: ${STAGE1_CHECKPOINT}"
  echo "rewrite_root: ${REWRITE_ROOT}"
  echo "expected_pact_split: PACT/datasplit/rewrite_grouped"
  echo "output_dir: ${OUT}"
  echo "started_at_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "============================================================"
} | tee "$OUT/screen.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m ratio_rcdpo.train_stage2 \
  --stage1_checkpoint "$STAGE1_CHECKPOINT" \
  --train_data "${REWRITE_ROOT}/train.jsonl" \
  --val_data "${REWRITE_ROOT}/val.jsonl" \
  --test_data "${REWRITE_ROOT}/test.jsonl" \
  --output_dir "$OUT" \
  --seed "$SEED" \
  --mixed_precision "$MIXED_PRECISION" \
  2>&1 | tee -a "$OUT/screen.log"

echo "finished_at_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/screen.log"
