# RATIO Stage 2: RCDPO

This package contains the formal second-stage robust training path for RATIO.
It initializes from a Stage-1 RATIO checkpoint and uses that checkpoint as a
frozen reference model.

## Objective

Stage 2 trains on clean/rewrite pairs with:

- supervised detector loss on clean `mixed_text`
- supervised detector loss on `rewritten_text`
- RCDPO label-preference loss against the frozen Stage-1 reference
- clean/rewrite prediction consistency
- KL preservation to the frozen Stage-1 detector on clean text

The supervised detector loss uses the same classification, LIR regression, and
classification-regression consistency structure as Stage 1.

## Data

Each rewrite record should contain:

```json
{
  "id": "sample_id",
  "mixed_text": "original mixed-authorship text",
  "rewritten_text": "humanized rewrite",
  "target_ai_ratio": 0.4,
  "lir": 0.37,
  "rewrite_info": {"status": "ok"}
}
```

Use the split layout documented in the root README:

```text
data/rewrite/train.jsonl
data/rewrite/val.jsonl
data/rewrite/test.jsonl
```

## Run

```bash
python -m ratio_rcdpo.train_stage2 \
  --stage1_checkpoint outputs/ratio_stage1/best_model.pt \
  --train_data data/rewrite/train.jsonl \
  --val_data data/rewrite/val.jsonl \
  --test_data data/rewrite/test.jsonl \
  --output_dir outputs/ratio_rcdpo
```

Or use the maintained launcher:

```bash
STAGE1_CHECKPOINT=outputs/ratio_stage1/best_model.pt \
REWRITE_ROOT=data/rewrite \
OUT=outputs/ratio_rcdpo \
bash scripts/train_stage2_rcdpo.sh
```
