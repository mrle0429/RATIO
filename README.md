<div align="center">

<h1 align="center">RATIO: Robust AI Text Involvement Estimator</h1>

<p align="center">
   <a href="https://github.com/mrle0429">Le Liu</a>, <a href="https://github.com/YunhanGa0">Yunhan Gao</a>, <a href="https://github.com/wangLyndon">Ziheng Wang</a>, <a href="">Sicheng Yi</a>, <a href="https://github.com/beihaizhang11">Bohan Zhang</a>
</p>

</div>


RATIO is a two-stage detector for fine-grained AI involvement ratio estimation
in mixed-authorship text.

This repository is the main project entrypoint for the paper. The companion
PACT repository contains the dataset construction pipeline; RATIO consumes the
fixed JSONL splits exported by PACT and keeps the model, training, evaluation,
and reproduction code here.

The public repository keeps only the formal method path described in the paper:

1. **Stage 1: proportion-aware detector.** A DeBERTa-v3-large encoder is trained
   with a six-way ratio classification head and an auxiliary LIR regression
   head. The trainer also supports classification-regression consistency; 
2. **Stage 2: RCDPO robust training.** The Stage-1 checkpoint is used as a
   frozen reference while a policy detector is optimized on clean/rewrite pairs
   with supervised detection loss, reference-calibrated preference loss,
   clean-rewrite consistency, and reference-preservation KL.



## Repository Layout

```text
ratio_detector/      # Stage-1 RATIO detector package
ratio_rcdpo/         # Stage-2 RCDPO robust training package
scripts/             # Maintained launch scripts for the two formal stages
configs/             # Reference command/configuration notes
examples/            # Tiny JSONL examples showing expected fields
docs/                # Data format and release notes
requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

PACT is maintained as a separate repository because it is the dataset
construction pipeline. RATIO does not vendor the full dataset builder.

Dataset repository: [mrle0429/PACT](https://github.com/mrle0429/PACT)

Use the PACT exported grouped splits:

```text
PACT/datasplit/benchmark_grouped/   -> RATIO/data/pact/
PACT/datasplit/rewrite_grouped/     -> RATIO/data/rewrite/
```

If the two repositories are cloned side by side during development, this is
enough:

```bash
mkdir -p data
ln -s ../../PACT/datasplit/benchmark_grouped data/pact
ln -s ../../PACT/datasplit/rewrite_grouped data/rewrite
```

For a public release, download or build the same PACT split version and place
the files under:

```text
data/pact/train.jsonl
data/pact/val.jsonl
data/pact/test.jsonl
data/rewrite/train.jsonl
data/rewrite/val.jsonl
data/rewrite/test.jsonl
```

The expected schema is documented in `docs/DATA_FORMAT.md`. Tiny examples are
provided under `examples/`; they are format checks, not useful training data.

## Stage 1: Train RATIO

The default Stage-1 config in
`configs/stage1_ratio.json` targets the archived run
`lir_joint_mainreg`, which reports test Macro-F1 `0.5961684203974654`
and test MAE `0.1059472484423763`.

```bash
GPU_ID=0 DATA_ROOT=data/pact OUT=outputs/ratio_stage1 bash scripts/train_stage1_ratio.sh
```

Equivalent module call:

```bash
python -m ratio_detector.train \
  --train_data data/pact/train.jsonl \
  --val_data data/pact/val.jsonl \
  --test_data data/pact/test.jsonl \
  --model_name microsoft/deberta-v3-large \
  --output_dir outputs/ratio_stage1 \
  --batch_size 8 \
  --epochs 10 \
  --lr_backbone 3e-6 \
  --lr_heads 3e-4 \
  --dropout 0.2 \
  --weight_decay 0.05 \
  --warmup_ratio 0.06 \
  --grad_accum_steps 3 \
  --max_grad_norm 0.5 \
  --w_ce 1.0 \
  --w_mae 1.0 \
  --w_mse 0.0 \
  --w_huber 0.0 \
  --w_consistency 0.0 \
  --mixed_precision no \
  --no-use_weighted_sampler \
  --no-use_class_weights \
  --no-use_aux_targets \
  --no-use_continuous_features \
  --seed 42
```

Stage-1 objective:

```text
L_stage1 = w_ce * CE(class_bin)
         + w_mae * L1(LIR)
         + w_mse * MSE(LIR)
         + w_huber * Huber(LIR)
         + w_consistency * L1(expected_class_ratio, predicted_LIR)
```

For the reported `0.596` run, `w_consistency = 0.0`.

## Stage 2: Train RCDPO

```bash
GPU_ID=0 \
STAGE1_CHECKPOINT=outputs/ratio_stage1/best_model.pt \
REWRITE_ROOT=data/rewrite \
OUT=outputs/ratio_rcdpo \
bash scripts/train_stage2_rcdpo.sh
```

Equivalent module call:

```bash
python -m ratio_rcdpo.train_stage2 \
  --stage1_checkpoint outputs/ratio_stage1/best_model.pt \
  --train_data data/rewrite/train.jsonl \
  --val_data data/rewrite/val.jsonl \
  --test_data data/rewrite/test.jsonl \
  --output_dir outputs/ratio_rcdpo
```

## Evaluation

```bash
python -m ratio_detector.eval_lir_joint_checkpoint \
  --checkpoint outputs/ratio_stage1/best_model.pt \
  --test_data data/pact/test.jsonl
```

Evaluate a Stage-1 checkpoint on rewritten text:

```bash
python -m ratio_detector.eval_rewrite_test \
  --checkpoint outputs/ratio_stage1/best_model.pt \
  --test_data data/rewrite/test.jsonl \
  --text_field rewritten_text
```

## Release Notes

- `data/`, `outputs/`, `wandb/`, checkpoints, and logs are gitignored.
- Legacy code and full local artifacts are archived in
  `../advance_baseline_detector_archive/`.
- The maintained code path is `ratio_detector` for Stage 1 and `ratio_rcdpo`
  for Stage 2.
