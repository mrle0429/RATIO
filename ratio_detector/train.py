"""
Main RATIO detector training script.

The Stage-1 trainer supports the supervised signals used by RATIO:
1. 6-way document classification over target_ai_ratio bins
2. LIR as the main regression target
3. optional classification-regression consistency between the expected class
   ratio and the LIR regression estimate
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from ratio_detector.model import MultiHeadDetector
from ratio_detector.training_utils import (
    assert_finite_model_outputs,
    aux_loss_weights,
    build_param_groups,
    count_bins,
    detector_input_mode,
    enable_gradient_checkpointing,
    finite_gradient_report,
    finite_parameter_report,
    get_aux_target_mask,
    get_aux_targets,
    get_continuous_features,
    load_jsonl_records,
    monitor_mode,
    monitor_value,
    safe_torch_save,
    sanitize_model_outputs,
)
from ratio_detector.utils import (
    AUX_TARGET_NAMES,
    BIN_NAMES,
    EarlyStopping,
    ProportionDataset,
    compute_class_weights,
    compute_metrics,
    get_weighted_sampler,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-1 joint training: 6-way classification + LIR regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--train_data", default="data/pact/train.jsonl")
    p.add_argument("--val_data", default="data/pact/val.jsonl")
    p.add_argument("--test_data", default="data/pact/test.jsonl")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)

    # Model
    p.add_argument("--model_name", default="microsoft/deberta-v3-large")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument(
        "--use_continuous_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to feed jaccard/sentence_jaccard/cosine/lir as metadata inputs.",
    )

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_backbone", type=float, default=3e-6)
    p.add_argument("--lr_heads", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_accum_steps", type=int, default=3)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Sampling / weighting
    p.add_argument("--use_weighted_sampler", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--allow_double_balancing", action="store_true", default=False)
    p.add_argument("--weighting_method", type=str, default="sqrt_inverse")
    p.add_argument("--zero_class_boost", type=float, default=6.0)

    # Main losses
    p.add_argument("--w_ce", type=float, default=1.0)
    p.add_argument("--w_mae", type=float, default=1.0, help="Main regression L1 weight against target_lir.")
    p.add_argument("--w_mse", type=float, default=0.0)
    p.add_argument("--w_huber", type=float, default=0.0)
    p.add_argument(
        "--w_consistency",
        type=float,
        default=0.0,
        help="L1 consistency weight between the expected class ratio and LIR regression output.",
    )

    # Retained only for checkpoint/script compatibility with older experiments.
    p.add_argument("--w_cls_ratio", type=float, default=0.0)
    p.add_argument("--w_ordinal", type=float, default=0.0)
    p.add_argument("--w_lir_ratio_consistency", type=float, default=0.0)

    # Optional auxiliary metric prediction head
    p.add_argument("--use_aux_targets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--w_aux", type=float, default=0.0)
    p.add_argument("--w_aux_mse", type=float, default=0.0)
    p.add_argument("--w_aux_lir", type=float, default=0.0)
    p.add_argument("--w_aux_jaccard", type=float, default=1.0)
    p.add_argument("--w_aux_sentence_jaccard", type=float, default=1.0)
    p.add_argument("--w_aux_cosine", type=float, default=0.5)

    # Regularization
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # Early stopping / monitoring
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument(
        "--monitor_metric",
        default="macro_f1",
        choices=["macro_f1", "mae", "composite", "auroc"],
    )

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mixed_precision", default="no")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output_dir", default="outputs/ratio_stage1")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_epoch_checkpoints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_last_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_nonfinite_steps", type=int, default=20)
    p.add_argument("--strict_finite_outputs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail_on_nonfinite_grad", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--debug_first_updates", type=int, default=3)
    p.add_argument("--max_train_batches", type=int, default=0)
    return p.parse_args()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def expected_ratio(logits: torch.Tensor) -> torch.Tensor:
    ratio_bins = torch.linspace(0.0, 1.0, steps=logits.size(-1), device=logits.device)
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * ratio_bins).sum(dim=-1)


def compute_loss_lir(
    model: MultiHeadDetector,
    batch: dict[str, torch.Tensor],
    cont_features: torch.Tensor | None,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    bins = batch["proportion_bin"]
    lir_targets = torch.nan_to_num(
        batch["target_lir"].float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    aux_targets = get_aux_targets(batch)
    aux_mask = get_aux_target_mask(batch)
    use_aux = bool(args.use_aux_targets and aux_targets is not None)

    model_outputs = model(
        ids,
        attn,
        continuous_features=cont_features,
        return_aux=use_aux,
    )
    if use_aux:
        logits, reg, aux_pred = model_outputs
    else:
        logits, reg = model_outputs
        aux_pred = None

    if args.strict_finite_outputs:
        assert_finite_model_outputs(logits, reg, aux_pred)
    logits, reg, aux_pred = sanitize_model_outputs(logits, reg, aux_pred)

    ce_kwargs: dict[str, Any] = {"label_smoothing": args.label_smoothing}
    if class_weights is not None:
        ce_kwargs["weight"] = class_weights.to(logits.device)
    l_ce = F.cross_entropy(logits, bins.long(), **ce_kwargs)

    l_mae = F.l1_loss(reg, lir_targets)
    l_mse = F.mse_loss(reg, lir_targets)
    l_huber = F.smooth_l1_loss(reg, lir_targets)

    l_cons = F.l1_loss(expected_ratio(logits), reg)

    if args.use_aux_targets and aux_pred is not None and aux_targets is not None:
        weights = aux_loss_weights(args, aux_pred.device)
        if aux_mask is None:
            aux_mask = torch.ones_like(aux_targets)
        weighted_l1 = (aux_pred - aux_targets).abs() * aux_mask
        weighted_mse = ((aux_pred - aux_targets) ** 2) * aux_mask
        per_target_den = aux_mask.sum(dim=0).clamp(min=1.0)
        l_aux_l1_per_target = weighted_l1.sum(dim=0) / per_target_den
        l_aux_mse_per_target = weighted_mse.sum(dim=0) / per_target_den
        l_aux = (l_aux_l1_per_target * weights).sum() / weights.sum().clamp(min=1e-8)
        l_aux_mse = (l_aux_mse_per_target * weights).sum() / weights.sum().clamp(min=1e-8)
    else:
        l_aux = torch.tensor(0.0, device=logits.device)
        l_aux_mse = torch.tensor(0.0, device=logits.device)

    loss = (
        args.w_ce * l_ce
        + args.w_mae * l_mae
        + args.w_mse * l_mse
        + args.w_huber * l_huber
        + args.w_consistency * l_cons
        + args.w_aux * l_aux
        + args.w_aux_mse * l_aux_mse
    )
    return loss, l_ce, l_mae, l_mse, l_huber, l_cons, l_aux, l_aux_mse


def evaluate_lir(
    model: MultiHeadDetector,
    dataloader: DataLoader,
    cont_features_enabled: bool,
    accelerator: Accelerator,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels, all_reg_preds, all_reg_targets = [], [], [], []
    all_aux_preds, all_aux_targets, all_aux_masks, all_cls_probs = [], [], [], []
    loss_sums = {
        "loss": 0.0,
        "ce": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "huber": 0.0,
        "consistency": 0.0,
        "aux": 0.0,
        "aux_mse": 0.0,
    }
    loss_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            cont = get_continuous_features(batch, cont_features_enabled)
            loss, l_ce, l_mae, l_mse, l_huber, l_cons, l_aux, l_aux_mse = compute_loss_lir(
                model,
                batch,
                cont,
                args,
                class_weights=class_weights,
            )

            bins = batch["proportion_bin"]
            lir_targets = torch.nan_to_num(
                batch["target_lir"].float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"],
                continuous_features=cont,
                return_aux=bool(args.use_aux_targets and "aux_targets" in batch),
            )
            if isinstance(outputs, tuple) and len(outputs) == 3:
                logits, reg, aux_pred = outputs
            else:
                logits, reg = outputs
                aux_pred = None
            logits, reg, aux_pred = sanitize_model_outputs(logits, reg, aux_pred)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1)

            preds = accelerator.gather_for_metrics(preds.detach())
            bins = accelerator.gather_for_metrics(bins.detach())
            reg = accelerator.gather_for_metrics(reg.detach())
            lir_targets = accelerator.gather_for_metrics(lir_targets.detach())
            probs = accelerator.gather_for_metrics(probs.detach())

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(bins.cpu().tolist())
            all_reg_preds.extend(reg.cpu().tolist())
            all_reg_targets.extend(lir_targets.cpu().tolist())
            if aux_pred is not None and "aux_targets" in batch:
                gathered_aux_pred = accelerator.gather_for_metrics(aux_pred.detach())
                gathered_aux_targets = accelerator.gather_for_metrics(batch["aux_targets"].detach())
                all_aux_preds.extend(gathered_aux_pred.cpu().tolist())
                all_aux_targets.extend(gathered_aux_targets.cpu().tolist())
                if "aux_target_mask" in batch:
                    gathered_aux_mask = accelerator.gather_for_metrics(batch["aux_target_mask"].detach())
                    all_aux_masks.extend(gathered_aux_mask.cpu().tolist())
            all_cls_probs.append(probs.cpu().numpy())

            loss_sums["loss"] += float(loss.item())
            loss_sums["ce"] += float(l_ce.item())
            loss_sums["mae"] += float(l_mae.item())
            loss_sums["mse"] += float(l_mse.item())
            loss_sums["huber"] += float(l_huber.item())
            loss_sums["consistency"] += float(l_cons.item())
            loss_sums["aux"] += float(l_aux.item())
            loss_sums["aux_mse"] += float(l_aux_mse.item())
            loss_steps += 1

    loss_vector = torch.tensor(
        [
            loss_sums["loss"],
            loss_sums["ce"],
            loss_sums["mae"],
            loss_sums["mse"],
            loss_sums["huber"],
            loss_sums["consistency"],
            loss_sums["aux"],
            loss_sums["aux_mse"],
            float(loss_steps),
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    loss_vector = accelerator.reduce(loss_vector, reduction="sum")
    loss_steps = int(loss_vector[-1].item())

    metrics = compute_metrics(
        all_preds,
        all_labels,
        all_reg_preds,
        all_reg_targets,
        cls_probs=np.concatenate(all_cls_probs, axis=0),
        aux_preds=all_aux_preds if all_aux_preds else None,
        aux_targets=all_aux_targets if all_aux_targets else None,
        aux_masks=all_aux_masks if all_aux_masks else None,
        aux_names=AUX_TARGET_NAMES,
    )
    if loss_steps > 0:
        metrics.update(
            {
                "loss": float(loss_vector[0].item()) / loss_steps,
                "ce_loss": float(loss_vector[1].item()) / loss_steps,
                "mae_loss": float(loss_vector[2].item()) / loss_steps,
                "mse_loss": float(loss_vector[3].item()) / loss_steps,
                "huber_loss": float(loss_vector[4].item()) / loss_steps,
                "consistency_loss": float(loss_vector[5].item()) / loss_steps,
                "aux_loss": float(loss_vector[6].item()) / loss_steps,
                "aux_mse_loss": float(loss_vector[7].item()) / loss_steps,
            }
        )
    metrics["main_regression_target"] = "lir"
    metrics["stage1_objective"] = "ce_lir_consistency"
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    if any(v != 0.0 for v in (args.w_cls_ratio, args.w_ordinal, args.w_lir_ratio_consistency)):
        if accelerator.is_main_process:
            print("[WARN] Legacy ratio-target losses are disabled because RATIO regresses LIR.")
        args.w_cls_ratio = 0.0
        args.w_ordinal = 0.0
        args.w_lir_ratio_consistency = 0.0

    if args.use_weighted_sampler and args.use_class_weights and not args.allow_double_balancing:
        if accelerator.is_main_process:
            print("[WARN] WeightedRandomSampler and CE class weights were both enabled; CE class weights are disabled for this run.")
        args.use_class_weights = False

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("  RATIO Stage-1 Training")
        print("=" * 60)
        print(f"  输出目录: {output_dir}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  训练轮数: {args.epochs}")
        print(f"  早停 patience: {args.early_stop_patience}")
        print(f"  Monitor: {args.monitor_metric}")
        print(f"  LR backbone / heads: {args.lr_backbone} / {args.lr_heads}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Warmup ratio: {args.warmup_ratio}")
        print(f"  Grad accum / max grad norm: {args.grad_accum_steps} / {args.max_grad_norm}")
        print(f"  回归主目标: LIR")
        print(f"  6分类目标: target_ai_ratio bins")
        print(f"  分类-回归一致性权重: {args.w_consistency}")
        print(f"  Seed: {args.seed}")
        print(f"  Deterministic: {args.deterministic}")
        print(f"  连续指标辅助输出: {'启用' if args.use_aux_targets else '禁用'}")
        print("=" * 60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_records = load_jsonl_records(args.train_data)
    val_records = load_jsonl_records(args.val_data)
    test_records = load_jsonl_records(args.test_data)

    train_ds = ProportionDataset(train_records, tokenizer, max_length=args.max_seq_len, load_continuous_features=args.use_continuous_features)
    val_ds = ProportionDataset(val_records, tokenizer, max_length=args.max_seq_len, load_continuous_features=args.use_continuous_features)
    test_ds = ProportionDataset(test_records, tokenizer, max_length=args.max_seq_len, load_continuous_features=args.use_continuous_features)

    class_weights = None
    weighted_sampler = None
    if args.use_class_weights or args.use_weighted_sampler:
        train_labels = train_ds.bins.numpy()
        class_weights = compute_class_weights(
            train_labels,
            method=args.weighting_method,
            zero_class_boost=args.zero_class_boost,
        )
        if not args.use_class_weights:
            class_weights = None
        if args.use_weighted_sampler:
            weighted_sampler = get_weighted_sampler(
                train_labels,
                num_samples=len(train_ds),
                zero_class_boost=args.zero_class_boost,
                method=args.weighting_method,
            )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_common = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0 and args.persistent_workers:
        loader_common["persistent_workers"] = True
    train_kwargs = {"batch_size": args.batch_size, **loader_common}
    if weighted_sampler is not None:
        train_kwargs["sampler"] = weighted_sampler
    else:
        train_kwargs["shuffle"] = True

    train_dl = DataLoader(train_ds, **train_kwargs)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, **loader_common)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2, **loader_common)

    model = MultiHeadDetector(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout=args.dropout,
        use_continuous_features=args.use_continuous_features,
    )
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    model.set_special_token_ids(tokenizer)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, args),
        eps=args.adam_eps,
        foreach=False,
        fused=False,
    )
    steps_per_epoch = (len(train_dl) + args.grad_accum_steps - 1) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1

    model, optimizer, train_dl, val_dl, test_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, test_dl, scheduler
    )

    stopper = EarlyStopping(patience=args.early_stop_patience, mode=monitor_mode(args.monitor_metric))
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {
            "total": 0.0,
            "ce": 0.0,
            "mae": 0.0,
            "mse": 0.0,
            "huber": 0.0,
            "consistency": 0.0,
            "aux": 0.0,
            "aux_mse": 0.0,
        }
        num_updates = 0
        nonfinite_steps = 0
        sampled_counts = torch.zeros(args.num_classes, dtype=torch.long)

        epoch_train_batches = len(train_dl)
        if args.max_train_batches > 0:
            epoch_train_batches = min(epoch_train_batches, args.max_train_batches)
        train_iter = iter(train_dl)
        pbar = tqdm(range(epoch_train_batches), desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)

        for _ in pbar:
            batch = next(train_iter)
            with accelerator.accumulate(model):
                bins = batch["proportion_bin"]
                cont = get_continuous_features(batch, args.use_continuous_features)
                loss, l_ce, l_mae, l_mse, l_huber, l_cons, l_aux, l_aux_mse = compute_loss_lir(
                    model, batch, cont, args, class_weights=class_weights
                )

                if not torch.isfinite(loss):
                    nonfinite_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if nonfinite_steps >= args.max_nonfinite_steps:
                        raise RuntimeError("Too many non-finite training steps.")
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_ok, grad_norm, grad_max_abs, bad_grads = finite_gradient_report(model)
                    if accelerator.is_main_process and num_updates < args.debug_first_updates:
                        print(f"[DEBUG] update {num_updates + 1}: grad_norm={grad_norm:.4e}, grad_max_abs={grad_max_abs:.4e}, bad_grads={bad_grads}")
                    if not grad_ok:
                        nonfinite_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        if args.fail_on_nonfinite_grad:
                            raise FloatingPointError("Backward produced NaN/Inf gradients: " + "; ".join(bad_grads))
                        if nonfinite_steps >= args.max_nonfinite_steps:
                            raise RuntimeError("Too many non-finite gradient steps.")
                        continue
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if accelerator.sync_gradients and args.strict_finite_outputs:
                    params_ok, bad_params = finite_parameter_report(model)
                    if not params_ok:
                        raise FloatingPointError("optimizer.step() produced NaN/Inf parameters: " + "; ".join(bad_params))
                scheduler.step()
                optimizer.zero_grad()

                epoch_losses["total"] += loss.item()
                epoch_losses["ce"] += l_ce.item()
                epoch_losses["mae"] += l_mae.item()
                epoch_losses["mse"] += l_mse.item()
                epoch_losses["huber"] += l_huber.item()
                epoch_losses["consistency"] += l_cons.item()
                epoch_losses["aux"] += l_aux.item()
                epoch_losses["aux_mse"] += l_aux_mse.item()
                num_updates += 1
                sampled_counts += torch.bincount(bins.detach().cpu().long(), minlength=args.num_classes)
                pbar.set_postfix(
                    L=f"{loss.item():.3f}",
                    CE=f"{l_ce.item():.3f}",
                    LIR=f"{l_mae.item():.3f}",
                    CON=f"{l_cons.item():.3f}",
                    Aux=f"{l_aux.item():.3f}",
                )

        for k in epoch_losses:
            epoch_losses[k] /= max(num_updates, 1)

        sampled_counts_global = accelerator.reduce(sampled_counts.to(accelerator.device), reduction="sum").cpu()
        val_metrics = evaluate_lir(
            accelerator.unwrap_model(model),
            val_dl,
            cont_features_enabled=args.use_continuous_features,
            accelerator=accelerator,
            args=args,
            class_weights=class_weights,
        )
        current_monitor = monitor_value(val_metrics, args.monitor_metric)
        sampled_distribution = {BIN_NAMES[i]: int(sampled_counts_global[i].item()) for i in range(args.num_classes)}

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: {epoch_losses['total']:.4f}")
            print(f"  Val Macro F1: {val_metrics.get('macro_F1', 0):.4f}")
            print(f"  Val MAE(LIR): {val_metrics.get('mae', 0):.4f}")
            print(f"  Val Consistency Loss: {val_metrics.get('consistency_loss', 0):.4f}")
            print(f"  Val AUROC: {val_metrics.get('auroc', 0):.4f}")
            print(f"  Monitor ({args.monitor_metric}): {current_monitor:.4f}")
            print(f"  Sampled per epoch: {sampled_distribution}")

        should_stop = stopper.step(current_monitor)

        if accelerator.is_main_process:
            epoch_record = {
                "epoch": epoch + 1,
                "detector_input_mode": detector_input_mode(args),
                "main_regression_target": "lir",
                "stage1_objective": "ce_lir_consistency",
                "train_loss": epoch_losses,
                "val_metrics": val_metrics,
                "monitor_metric": args.monitor_metric,
                "monitor_value": current_monitor,
                "sampled_train_counts": sampled_distribution,
            }
            history.append(epoch_record)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()},
                "model_config": accelerator.unwrap_model(model).get_config(),
                "detector_input_mode": detector_input_mode(args),
                "main_regression_target": "lir",
                "stage1_objective": "ce_lir_consistency",
                "val_metrics": val_metrics,
                "monitor_metric": args.monitor_metric,
                "monitor_value": current_monitor,
                "sampled_train_counts": sampled_distribution,
                "args": vars(args),
            }
            if args.save_epoch_checkpoints:
                safe_torch_save(checkpoint, output_dir / f"checkpoint_epoch{epoch+1}.pt")
            if args.save_last_checkpoint:
                safe_torch_save(checkpoint, output_dir / "last_model.pt")
            if stopper.is_best:
                safe_torch_save(checkpoint, output_dir / "best_model.pt")
                print("  [NEW BEST]")
            with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        if should_stop:
            if accelerator.is_main_process:
                print(f"Early stopping at epoch {epoch+1}")
            break

    best_ckpt_path = output_dir / "best_model.pt"
    has_best_ckpt = torch.tensor(1 if best_ckpt_path.exists() else 0, device=accelerator.device, dtype=torch.int64)
    has_best_ckpt = accelerator.reduce(has_best_ckpt, reduction="max")
    if int(has_best_ckpt.item()) > 0:
        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("  测试集结果")
            print("=" * 60)

        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(best_ckpt["model_state_dict"], strict=False)
        accelerator.wait_for_everyone()

        test_metrics = evaluate_lir(
            accelerator.unwrap_model(model),
            test_dl,
            cont_features_enabled=args.use_continuous_features,
            accelerator=accelerator,
            args=args,
            class_weights=class_weights,
        )

        if accelerator.is_main_process:
            print(f"  Macro F1: {test_metrics.get('macro_F1', 0):.4f}")
            print(f"  MAE(LIR): {test_metrics.get('mae', 0):.4f}")
            print(f"  Consistency Loss: {test_metrics.get('consistency_loss', 0):.4f}")
            print(f"  AUROC: {test_metrics.get('auroc', 0):.4f}")
            results = {
                "best_epoch": int(best_ckpt.get("epoch", -1)) + 1,
                "monitor_metric": args.monitor_metric,
                "detector_input_mode": detector_input_mode(args),
                "experiment": "ratio_stage1",
                "classification_target": "target_ai_ratio_bin",
                "main_regression_target": "lir",
                "main_regression_target_field": "target_lir",
                "stage1_objective": "ce_lir_consistency",
                "inference_requires_continuous_features": bool(args.use_continuous_features),
                "use_continuous_features": args.use_continuous_features,
                "use_aux_targets": args.use_aux_targets,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "aux_target_names": AUX_TARGET_NAMES,
                "aux_loss_weights": {
                    "w_ce": args.w_ce,
                    "w_mae": args.w_mae,
                    "w_mse": args.w_mse,
                    "w_huber": args.w_huber,
                    "w_consistency": args.w_consistency,
                    "w_aux": args.w_aux,
                    "w_aux_mse": args.w_aux_mse,
                    "w_aux_lir": args.w_aux_lir,
                    "w_aux_jaccard": args.w_aux_jaccard,
                    "w_aux_sentence_jaccard": args.w_aux_sentence_jaccard,
                    "w_aux_cosine": args.w_aux_cosine,
                },
                "zero_class_boost": args.zero_class_boost,
                "use_weighted_sampler": args.use_weighted_sampler,
                "use_class_weights": args.use_class_weights,
                "allow_double_balancing": args.allow_double_balancing,
                "data_counts": {
                    "train": count_bins(train_records),
                    "val": count_bins(val_records),
                    "test": count_bins(test_records),
                },
                "training_history_path": str(output_dir / "training_history.json"),
                "val_metrics": best_ckpt.get("val_metrics", {}),
                "test_metrics": test_metrics,
            }
            with open(output_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("\n结果已保存到:", output_dir / "results.json")


if __name__ == "__main__":
    main()
