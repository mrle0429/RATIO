from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from ratio_detector.model import MultiHeadDetector
from ratio_rcdpo.stage2_dataset import (
    BIN_NAMES,
    CONT_FEATURE_NAMES,
    RewritePairDataset,
    SingleTextDataset,
    class_counts,
    compute_class_weights,
    load_many_jsonl,
    is_attacked_record,
    metadata_feature_summary,
    ratio_to_bin,
    split_records,
)
from ratio_rcdpo.stage2_losses import (
    clean_rewrite_consistency_loss,
    format_metrics,
    label_dpo_loss,
    merge_metric_dicts,
    reference_kl_loss,
    supervised_detector_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "RATIO stage-2 robust detector training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--rewrite_data", nargs="+", default=None,
                   help="One or more rewrite JSONL files. Used when explicit train/val/test files are not supplied.")
    p.add_argument("--train_data", nargs="+", default=None, help="Explicit train JSONL file(s) with rewritten_text.")
    p.add_argument("--val_data", nargs="+", default=None, help="Explicit val JSONL file(s) with rewritten_text.")
    p.add_argument("--test_data", nargs="+", default=None, help="Explicit test JSONL file(s) with rewritten_text.")
    p.add_argument("--stage1_checkpoint", required=True,
                   help="Stage-1 checkpoint from the aligned detector, e.g. outputs_simple_v2_deberta.../best_model.pt.")
    p.add_argument(
        "--model_name",
        default=None,
        help="Backbone model name/path. If omitted, inherit from the Stage-1 checkpoint args.",
    )
    p.add_argument("--output_dir", default="ratio_rcdpo/outputs_stage2")
    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Detector head dropout. If omitted, inherit from the Stage-1 checkpoint args.",
    )
    p.add_argument(
        "--use_continuous_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Stage-1 metadata inputs [jaccard, sentence_jaccard, cosine, lir]. "
             "Default is off to match the current text-only Stage-1 checkpoint.",
    )
    p.add_argument(
        "--rewrite_continuous_source",
        default="mixed",
        choices=["mixed", "rewrite", "zero"],
        help="Continuous features used when the text input is rewritten_text. "
             "'mixed' reuses mixed_* features and avoids leaking rewrite_* metrics; "
             "'rewrite' feeds rewrite_* metrics; 'zero' is an ablation and is usually "
             "out-of-distribution for a metadata-aware checkpoint.",
    )

    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    p.add_argument("--lr_backbone", type=float, default=1e-6)
    p.add_argument("--lr_heads", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--adam_eps", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--mixed_precision", default="no", choices=["bf16", "fp16", "no"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--zero_class_boost", type=float, default=1.0)
    p.add_argument("--use_weighted_sampler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include_human_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--eval_attacked_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report metrics on records that were actually rewritten: "
             "target_ai_ratio > 0 and rewrite_info.status == 'ok'.",
    )
    p.add_argument("--sup_w_ce", type=float, default=1.0)
    p.add_argument("--sup_w_mae", type=float, default=1.0)
    p.add_argument("--sup_w_mse", type=float, default=0.0)
    p.add_argument("--sup_w_huber", type=float, default=0.0)
    p.add_argument("--sup_w_consistency", type=float, default=0.2)
    p.add_argument("--sup_w_cls_ratio", type=float, default=0.0)
    p.add_argument("--sup_w_ordinal", type=float, default=0.0)

    p.add_argument("--w_clean_sup", type=float, default=1.2,
                   help="Supervised loss weight on original mixed_text.")
    p.add_argument("--w_rewrite_sup", type=float, default=1.0,
                   help="Supervised loss weight on rewritten_text.")
    p.add_argument("--w_label_dpo", type=float, default=0.8,
                   help="RCDPO-like label preference loss on rewritten_text.")
    p.add_argument("--w_pair_consistency", type=float, default=0.15,
                   help="Consistency between clean mixed_text and rewritten_text predictions.")
    p.add_argument("--w_clean_ref_kl", type=float, default=0.2,
                   help="KL to frozen stage-1 detector on clean mixed_text.")
    p.add_argument("--dpo_beta", type=float, default=0.5)
    p.add_argument("--dpo_margin", type=float, default=0.0)
    p.add_argument("--kl_temperature", type=float, default=1.0)

    p.add_argument(
        "--rewrite_loss_attacked_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply rewrite supervised/RCDPO losses only to rows where the rewrite attack actually changed the text.",
    )
    p.add_argument(
        "--monitor",
        default="val_attacked_rewrite_active_macro_f1",
        choices=[
            "val_rewrite_composite",
            "val_rewrite_macro_f1",
            "val_rewrite_mae",
            "val_attacked_rewrite_active_macro_f1",
            "val_attacked_rewrite_composite",
        ],
    )
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--save_every_epoch", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fail_on_nonfinite", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def build_param_groups(model: MultiHeadDetector, args: argparse.Namespace) -> list[dict[str, Any]]:
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    backbone_decay = []
    backbone_no_decay = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "cls_" in name or "reg_" in name or "aux_" in name or "cont_" in name or "pool_" in name:
            head_params.append(param)
        elif any(nd in name for nd in no_decay):
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    return [
        {"params": backbone_decay, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": backbone_no_decay, "lr": args.lr_backbone, "weight_decay": 0.0},
        {"params": head_params, "lr": args.lr_heads, "weight_decay": 0.0},
    ]


def enable_gradient_checkpointing(model: MultiHeadDetector) -> None:
    if hasattr(model.mlm_model, "gradient_checkpointing_enable"):
        model.mlm_model.gradient_checkpointing_enable()
        return
    if hasattr(model.mlm_model, "base_model") and hasattr(model.mlm_model.base_model, "gradient_checkpointing_enable"):
        model.mlm_model.base_model.gradient_checkpointing_enable()


def assert_finite_tensors(named_tensors: dict[str, torch.Tensor]) -> None:
    bad_parts = []
    for name, tensor in named_tensors.items():
        finite = torch.isfinite(tensor)
        if not bool(finite.all().item()):
            bad_parts.append(f"{name}: {int((~finite).sum().item())}/{tensor.numel()} non-finite")
    if bad_parts:
        raise FloatingPointError("NaN/Inf detected: " + "; ".join(bad_parts))


def maybe_cont(batch: dict[str, torch.Tensor], key: str, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    value = batch.get(key)
    return value.float() if value is not None else None


def load_records_from_args(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    if args.train_data and args.val_data:
        train = load_many_jsonl(args.train_data)
        val = load_many_jsonl(args.val_data)
        test = load_many_jsonl(args.test_data) if args.test_data else []
        return train, val, test

    if not args.rewrite_data:
        raise ValueError("Provide either --rewrite_data or explicit --train_data/--val_data files.")

    records = load_many_jsonl(args.rewrite_data)
    train, val, test = split_records(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return train, val, test


def eval_continuous_source_for_text(text_field: str, rewrite_source: str) -> str:
    if text_field != "rewritten_text":
        return "self"
    if rewrite_source == "rewrite":
        return "self"
    return rewrite_source


def sampler_for_dataset(dataset: RewritePairDataset, records: list[dict], args: argparse.Namespace):
    if not args.use_weighted_sampler:
        return None
    weights = compute_class_weights(records, zero_class_boost=args.zero_class_boost)
    sample_weights = weights[dataset.bins]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(dataset),
        replacement=True,
    )


@torch.no_grad()
def cache_reference_predictions(
    reference: MultiHeadDetector,
    dataset: RewritePairDataset,
    accelerator: Accelerator,
    batch_size: int,
    num_workers: int,
    use_continuous_features: bool,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    reference.eval()
    device = next(reference.parameters()).device
    clean_logits_list: list[torch.Tensor] = []
    clean_reg_list: list[torch.Tensor] = []
    rewrite_logits_list: list[torch.Tensor] = []
    rewrite_reg_list: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="cache ref", disable=not accelerator.is_main_process):
        clean_ids = batch["clean_input_ids"].to(device)
        clean_mask = batch["clean_attention_mask"].to(device)
        rewrite_ids = batch["rewrite_input_ids"].to(device)
        rewrite_mask = batch["rewrite_attention_mask"].to(device)
        clean_cont = batch["clean_cont_features"].to(device) if use_continuous_features else None
        rewrite_cont = batch["rewrite_cont_features"].to(device) if use_continuous_features else None
        clean_logits, clean_reg = reference(clean_ids, clean_mask, continuous_features=clean_cont)
        rewrite_logits, rewrite_reg = reference(rewrite_ids, rewrite_mask, continuous_features=rewrite_cont)
        clean_logits_list.append(clean_logits.detach().cpu())
        clean_reg_list.append(clean_reg.detach().cpu())
        rewrite_logits_list.append(rewrite_logits.detach().cpu())
        rewrite_reg_list.append(rewrite_reg.detach().cpu())

    dataset.attach_reference_cache(
        torch.cat(clean_logits_list, dim=0),
        torch.cat(clean_reg_list, dim=0),
        torch.cat(rewrite_logits_list, dim=0),
        torch.cat(rewrite_reg_list, dim=0),
    )


@torch.no_grad()
def evaluate(
    model: MultiHeadDetector,
    loader: DataLoader,
    accelerator: Accelerator,
    prefix: str,
    use_continuous_features: bool,
) -> dict[str, Any]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_regs: list[float] = []
    all_targets: list[float] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"eval {prefix}", disable=not accelerator.is_main_process):
        cont = maybe_cont(batch, "cont_features", use_continuous_features)
        logits, reg = model(batch["input_ids"], batch["attention_mask"], continuous_features=cont)
        if torch.isfinite(logits).all() and torch.isfinite(reg).all():
            pass
        else:
            raise FloatingPointError(f"NaN/Inf detected during {prefix} evaluation.")
        probs = torch.softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1)
        gathered = accelerator.gather_for_metrics((
            preds,
            batch["proportion_bin"],
            reg,
            batch["exact_pct"],
            probs,
        ))
        preds_g, labels_g, reg_g, targets_g, probs_g = gathered
        all_preds.extend(preds_g.detach().cpu().tolist())
        all_labels.extend(labels_g.detach().cpu().tolist())
        all_regs.extend(reg_g.detach().cpu().tolist())
        all_targets.extend(targets_g.detach().cpu().tolist())
        all_probs.append(probs_g.detach().cpu().numpy())

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    regs_np = np.asarray(all_regs)
    targets_np = np.asarray(all_targets)
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 6))

    per_f1 = f1_score(labels_np, preds_np, labels=list(range(6)), average=None, zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(
        labels_np, preds_np, labels=list(range(6)), average=None, zero_division=0
    )
    support_np = np.asarray(support)
    active_mask = support_np > 0
    active_macro_f1 = float(per_f1[active_mask].mean()) if active_mask.any() else 0.0
    pred_counts = {name: int((preds_np == i).sum()) for i, name in enumerate(BIN_NAMES)}
    label_counts = {name: int((labels_np == i).sum()) for i, name in enumerate(BIN_NAMES)}
    mean_probs = {
        name: float(probs_np[:, i].mean()) if probs_np.size else 0.0
        for i, name in enumerate(BIN_NAMES)
    }
    metrics: dict[str, Any] = {
        f"{prefix}_macro_f1": float(per_f1.mean()),
        f"{prefix}_active_macro_f1": active_macro_f1,
        f"{prefix}_mae": float(np.abs(regs_np - targets_np).mean()),
        f"{prefix}_mse": float(((regs_np - targets_np) ** 2).mean()),
        f"{prefix}_rmse": float(np.sqrt(((regs_np - targets_np) ** 2).mean())),
        f"{prefix}_per_class_f1": {
            name: float(per_f1[i])
            for i, name in enumerate(BIN_NAMES)
        },
        f"{prefix}_pred_class_counts": pred_counts,
        f"{prefix}_label_class_counts": label_counts,
        f"{prefix}_mean_pred_prob_by_class": mean_probs,
    }
    composite_f1_key = f"{prefix}_active_macro_f1" if "attacked" in prefix else f"{prefix}_macro_f1"
    metrics[f"{prefix}_composite"] = metrics[composite_f1_key] - metrics[f"{prefix}_mae"]

    for i, name in enumerate(BIN_NAMES):
        mask = labels_np == i
        metrics[f"{prefix}_F1@{i * 20}"] = float(per_f1[i])
        metrics[f"{prefix}_precision_{name}"] = float(precision[i])
        metrics[f"{prefix}_recall_{name}"] = float(recall[i])
        metrics[f"{prefix}_support_{name}"] = int(support[i])
        if mask.any():
            metrics[f"{prefix}_mae_{name}"] = float(np.abs(regs_np[mask] - targets_np[mask]).mean())
        else:
            metrics[f"{prefix}_mae_{name}"] = 0.0

    try:
        from sklearn.preprocessing import label_binarize

        labels_bin = label_binarize(labels_np, classes=list(range(6)))
        aurocs = []
        for i, name in enumerate(BIN_NAMES):
            if labels_bin[:, i].sum() > 0 and (labels_bin[:, i] == 0).sum() > 0:
                auroc = float(roc_auc_score(labels_bin[:, i], probs_np[:, i]))
            else:
                auroc = 0.0
            metrics[f"{prefix}_auroc_{name}"] = auroc
            aurocs.append(auroc)
        metrics[f"{prefix}_multi_class_AUROC"] = float(np.mean(aurocs))
    except Exception:
        metrics[f"{prefix}_multi_class_AUROC"] = 0.0

    return metrics


@torch.no_grad()
def evaluate_unprepared(
    model: MultiHeadDetector,
    loader: DataLoader,
    prefix: str,
    device: torch.device,
    use_continuous_features: bool,
) -> dict[str, Any]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_regs: list[float] = []
    all_targets: list[float] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"eval {prefix}"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cont = batch["cont_features"].to(device) if use_continuous_features else None
        logits, reg = model(ids, mask, continuous_features=cont)
        assert_finite_tensors({f"{prefix}_logits": logits, f"{prefix}_reg": reg})
        probs = torch.softmax(logits.float(), dim=-1)
        all_preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        all_labels.extend(batch["proportion_bin"].detach().cpu().tolist())
        all_regs.extend(reg.detach().cpu().tolist())
        all_targets.extend(batch["exact_pct"].detach().cpu().tolist())
        all_probs.append(probs.detach().cpu().numpy())

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    regs_np = np.asarray(all_regs)
    targets_np = np.asarray(all_targets)
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 6))

    per_f1 = f1_score(labels_np, preds_np, labels=list(range(6)), average=None, zero_division=0)
    support_np = np.asarray([(labels_np == i).sum() for i in range(6)])
    active_mask = support_np > 0
    active_macro_f1 = float(per_f1[active_mask].mean()) if active_mask.any() else 0.0
    pred_counts = {name: int((preds_np == i).sum()) for i, name in enumerate(BIN_NAMES)}
    label_counts = {name: int((labels_np == i).sum()) for i, name in enumerate(BIN_NAMES)}
    mean_probs = {
        name: float(probs_np[:, i].mean()) if probs_np.size else 0.0
        for i, name in enumerate(BIN_NAMES)
    }
    metrics: dict[str, Any] = {
        f"{prefix}_macro_f1": float(per_f1.mean()),
        f"{prefix}_active_macro_f1": active_macro_f1,
        f"{prefix}_mae": float(np.abs(regs_np - targets_np).mean()),
        f"{prefix}_mse": float(((regs_np - targets_np) ** 2).mean()),
        f"{prefix}_per_class_f1": {
            name: float(per_f1[i])
            for i, name in enumerate(BIN_NAMES)
        },
        f"{prefix}_pred_class_counts": pred_counts,
        f"{prefix}_label_class_counts": label_counts,
        f"{prefix}_mean_pred_prob_by_class": mean_probs,
    }
    composite_f1_key = f"{prefix}_active_macro_f1" if "attacked" in prefix else f"{prefix}_macro_f1"
    metrics[f"{prefix}_composite"] = metrics[composite_f1_key] - metrics[f"{prefix}_mae"]
    for i, name in enumerate(BIN_NAMES):
        metrics[f"{prefix}_F1@{i * 20}"] = float(per_f1[i])
        mask_i = labels_np == i
        metrics[f"{prefix}_mae_{name}"] = float(np.abs(regs_np[mask_i] - targets_np[mask_i]).mean()) if mask_i.any() else 0.0
    try:
        from sklearn.preprocessing import label_binarize

        labels_bin = label_binarize(labels_np, classes=list(range(6)))
        aurocs = []
        for i, name in enumerate(BIN_NAMES):
            if labels_bin[:, i].sum() > 0 and (labels_bin[:, i] == 0).sum() > 0:
                aurocs.append(float(roc_auc_score(labels_bin[:, i], probs_np[:, i])))
            else:
                aurocs.append(0.0)
        metrics[f"{prefix}_multi_class_AUROC"] = float(np.mean(aurocs))
    except Exception:
        metrics[f"{prefix}_multi_class_AUROC"] = 0.0
    return metrics


def monitor_value(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    if args.monitor == "val_rewrite_composite":
        return float(metrics.get("val_rewrite_composite", -1e9))
    if args.monitor == "val_rewrite_macro_f1":
        return float(metrics.get("val_rewrite_macro_f1", -1e9))
    if args.monitor == "val_rewrite_mae":
        return -float(metrics.get("val_rewrite_mae", 1e9))
    if args.monitor == "val_attacked_rewrite_active_macro_f1":
        return float(metrics.get("val_attacked_rewrite_active_macro_f1", metrics.get("val_rewrite_macro_f1", -1e9)))
    if args.monitor == "val_attacked_rewrite_composite":
        return float(metrics.get("val_attacked_rewrite_composite", metrics.get("val_rewrite_composite", -1e9)))
    raise ValueError(args.monitor)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def checkpoint_args(ckpt: dict[str, Any]) -> dict[str, Any]:
    args = ckpt.get("args")
    return args if isinstance(args, dict) else {}


def checkpoint_model_name(ckpt: dict[str, Any]) -> str | None:
    args = checkpoint_args(ckpt)
    value = args.get("model_name")
    if isinstance(value, str) and value:
        return value
    config = ckpt.get("model_config")
    if isinstance(config, dict):
        value = config.get("model_name")
        if isinstance(value, str) and value:
            return value
    return None


def checkpoint_dropout(ckpt: dict[str, Any]) -> float | None:
    args = checkpoint_args(ckpt)
    value = args.get("dropout")
    if value is None:
        config = ckpt.get("model_config")
        if isinstance(config, dict):
            value = config.get("dropout")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_model_args_from_checkpoint(args: argparse.Namespace, ckpt: dict[str, Any]) -> None:
    if args.model_name is None:
        args.model_name = checkpoint_model_name(ckpt) or "microsoft/deberta-v3-large"
    if args.dropout is None:
        args.dropout = checkpoint_dropout(ckpt)
    if args.dropout is None:
        args.dropout = 0.1


def checkpoint_feature_mode(ckpt: dict[str, Any]) -> bool | None:
    args = checkpoint_args(ckpt)
    if "use_continuous_features" in args:
        return bool(args["use_continuous_features"])
    return None


def assert_checkpoint_compatible(ckpt: dict[str, Any], use_continuous_features: bool) -> None:
    ckpt_use_cont = checkpoint_feature_mode(ckpt)
    if ckpt_use_cont is None or ckpt_use_cont == use_continuous_features:
        return
    raise ValueError(
        "Stage-1 checkpoint feature mode does not match Stage-2. "
        f"checkpoint use_continuous_features={ckpt_use_cont}, "
        f"stage2 use_continuous_features={use_continuous_features}. "
        "Use a metadata-aware Stage-1 checkpoint with --use_continuous_features, or "
        "use a text-only Stage-1 checkpoint with --no-use_continuous_features."
    )


def robustness_summary(reference: dict[str, Any], stage2: dict[str, Any], split: str) -> dict[str, float]:
    def _macro_key(prefix: str) -> str:
        return f"{prefix}_active_macro_f1" if "attacked" in prefix else f"{prefix}_macro_f1"

    clean_prefix = f"reference_{split}_clean"
    rewrite_prefix = f"reference_{split}_rewrite"
    stage2_prefix = f"{split}_rewrite"

    clean_f1 = float(reference.get(_macro_key(clean_prefix), 0.0))
    rewrite_f1 = float(reference.get(_macro_key(rewrite_prefix), 0.0))
    stage2_rewrite_f1 = float(stage2.get(_macro_key(stage2_prefix), 0.0))
    clean_mae = float(reference.get(f"reference_{split}_clean_mae", 0.0))
    rewrite_mae = float(reference.get(f"reference_{split}_rewrite_mae", 0.0))
    stage2_rewrite_mae = float(stage2.get(f"{split}_rewrite_mae", 0.0))
    clean_mse = float(reference.get(f"reference_{split}_clean_mse", 0.0))
    rewrite_mse = float(reference.get(f"reference_{split}_rewrite_mse", 0.0))
    stage2_rewrite_mse = float(stage2.get(f"{split}_rewrite_mse", 0.0))
    return {
        f"{split}_attack_f1_drop": clean_f1 - rewrite_f1,
        f"{split}_dpo_f1_recovery": stage2_rewrite_f1 - rewrite_f1,
        f"{split}_attack_mae_increase": rewrite_mae - clean_mae,
        f"{split}_dpo_mae_reduction": rewrite_mae - stage2_rewrite_mae,
        f"{split}_attack_mse_increase": rewrite_mse - clean_mse,
        f"{split}_dpo_mse_reduction": rewrite_mse - stage2_rewrite_mse,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    resolve_model_args_from_checkpoint(args, ckpt)
    assert_checkpoint_compatible(ckpt, args.use_continuous_features)
    state = ckpt.get("model_state_dict", ckpt)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 72)
        print("RATIO Stage 2: robust detector training on humanized rewrites")
        print("=" * 72)
        print(f"stage1_checkpoint: {args.stage1_checkpoint}")
        print(f"model_name:        {args.model_name}")
        print(f"output_dir:        {output_dir}")
        print(f"batch/accum:       {args.batch_size}/{args.grad_accum_steps}")
        print(f"continuous inputs: {args.use_continuous_features} ({CONT_FEATURE_NAMES})")
        print(f"rewrite cont src:  {args.rewrite_continuous_source}")
        print(f"rewrite loss on attacked only: {args.rewrite_loss_attacked_only}")
        print(f"attacked eval:     {args.eval_attacked_only}")
        print(f"loss weights:      clean={args.w_clean_sup}, rewrite={args.w_rewrite_sup}, "
              f"dpo={args.w_label_dpo}, pair={args.w_pair_consistency}, ref_kl={args.w_clean_ref_kl}")
        print("=" * 72 + "\n")

    train_records, val_records, test_records = load_records_from_args(args)
    train_records = [r for r in train_records if "rewritten_text" in r and "mixed_text" in r]
    val_records = [r for r in val_records if "rewritten_text" in r and "mixed_text" in r]
    test_records = [r for r in test_records if "rewritten_text" in r and "mixed_text" in r]
    if not train_records or not val_records:
        raise ValueError("Need non-empty train and val rewrite records.")

    val_attacked_records = [r for r in val_records if is_attacked_record(r)]
    test_attacked_records = [r for r in test_records if is_attacked_record(r)]

    if accelerator.is_main_process:
        data_summary = {
            "train": {"n": len(train_records), "bins": class_counts(train_records)},
            "val": {"n": len(val_records), "bins": class_counts(val_records)},
            "test": {"n": len(test_records), "bins": class_counts(test_records)} if test_records else None,
            "val_attacked": {"n": len(val_attacked_records), "bins": class_counts(val_attacked_records)},
            "test_attacked": {"n": len(test_attacked_records), "bins": class_counts(test_attacked_records)}
            if test_attacked_records else None,
            "continuous_feature_order": CONT_FEATURE_NAMES,
            "rewrite_continuous_source": args.rewrite_continuous_source,
            "metadata_summary": {
                "train": metadata_feature_summary(train_records),
                "val": metadata_feature_summary(val_records),
                "test": metadata_feature_summary(test_records) if test_records else None,
            },
            "args": vars(args),
        }
        save_json(output_dir / "data_summary.json", data_summary)
        print("Data distribution:")
        print(json.dumps(data_summary, ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = RewritePairDataset(
        train_records,
        tokenizer,
        max_length=args.max_seq_len,
        include_human_only=args.include_human_only,
        use_continuous_features=args.use_continuous_features,
        rewrite_continuous_source=args.rewrite_continuous_source,
    )
    sampler = sampler_for_dataset(train_ds, train_records, args)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_clean_ds = SingleTextDataset(
        val_records,
        tokenizer,
        "mixed_text",
        max_length=args.max_seq_len,
        use_continuous_features=args.use_continuous_features,
        continuous_source=eval_continuous_source_for_text("mixed_text", args.rewrite_continuous_source),
    )
    val_rewrite_ds = SingleTextDataset(
        val_records,
        tokenizer,
        "rewritten_text",
        max_length=args.max_seq_len,
        use_continuous_features=args.use_continuous_features,
        continuous_source=eval_continuous_source_for_text("rewritten_text", args.rewrite_continuous_source),
    )
    val_clean_dl = DataLoader(val_clean_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)
    val_rewrite_dl = DataLoader(val_rewrite_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)

    test_clean_dl = None
    test_rewrite_dl = None
    if test_records:
        test_clean_ds = SingleTextDataset(
            test_records,
            tokenizer,
            "mixed_text",
            max_length=args.max_seq_len,
            use_continuous_features=args.use_continuous_features,
            continuous_source=eval_continuous_source_for_text("mixed_text", args.rewrite_continuous_source),
        )
        test_rewrite_ds = SingleTextDataset(
            test_records,
            tokenizer,
            "rewritten_text",
            max_length=args.max_seq_len,
            use_continuous_features=args.use_continuous_features,
            continuous_source=eval_continuous_source_for_text("rewritten_text", args.rewrite_continuous_source),
        )
        test_clean_dl = DataLoader(test_clean_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)
        test_rewrite_dl = DataLoader(test_rewrite_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)

    val_attacked_clean_dl = None
    val_attacked_rewrite_dl = None
    if args.eval_attacked_only and val_attacked_records:
        val_attacked_clean_ds = SingleTextDataset(
            val_attacked_records,
            tokenizer,
            "mixed_text",
            max_length=args.max_seq_len,
            use_continuous_features=args.use_continuous_features,
            continuous_source=eval_continuous_source_for_text("mixed_text", args.rewrite_continuous_source),
        )
        val_attacked_rewrite_ds = SingleTextDataset(
            val_attacked_records,
            tokenizer,
            "rewritten_text",
            max_length=args.max_seq_len,
            use_continuous_features=args.use_continuous_features,
            continuous_source=eval_continuous_source_for_text("rewritten_text", args.rewrite_continuous_source),
        )
        val_attacked_clean_dl = DataLoader(
            val_attacked_clean_ds,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_attacked_rewrite_dl = DataLoader(
            val_attacked_rewrite_ds,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    test_attacked_clean_dl = None
    test_attacked_rewrite_dl = None
    if args.eval_attacked_only and test_attacked_records:
        test_attacked_clean_ds = SingleTextDataset(
            test_attacked_records,
            tokenizer,
            "mixed_text",
            max_length=args.max_seq_len,
            use_continuous_features=args.use_continuous_features,
            continuous_source=eval_continuous_source_for_text("mixed_text", args.rewrite_continuous_source),
        )
        test_attacked_rewrite_ds = SingleTextDataset(
            test_attacked_records,
            tokenizer,
            "rewritten_text",
            max_length=args.max_seq_len,
            use_continuous_features=args.use_continuous_features,
            continuous_source=eval_continuous_source_for_text("rewritten_text", args.rewrite_continuous_source),
        )
        test_attacked_clean_dl = DataLoader(
            test_attacked_clean_ds,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_attacked_rewrite_dl = DataLoader(
            test_attacked_rewrite_ds,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # First run the frozen Stage-1 detector once and cache its predictions.
    # This avoids keeping both the reference and policy DeBERTa models on GPU
    # during the actual training stage.
    reference = MultiHeadDetector(
        model_name=args.model_name,
        dropout=args.dropout,
        use_continuous_features=args.use_continuous_features,
    )
    reference.load_state_dict(state, strict=False)
    reference.set_special_token_ids(tokenizer)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False
    reference = accelerator.prepare(reference)

    if accelerator.is_main_process:
        print("Caching reference predictions for the rewrite training set...")
    reference_baseline: dict[str, Any] = {}
    reference_device = next(accelerator.unwrap_model(reference).parameters()).device
    if accelerator.is_main_process:
        print("Evaluating frozen Stage-1 reference before Stage-2 updates...")
        reference_baseline.update(evaluate_unprepared(
            accelerator.unwrap_model(reference),
            val_clean_dl,
            "reference_val_clean",
            reference_device,
            args.use_continuous_features,
        ))
        reference_baseline.update(evaluate_unprepared(
            accelerator.unwrap_model(reference),
            val_rewrite_dl,
            "reference_val_rewrite",
            reference_device,
            args.use_continuous_features,
        ))
        if test_clean_dl is not None and test_rewrite_dl is not None:
            reference_baseline.update(evaluate_unprepared(
                accelerator.unwrap_model(reference),
                test_clean_dl,
                "reference_test_clean",
                reference_device,
                args.use_continuous_features,
            ))
            reference_baseline.update(evaluate_unprepared(
                accelerator.unwrap_model(reference),
                test_rewrite_dl,
                "reference_test_rewrite",
                reference_device,
                args.use_continuous_features,
            ))
        if val_attacked_clean_dl is not None and val_attacked_rewrite_dl is not None:
            reference_baseline.update(evaluate_unprepared(
                accelerator.unwrap_model(reference),
                val_attacked_clean_dl,
                "reference_val_attacked_clean",
                reference_device,
                args.use_continuous_features,
            ))
            reference_baseline.update(evaluate_unprepared(
                accelerator.unwrap_model(reference),
                val_attacked_rewrite_dl,
                "reference_val_attacked_rewrite",
                reference_device,
                args.use_continuous_features,
            ))
        if test_attacked_clean_dl is not None and test_attacked_rewrite_dl is not None:
            reference_baseline.update(evaluate_unprepared(
                accelerator.unwrap_model(reference),
                test_attacked_clean_dl,
                "reference_test_attacked_clean",
                reference_device,
                args.use_continuous_features,
            ))
            reference_baseline.update(evaluate_unprepared(
                accelerator.unwrap_model(reference),
                test_attacked_rewrite_dl,
                "reference_test_attacked_rewrite",
                reference_device,
                args.use_continuous_features,
            ))
        save_json(output_dir / "reference_baseline.json", reference_baseline)
    accelerator.wait_for_everyone()
    cache_reference_predictions(
        accelerator.unwrap_model(reference),
        train_ds,
        accelerator,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        use_continuous_features=args.use_continuous_features,
    )
    accelerator.wait_for_everyone()
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    policy = MultiHeadDetector(
        model_name=args.model_name,
        dropout=args.dropout,
        use_continuous_features=args.use_continuous_features,
    )
    policy.load_state_dict(state, strict=False)
    policy.set_special_token_ids(tokenizer)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(policy)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(
            train_records,
            zero_class_boost=args.zero_class_boost,
        )

    optimizer = torch.optim.AdamW(
        build_param_groups(policy, args),
        eps=args.adam_eps,
        foreach=False,
        fused=False,
    )
    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    prepared = accelerator.prepare(
        policy,
        optimizer,
        train_dl,
        val_clean_dl,
        val_rewrite_dl,
        scheduler,
    )
    policy, optimizer, train_dl, val_clean_dl, val_rewrite_dl, scheduler = prepared
    if test_clean_dl is not None and test_rewrite_dl is not None:
        test_clean_dl, test_rewrite_dl = accelerator.prepare(test_clean_dl, test_rewrite_dl)
    if val_attacked_clean_dl is not None and val_attacked_rewrite_dl is not None:
        val_attacked_clean_dl, val_attacked_rewrite_dl = accelerator.prepare(
            val_attacked_clean_dl,
            val_attacked_rewrite_dl,
        )
    if test_attacked_clean_dl is not None and test_attacked_rewrite_dl is not None:
        test_attacked_clean_dl, test_attacked_rewrite_dl = accelerator.prepare(
            test_attacked_clean_dl,
            test_attacked_rewrite_dl,
        )

    if class_weights is not None:
        class_weights = class_weights.to(accelerator.device)

    best_monitor = -1e18
    best_metrics: dict[str, Any] = {}
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_items: list[dict[str, float]] = []
        pbar = tqdm(train_dl, desc=f"stage2 epoch {epoch}/{args.epochs}", disable=not accelerator.is_main_process)

        for batch in pbar:
            with accelerator.accumulate(policy):
                bins = batch["proportion_bin"]
                ratios = batch["exact_pct"]
                clean_cont = maybe_cont(batch, "clean_cont_features", args.use_continuous_features)
                rewrite_cont = maybe_cont(batch, "rewrite_cont_features", args.use_continuous_features)

                clean_logits, clean_reg = policy(
                    batch["clean_input_ids"],
                    batch["clean_attention_mask"],
                    continuous_features=clean_cont,
                )
                rewrite_logits, rewrite_reg = policy(
                    batch["rewrite_input_ids"],
                    batch["rewrite_attention_mask"],
                    continuous_features=rewrite_cont,
                )
                if args.fail_on_nonfinite:
                    assert_finite_tensors({
                        "clean_logits": clean_logits,
                        "clean_reg": clean_reg,
                        "rewrite_logits": rewrite_logits,
                        "rewrite_reg": rewrite_reg,
                    })

                ref_clean_logits = batch["ref_clean_logits"].to(accelerator.device)
                ref_rewrite_logits = batch["ref_rewrite_logits"].to(accelerator.device)
                if args.rewrite_loss_attacked_only:
                    rewrite_loss_mask = batch.get("is_attacked")
                    if rewrite_loss_mask is None:
                        rewrite_loss_mask = torch.ones_like(bins, dtype=torch.bool)
                    else:
                        rewrite_loss_mask = rewrite_loss_mask.bool()
                else:
                    rewrite_loss_mask = torch.ones_like(bins, dtype=torch.bool)
                rewrite_loss_count = int(rewrite_loss_mask.detach().sum().item())

                clean_sup, clean_sup_m = supervised_detector_loss(
                    clean_logits,
                    clean_reg,
                    bins,
                    ratios,
                    class_weights=class_weights,
                    label_smoothing=args.label_smoothing,
                    w_ce=args.sup_w_ce,
                    w_mae=args.sup_w_mae,
                    w_mse=args.sup_w_mse,
                    w_huber=args.sup_w_huber,
                    w_consistency=args.sup_w_consistency,
                    w_cls_ratio=args.sup_w_cls_ratio,
                    w_ordinal=args.sup_w_ordinal,
                )
                if rewrite_loss_count > 0:
                    rewrite_sup, rewrite_sup_m = supervised_detector_loss(
                        rewrite_logits[rewrite_loss_mask],
                        rewrite_reg[rewrite_loss_mask],
                        bins[rewrite_loss_mask],
                        ratios[rewrite_loss_mask],
                        class_weights=class_weights,
                        label_smoothing=args.label_smoothing,
                        w_ce=args.sup_w_ce,
                        w_mae=args.sup_w_mae,
                        w_mse=args.sup_w_mse,
                        w_huber=args.sup_w_huber,
                        w_consistency=args.sup_w_consistency,
                        w_cls_ratio=args.sup_w_cls_ratio,
                        w_ordinal=args.sup_w_ordinal,
                    )
                    dpo_loss, dpo_m = label_dpo_loss(
                        rewrite_logits[rewrite_loss_mask],
                        ref_rewrite_logits[rewrite_loss_mask],
                        bins[rewrite_loss_mask],
                        beta=args.dpo_beta,
                        margin=args.dpo_margin,
                    )
                else:
                    rewrite_sup = rewrite_logits.sum() * 0.0
                    rewrite_sup_m = {
                        "ce": 0.0,
                        "mae": 0.0,
                        "mse": 0.0,
                        "huber": 0.0,
                        "consistency": 0.0,
                        "cls_ratio": 0.0,
                        "ordinal": 0.0,
                    }
                    dpo_loss = rewrite_logits.sum() * 0.0
                    dpo_m = {
                        "label_dpo": 0.0,
                        "policy_margin": 0.0,
                        "ref_margin": 0.0,
                        "policy_acc": 0.0,
                        "ref_acc": 0.0,
                        "true_prob": 0.0,
                        "wrong_prob": 0.0,
                    }
                pair_loss, pair_m = clean_rewrite_consistency_loss(
                    clean_logits,
                    clean_reg,
                    rewrite_logits,
                    rewrite_reg,
                    temperature=args.kl_temperature,
                )
                clean_kl = reference_kl_loss(clean_logits, ref_clean_logits, temperature=args.kl_temperature)

                loss = (
                    args.w_clean_sup * clean_sup
                    + args.w_rewrite_sup * rewrite_sup
                    + args.w_label_dpo * dpo_loss
                    + args.w_pair_consistency * pair_loss
                    + args.w_clean_ref_kl * clean_kl
                )
                if args.fail_on_nonfinite and not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError("Stage-2 loss became NaN/Inf before backward().")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                metrics = {
                    "loss": float(loss.detach().item()),
                    "clean_sup": float(clean_sup.detach().item()),
                    "rewrite_sup": float(rewrite_sup.detach().item()),
                    "clean_ref_kl": float(clean_kl.detach().item()),
                    "rewrite_loss_attacked_frac": float(rewrite_loss_count / max(int(bins.numel()), 1)),
                    **{f"clean_{k}": v for k, v in clean_sup_m.items()},
                    **{f"rewrite_{k}": v for k, v in rewrite_sup_m.items()},
                    **dpo_m,
                    **pair_m,
                }
                epoch_items.append(metrics)
                pbar.set_postfix(
                    L=f"{metrics['loss']:.3f}",
                    RMAE=f"{metrics['rewrite_mae']:.3f}",
                    RCDPO=f"{metrics['label_dpo']:.3f}",
                    ACC=f"{metrics['policy_acc']:.3f}",
                )

        train_metrics = merge_metric_dicts(epoch_items)
        val_metrics: dict[str, Any] = {}
        val_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            val_clean_dl,
            accelerator,
            "val_clean",
            args.use_continuous_features,
        ))
        val_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            val_rewrite_dl,
            accelerator,
            "val_rewrite",
            args.use_continuous_features,
        ))
        if val_attacked_clean_dl is not None and val_attacked_rewrite_dl is not None:
            val_metrics.update(evaluate(
                accelerator.unwrap_model(policy),
                val_attacked_clean_dl,
                accelerator,
                "val_attacked_clean",
                args.use_continuous_features,
            ))
            val_metrics.update(evaluate(
                accelerator.unwrap_model(policy),
                val_attacked_rewrite_dl,
                accelerator,
                "val_attacked_rewrite",
                args.use_continuous_features,
            ))
        current = monitor_value(val_metrics, args)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "monitor": current,
        }
        history.append(epoch_record)

        if accelerator.is_main_process:
            print(
                f"\nEpoch {epoch}: "
                + format_metrics(train_metrics, ["loss", "rewrite_mae", "label_dpo", "policy_acc", "ref_acc"])
            )
            print(
                "Validation: "
                + format_metrics(val_metrics, [
                    "val_clean_macro_f1",
                    "val_clean_mae",
                    "val_rewrite_macro_f1",
                    "val_rewrite_mae",
                    "val_attacked_rewrite_active_macro_f1",
                    "val_rewrite_composite",
                ])
            )
            print(f"Val rewrite predicted counts: {val_metrics.get('val_rewrite_pred_class_counts', {})}")
            print(f"Val rewrite per-class F1: {val_metrics.get('val_rewrite_per_class_f1', {})}")
            if "val_attacked_rewrite_active_macro_f1" in val_metrics:
                print(f"Val attacked rewrite active F1: {val_metrics.get('val_attacked_rewrite_active_macro_f1', 0):.4f}")
            save_json(output_dir / "training_history.json", {"history": history})

        improved = current > best_monitor
        if improved:
            best_monitor = current
            best_metrics = val_metrics
            best_epoch = epoch
            patience = 0
            if accelerator.is_main_process:
                raw_policy = accelerator.unwrap_model(policy)
                torch.save(
                    {
                        "model_state_dict": raw_policy.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "train_metrics": train_metrics,
                        "monitor": args.monitor,
                        "monitor_value": current,
                        "stage1_checkpoint": args.stage1_checkpoint,
                        "args": vars(args),
                    },
                    output_dir / "best_model.pt",
                )
                print(f"New best model saved at epoch {epoch}.")
        else:
            patience += 1

        if args.save_every_epoch and accelerator.is_main_process:
            raw_policy = accelerator.unwrap_model(policy)
            torch.save(
                {
                    "model_state_dict": raw_policy.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "monitor": args.monitor,
                    "monitor_value": current,
                    "stage1_checkpoint": args.stage1_checkpoint,
                    "args": vars(args),
                },
                output_dir / f"checkpoint_epoch{epoch}.pt",
            )

        if patience >= args.early_stop_patience:
            if accelerator.is_main_process:
                print(f"Early stopping at epoch {epoch}.")
            break

    if accelerator.is_main_process:
        print(f"\nBest epoch: {best_epoch}, monitor={best_monitor:.6f}")

    if test_clean_dl is not None and test_rewrite_dl is not None:
        best_path = output_dir / "best_model.pt"
        if best_path.exists():
            raw_policy = accelerator.unwrap_model(policy)
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            raw_policy.load_state_dict(best_ckpt["model_state_dict"], strict=False)

        test_metrics: dict[str, Any] = {}
        test_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            test_clean_dl,
            accelerator,
            "test_clean",
            args.use_continuous_features,
        ))
        test_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            test_rewrite_dl,
            accelerator,
            "test_rewrite",
            args.use_continuous_features,
        ))
    else:
        test_metrics = {}

    attacked_metrics: dict[str, Any] = {}
    if val_attacked_clean_dl is not None and val_attacked_rewrite_dl is not None:
        attacked_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            val_attacked_clean_dl,
            accelerator,
            "val_attacked_clean",
            args.use_continuous_features,
        ))
        attacked_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            val_attacked_rewrite_dl,
            accelerator,
            "val_attacked_rewrite",
            args.use_continuous_features,
        ))
    if test_attacked_clean_dl is not None and test_attacked_rewrite_dl is not None:
        attacked_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            test_attacked_clean_dl,
            accelerator,
            "test_attacked_clean",
            args.use_continuous_features,
        ))
        attacked_metrics.update(evaluate(
            accelerator.unwrap_model(policy),
            test_attacked_rewrite_dl,
            accelerator,
            "test_attacked_rewrite",
            args.use_continuous_features,
        ))

    if accelerator.is_main_process:
        summary: dict[str, float] = {}
        summary.update(robustness_summary(reference_baseline, best_metrics, "val"))
        if test_metrics:
            summary.update(robustness_summary(reference_baseline, test_metrics, "test"))
        if attacked_metrics:
            final_metrics_for_summary = {**best_metrics, **test_metrics, **attacked_metrics}
            if "val_attacked_rewrite_macro_f1" in attacked_metrics:
                summary.update(robustness_summary(reference_baseline, final_metrics_for_summary, "val_attacked"))
            if "test_attacked_rewrite_macro_f1" in attacked_metrics:
                summary.update(robustness_summary(reference_baseline, final_metrics_for_summary, "test_attacked"))
        results = {
            "best_epoch": best_epoch,
            "best_monitor": best_monitor,
            "reference_baseline": reference_baseline,
            "best_val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "attacked_metrics": attacked_metrics,
            "robustness_summary": summary,
            "data_counts": {
                "train": class_counts(train_records),
                "val": class_counts(val_records),
                "test": class_counts(test_records) if test_records else None,
                "val_attacked": class_counts(val_attacked_records),
                "test_attacked": class_counts(test_attacked_records) if test_attacked_records else None,
            },
            "stage1_checkpoint": args.stage1_checkpoint,
            "args": vars(args),
        }
        save_json(output_dir / "results.json", results)
        print(f"Results saved to {output_dir / 'results.json'}")
        print(f"Best checkpoint saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
