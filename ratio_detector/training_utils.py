"""Shared training utilities for the RATIO Stage-1 trainer."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ratio_detector.model import MultiHeadDetector
from ratio_detector.utils import BIN_NAMES


def enable_gradient_checkpointing(model: MultiHeadDetector) -> None:
    if hasattr(model.mlm_model, "gradient_checkpointing_enable"):
        model.mlm_model.gradient_checkpointing_enable()
        return
    if hasattr(model.mlm_model, "base_model") and hasattr(model.mlm_model.base_model, "gradient_checkpointing_enable"):
        model.mlm_model.base_model.gradient_checkpointing_enable()


def get_continuous_features(batch: dict[str, torch.Tensor], enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None

    feature_names = ("jaccard", "sentence_jaccard", "cosine", "lir")
    if not all(name in batch for name in feature_names):
        return None

    device = batch["input_ids"].device
    return torch.stack([batch[name] for name in feature_names], dim=1).float().to(device)


def get_aux_targets(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "aux_targets" not in batch:
        return None
    return batch["aux_targets"].float().to(batch["input_ids"].device)


def get_aux_target_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "aux_target_mask" not in batch:
        return None
    return batch["aux_target_mask"].float().to(batch["input_ids"].device)


def aux_loss_weights(args: Any, device: torch.device) -> torch.Tensor:
    values = [
        args.w_aux_lir,
        args.w_aux_jaccard,
        args.w_aux_sentence_jaccard,
        args.w_aux_cosine,
    ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def detector_input_mode(args: Any) -> str:
    return "metadata_inputs" if args.use_continuous_features else "text_only"


def sanitize_model_outputs(
    logits: torch.Tensor,
    reg: torch.Tensor,
    aux_pred: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    reg = torch.nan_to_num(reg.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if aux_pred is not None:
        aux_pred = torch.nan_to_num(aux_pred.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return logits, reg, aux_pred


def assert_finite_model_outputs(
    logits: torch.Tensor,
    reg: torch.Tensor,
    aux_pred: torch.Tensor | None = None,
) -> None:
    tensors = {"logits": logits, "reg": reg}
    if aux_pred is not None:
        tensors["aux"] = aux_pred

    bad_parts = []
    for name, tensor in tensors.items():
        finite = torch.isfinite(tensor)
        if not bool(finite.all().item()):
            bad_parts.append(f"{name}: {int((~finite).sum().item())}/{tensor.numel()} non-finite")

    if bad_parts:
        raise FloatingPointError(
            "Raw model output contains NaN/Inf before loss computation: "
            + "; ".join(bad_parts)
            + ". Try --mixed_precision no with a smaller batch, or lower learning rates."
        )


def finite_gradient_report(
    model: torch.nn.Module,
    max_names: int = 8,
) -> tuple[bool, float, float, list[str]]:
    total_sq = 0.0
    max_abs = 0.0
    bad_names: list[str] = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_f = grad.detach().float()
        finite = torch.isfinite(grad_f)
        if not bool(finite.all().item()) and len(bad_names) < max_names:
            bad = int((~finite).sum().item())
            bad_names.append(f"{name} ({bad}/{grad.numel()})")
        finite_grad = torch.nan_to_num(grad_f, nan=0.0, posinf=0.0, neginf=0.0)
        total_sq += float(torch.sum(finite_grad * finite_grad).item())
        if finite_grad.numel() > 0:
            max_abs = max(max_abs, float(finite_grad.abs().max().item()))

    return len(bad_names) == 0, total_sq ** 0.5, max_abs, bad_names


def finite_parameter_report(
    model: torch.nn.Module,
    max_names: int = 8,
) -> tuple[bool, list[str]]:
    bad_names: list[str] = []
    for name, param in model.named_parameters():
        value = param.detach()
        finite = torch.isfinite(value)
        if not bool(finite.all().item()) and len(bad_names) < max_names:
            bad = int((~finite).sum().item())
            bad_names.append(f"{name} ({bad}/{param.numel()})")
    return len(bad_names) == 0, bad_names


def build_param_groups(model: torch.nn.Module, args: Any) -> list[dict[str, Any]]:
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


def load_jsonl_records(filepath: str) -> list[dict[str, Any]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_bins(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(float(record["target_ai_ratio"]) for record in records)
    return {name: int(counts.get(i / 5, 0)) for i, name in enumerate(BIN_NAMES)}


def monitor_value(metrics: dict[str, Any], monitor_metric: str) -> float:
    if monitor_metric == "macro_f1":
        return float(metrics.get("macro_F1", metrics.get("macro_f1", 0.0)))
    if monitor_metric == "mae":
        return float(metrics.get("mae", float("inf")))
    if monitor_metric == "auroc":
        return float(metrics.get("multi_class_AUROC", metrics.get("macro_auroc", 0.0)))
    if monitor_metric == "composite":
        macro_f1 = float(metrics.get("macro_F1", metrics.get("macro_f1", 0.0)))
        mae = float(metrics.get("mae", 1.0))
        return macro_f1 - mae
    raise ValueError(f"Unknown monitor metric: {monitor_metric}")


def monitor_mode(monitor_metric: str) -> str:
    return "min" if monitor_metric == "mae" else "max"


def safe_torch_save(obj: Any, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        print(f"[WARN] Failed to save checkpoint to {path}: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False
