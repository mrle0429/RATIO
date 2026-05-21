from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def expected_ratio(logits: torch.Tensor) -> torch.Tensor:
    ratio_bins = torch.linspace(0.0, 1.0, steps=logits.size(-1), device=logits.device)
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * ratio_bins).sum(dim=-1)


def ordinal_loss_from_logits(logits: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    ratio_bins = torch.linspace(0.0, 1.0, steps=logits.size(-1), device=logits.device)
    thresholds = ratio_bins[1:]
    cumulative = torch.flip(torch.cumsum(torch.flip(probs, dims=[-1]), dim=-1), dims=[-1])
    cumulative = cumulative[:, 1:].clamp(1e-6, 1.0 - 1e-6)
    targets = (ratios.unsqueeze(1) >= thresholds.unsqueeze(0)).float()
    return F.binary_cross_entropy(cumulative, targets)


def supervised_detector_loss(
    logits: torch.Tensor,
    reg: torch.Tensor,
    bins: torch.Tensor,
    ratios: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.05,
    w_ce: float = 1.0,
    w_mae: float = 2.0,
    w_mse: float = 0.5,
    w_huber: float = 0.5,
    w_consistency: float = 0.2,
    w_cls_ratio: float = 0.0,
    w_ordinal: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(
        logits,
        bins.long(),
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    mae = F.l1_loss(reg, ratios)
    mse = F.mse_loss(reg, ratios)
    huber = F.smooth_l1_loss(reg, ratios)
    cons = F.l1_loss(expected_ratio(logits), reg)
    cls_ratio = F.smooth_l1_loss(expected_ratio(logits), ratios)
    ordinal = ordinal_loss_from_logits(logits, ratios)
    loss = (
        w_ce * ce
        + w_mae * mae
        + w_mse * mse
        + w_huber * huber
        + w_consistency * cons
        + w_cls_ratio * cls_ratio
        + w_ordinal * ordinal
    )
    return loss, {
        "ce": float(ce.detach().item()),
        "mae": float(mae.detach().item()),
        "mse": float(mse.detach().item()),
        "huber": float(huber.detach().item()),
        "consistency": float(cons.detach().item()),
        "cls_ratio": float(cls_ratio.detach().item()),
        "ordinal": float(ordinal.detach().item()),
    }


def label_dpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    bins: torch.Tensor,
    beta: float = 0.2,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    RCDPO-like loss on detector label probabilities.

    The preferred label is the true proportion bin. The rejected label is the
    strongest wrong label under the frozen reference detector. This turns RCDPO's
    pairwise preference idea into a stable classifier-side objective:

        log pi(true) - log pi(wrong_ref) should increase relative to reference.
    """
    policy_logp = F.log_softmax(policy_logits.float(), dim=-1)
    ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
    bins = bins.long()

    true_policy = policy_logp.gather(1, bins.unsqueeze(1)).squeeze(1)
    true_ref = ref_logp.gather(1, bins.unsqueeze(1)).squeeze(1)

    ref_probs = torch.softmax(ref_logits.float(), dim=-1)
    masked_ref_probs = ref_probs.clone()
    masked_ref_probs.scatter_(1, bins.unsqueeze(1), -1.0)
    rejected = masked_ref_probs.argmax(dim=-1)

    wrong_policy = policy_logp.gather(1, rejected.unsqueeze(1)).squeeze(1)
    wrong_ref = ref_logp.gather(1, rejected.unsqueeze(1)).squeeze(1)

    policy_margin = true_policy - wrong_policy
    ref_margin = true_ref - wrong_ref
    logits = beta * (policy_margin - ref_margin - margin)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        acc = (policy_logits.argmax(dim=-1) == bins).float().mean()
        ref_acc = (ref_logits.argmax(dim=-1) == bins).float().mean()
        true_prob = torch.softmax(policy_logits.float(), dim=-1).gather(1, bins.unsqueeze(1)).mean()
        wrong_prob = torch.softmax(policy_logits.float(), dim=-1).gather(1, rejected.unsqueeze(1)).mean()

    return loss, {
        "label_dpo": float(loss.detach().item()),
        "policy_margin": float(policy_margin.detach().mean().item()),
        "ref_margin": float(ref_margin.detach().mean().item()),
        "policy_acc": float(acc.item()),
        "ref_acc": float(ref_acc.item()),
        "true_prob": float(true_prob.item()),
        "wrong_prob": float(wrong_prob.item()),
    }


def clean_rewrite_consistency_loss(
    clean_logits: torch.Tensor,
    clean_reg: torch.Tensor,
    rewrite_logits: torch.Tensor,
    rewrite_reg: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_probs = torch.softmax(clean_logits.detach().float() / temperature, dim=-1)
    rewrite_log_probs = F.log_softmax(rewrite_logits.float() / temperature, dim=-1)
    kl = F.kl_div(rewrite_log_probs, clean_probs, reduction="batchmean") * (temperature ** 2)
    reg = F.l1_loss(rewrite_reg, clean_reg.detach())
    loss = kl + reg
    return loss, {
        "pair_kl": float(kl.detach().item()),
        "pair_reg_l1": float(reg.detach().item()),
    }


def reference_kl_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    ref_probs = torch.softmax(ref_logits.detach().float() / temperature, dim=-1)
    policy_log_probs = F.log_softmax(policy_logits.float() / temperature, dim=-1)
    return F.kl_div(policy_log_probs, ref_probs, reduction="batchmean") * (temperature ** 2)


def merge_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            out[key] = out.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    for key in list(out):
        out[key] /= max(counts[key], 1)
    return out


def format_metrics(metrics: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.4f}")
    return "  ".join(parts)
