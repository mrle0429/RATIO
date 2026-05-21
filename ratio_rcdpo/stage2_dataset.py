from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ratio_detector.utils import continuous_targets_from_record


RATIO_TO_BIN: dict[float, int] = {
    0.0: 0,
    0.2: 1,
    0.4: 2,
    0.6: 3,
    0.8: 4,
    1.0: 5,
}

BIN_TO_RATIO = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float32)
BIN_NAMES = ["0pct", "20pct", "40pct", "60pct", "80pct", "100pct"]
CONT_FEATURE_NAMES = ["jaccard", "sentence_jaccard", "cosine", "lir"]
TEXT_FIELD_TO_META_PREFIX = {
    "mixed_text": "mixed",
    "rewritten_text": "rewrite",
}
PREFIXED_FEATURE_KEYS = {
    "mixed": {
        "jaccard": "mixed_jaccard_distance",
        "sentence_jaccard": "mixed_sentence_jaccard",
        "cosine": "mixed_cosine_distance",
        "lir": "mixed_lir",
    },
    "rewrite": {
        "jaccard": "rewrite_jaccard_distance",
        "sentence_jaccard": "rewrite_sentence_jaccard",
        "cosine": "rewrite_cosine_distance",
        "lir": "rewrite_lir",
    },
}


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _prefixed_feature_values(rec: dict[str, Any], prefix: str) -> list[float] | None:
    keys = PREFIXED_FEATURE_KEYS[prefix]
    values: list[float] = []
    for name in CONT_FEATURE_NAMES:
        key = keys[name]
        if key not in rec or rec.get(key) is None:
            return None
        try:
            value = float(rec[key])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        values.append(_clip01(value))
    return values


def ratio_to_bin(value: Any) -> int:
    ratio = float(value)
    rounded = round(ratio, 1)
    if rounded in RATIO_TO_BIN:
        return RATIO_TO_BIN[rounded]
    return int(np.clip(round(ratio * 5), 0, 5))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            rec.setdefault("_source_file", path.name)
            records.append(rec)
    return records


def load_many_jsonl(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return records


def has_rewrite_pair(rec: dict[str, Any]) -> bool:
    mixed = str(rec.get("mixed_text", "") or "").strip()
    rewritten = str(rec.get("rewritten_text", "") or "").strip()
    return bool(mixed and rewritten and "target_ai_ratio" in rec)


def is_attacked_record(rec: dict[str, Any]) -> bool:
    """
    True for rows where the rewrite attack actually changed AI-origin text.
    Human-only rows or unchanged rewrites return False.
    """
    try:
        ratio = float(rec.get("target_ai_ratio", 0.0))
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio <= 0.0:
        return False

    info = rec.get("rewrite_info")
    status = info.get("status") if isinstance(info, dict) else None
    if status is not None and status != "ok":
        return False

    mixed = str(rec.get("mixed_text", "") or "").strip()
    rewritten = str(rec.get("rewritten_text", "") or "").strip()
    if not mixed or not rewritten:
        return False
    if mixed == rewritten:
        return False
    return True


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified split by target_ai_ratio without requiring sklearn."""
    rng = np.random.default_rng(seed)
    by_bin: dict[int, list[dict[str, Any]]] = {i: [] for i in range(6)}
    for rec in records:
        by_bin[ratio_to_bin(rec["target_ai_ratio"])].append(rec)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for bin_id, group in by_bin.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        n_test = min(n_test, n)
        n_val = min(n_val, n - n_test)

        test.extend(group[:n_test])
        val.extend(group[n_test:n_test + n_val])
        train.extend(group[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in BIN_NAMES}
    for rec in records:
        counts[BIN_NAMES[ratio_to_bin(rec["target_ai_ratio"])]] += 1
    return counts


def metadata_feature_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize availability and scale of Stage-2 continuous metadata."""
    summary: dict[str, Any] = {"n": len(records)}
    for prefix in ("mixed", "rewrite"):
        rows: list[list[float]] = []
        for rec in records:
            values = _prefixed_feature_values(rec, prefix)
            if values is not None:
                rows.append(values)

        summary[f"{prefix}_meta_rows"] = len(rows)
        summary[f"{prefix}_meta_coverage"] = len(rows) / max(len(records), 1)
        if rows:
            arr = np.asarray(rows, dtype=np.float32)
            summary[f"{prefix}_feature_means"] = {
                name: float(arr[:, idx].mean())
                for idx, name in enumerate(CONT_FEATURE_NAMES)
            }
            summary[f"{prefix}_feature_mins"] = {
                name: float(arr[:, idx].min())
                for idx, name in enumerate(CONT_FEATURE_NAMES)
            }
            summary[f"{prefix}_feature_maxs"] = {
                name: float(arr[:, idx].max())
                for idx, name in enumerate(CONT_FEATURE_NAMES)
            }
        else:
            summary[f"{prefix}_feature_means"] = None
            summary[f"{prefix}_feature_mins"] = None
            summary[f"{prefix}_feature_maxs"] = None
    return summary


def compute_class_weights(
    records: list[dict[str, Any]],
    zero_class_boost: float = 1.0,
    clip_min: float = 0.5,
    clip_max: float = 6.0,
) -> torch.Tensor:
    labels = [ratio_to_bin(rec["target_ai_ratio"]) for rec in records]
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=6)
    counts = np.maximum(counts, 1)
    balanced = counts.sum() / 6.0
    weights = np.sqrt(balanced / counts)
    weights[0] *= zero_class_boost
    weights = np.clip(weights, clip_min, clip_max)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def continuous_features_for_text(
    rec: dict[str, Any],
    text_field: str,
    prefer_stored_clean_fields: bool = True,
) -> list[float]:
    """
    Return detector continuous inputs in the Stage-1 order:
    [jaccard, sentence_jaccard, cosine, lir].

    Preferred v2 rewrite_grouped fields:
      - mixed_jaccard_distance / mixed_sentence_jaccard / mixed_cosine_distance / mixed_lir
      - rewrite_jaccard_distance / rewrite_sentence_jaccard / rewrite_cosine_distance / rewrite_lir

    If these fields are missing, the function falls back to Stage-1 field names
    for mixed_text, then finally reconstructs metrics from original_text and the
    selected text field.
    """
    prefix = TEXT_FIELD_TO_META_PREFIX.get(text_field)
    if prefix is not None:
        values = _prefixed_feature_values(rec, prefix)
        if values is not None:
            return values

    tmp = dict(rec)
    tmp["mixed_text"] = str(rec.get(text_field, "") or "")

    if text_field != "mixed_text" or not prefer_stored_clean_fields:
        for key in ("lir", "jaccard_distance", "sentence_jaccard", "cosine_distance"):
            tmp.pop(key, None)

    targets, _ = continuous_targets_from_record(tmp)
    return [
        _clip01(float(targets["jaccard"])),
        _clip01(float(targets["sentence_jaccard"])),
        _clip01(float(targets["cosine"])),
        _clip01(float(targets["lir"])),
    ]


class RewritePairDataset(Dataset):
    """
    Returns clean mixed_text and adversarial rewritten_text for the same record.

    clean_text: the original stage-1 mixed_text.
    rewrite_text: the humanized rewrite where AI-labelled sentences were rewritten.
    target_ai_ratio remains the original proportion label.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        include_human_only: bool = True,
        use_continuous_features: bool = False,
        rewrite_continuous_source: str = "mixed",
    ) -> None:
        if rewrite_continuous_source not in {"mixed", "rewrite", "zero"}:
            raise ValueError(
                "rewrite_continuous_source must be one of: mixed, rewrite, zero"
            )
        filtered = [r for r in records if has_rewrite_pair(r)]
        if not include_human_only:
            filtered = [r for r in filtered if float(r["target_ai_ratio"]) > 0.0]
        if not filtered:
            raise ValueError("No valid records with mixed_text, rewritten_text, and target_ai_ratio.")

        self.records = filtered
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_attacked = torch.tensor([1 if is_attacked_record(r) else 0 for r in filtered], dtype=torch.long)

        clean_texts = [str(r["mixed_text"]) for r in filtered]
        rewrite_texts = [str(r["rewritten_text"]) for r in filtered]
        ratios = [float(r["target_ai_ratio"]) for r in filtered]
        bins = [ratio_to_bin(r) for r in ratios]

        clean_enc = tokenizer(
            clean_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rewrite_enc = tokenizer(
            rewrite_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.clean_input_ids = clean_enc["input_ids"]
        self.clean_attention_mask = clean_enc["attention_mask"]
        self.rewrite_input_ids = rewrite_enc["input_ids"]
        self.rewrite_attention_mask = rewrite_enc["attention_mask"]
        if use_continuous_features:
            self.clean_cont_features = torch.tensor(
                [continuous_features_for_text(r, "mixed_text") for r in filtered],
                dtype=torch.float32,
            )
            if rewrite_continuous_source == "mixed":
                rewrite_features = [continuous_features_for_text(r, "mixed_text") for r in filtered]
            elif rewrite_continuous_source == "rewrite":
                rewrite_features = [continuous_features_for_text(r, "rewritten_text") for r in filtered]
            else:
                rewrite_features = [[0.0] * len(CONT_FEATURE_NAMES) for _ in filtered]
            self.rewrite_cont_features = torch.tensor(rewrite_features, dtype=torch.float32)
        else:
            self.clean_cont_features = torch.zeros((len(filtered), len(CONT_FEATURE_NAMES)), dtype=torch.float32)
            self.rewrite_cont_features = torch.zeros((len(filtered), len(CONT_FEATURE_NAMES)), dtype=torch.float32)
        self.bins = torch.tensor(bins, dtype=torch.long)
        self.ratios = torch.tensor(ratios, dtype=torch.float32)
        self.ref_clean_logits: torch.Tensor | None = None
        self.ref_clean_reg: torch.Tensor | None = None
        self.ref_rewrite_logits: torch.Tensor | None = None
        self.ref_rewrite_reg: torch.Tensor | None = None

    def attach_reference_cache(
        self,
        clean_logits: torch.Tensor,
        clean_reg: torch.Tensor,
        rewrite_logits: torch.Tensor,
        rewrite_reg: torch.Tensor,
    ) -> None:
        self.ref_clean_logits = clean_logits.detach().cpu()
        self.ref_clean_reg = clean_reg.detach().cpu()
        self.ref_rewrite_logits = rewrite_logits.detach().cpu()
        self.ref_rewrite_reg = rewrite_reg.detach().cpu()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        item = {
            "clean_input_ids": self.clean_input_ids[idx],
            "clean_attention_mask": self.clean_attention_mask[idx],
            "rewrite_input_ids": self.rewrite_input_ids[idx],
            "rewrite_attention_mask": self.rewrite_attention_mask[idx],
            "clean_cont_features": self.clean_cont_features[idx],
            "rewrite_cont_features": self.rewrite_cont_features[idx],
            "proportion_bin": self.bins[idx],
            "exact_pct": self.ratios[idx],
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "sample_id": rec.get("id", f"sample_{idx}"),
            "is_attacked": self.is_attacked[idx],
        }
        if self.ref_clean_logits is not None:
            item["ref_clean_logits"] = self.ref_clean_logits[idx]
            item["ref_clean_reg"] = self.ref_clean_reg[idx]
        if self.ref_rewrite_logits is not None:
            item["ref_rewrite_logits"] = self.ref_rewrite_logits[idx]
            item["ref_rewrite_reg"] = self.ref_rewrite_reg[idx]
        return item


class SingleTextDataset(Dataset):
    """Evaluation dataset for either mixed_text or rewritten_text."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        text_field: str,
        max_length: int = 512,
        use_continuous_features: bool = False,
        continuous_source: str = "self",
    ) -> None:
        if continuous_source not in {"self", "mixed", "zero"}:
            raise ValueError("continuous_source must be one of: self, mixed, zero")
        filtered = [r for r in records if str(r.get(text_field, "") or "").strip()]
        if not filtered:
            raise ValueError(f"No valid records for text field {text_field!r}.")
        self.records = filtered
        self.text_field = text_field

        texts = [str(r[text_field]) for r in filtered]
        ratios = [float(r["target_ai_ratio"]) for r in filtered]
        bins = [ratio_to_bin(r) for r in ratios]

        enc = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        if use_continuous_features:
            if continuous_source == "self":
                features = [continuous_features_for_text(r, text_field) for r in filtered]
            elif continuous_source == "mixed":
                features = [continuous_features_for_text(r, "mixed_text") for r in filtered]
            else:
                features = [[0.0] * len(CONT_FEATURE_NAMES) for _ in filtered]
            self.cont_features = torch.tensor(features, dtype=torch.float32)
        else:
            self.cont_features = torch.zeros((len(filtered), len(CONT_FEATURE_NAMES)), dtype=torch.float32)
        self.bins = torch.tensor(bins, dtype=torch.long)
        self.ratios = torch.tensor(ratios, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "cont_features": self.cont_features[idx],
            "proportion_bin": self.bins[idx],
            "exact_pct": self.ratios[idx],
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "sample_id": rec.get("id", f"sample_{idx}"),
        }
