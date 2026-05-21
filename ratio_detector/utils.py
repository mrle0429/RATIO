"""
utils.py — Dataset, metrics, early-stopping, and data-split utilities
for the RATIO AI-text proportion detector.

RATIO:
    - Uses DeBERTa-v3-large as backbone
    - 6-class classification (0%, 20%, 40%, 60%, 80%, 100% AI content)
    - LIR regression for continuous AI involvement estimation
    - Stage-2 RCDPO robust training on clean/rewrite pairs
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
import warnings
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #
RATIO_TO_BIN: dict[float, int] = {
    0.0: 0, 0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1.0: 5,
}

BIN_TO_RATIO: dict[int, float] = {v: k for k, v in RATIO_TO_BIN.items()}

BIN_NAMES: list[str] = [
    "0pct", "20pct", "40pct", "60pct", "80pct", "100pct",
]

BIN_EDGES: list[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

CONTINUOUS_FEATURES: list[str] = [
    "jaccard_distance",
    "sentence_jaccard",
    "cosine_distance",
    "lir",
]

AUX_TARGET_NAMES: list[str] = [
    "lir",
    "jaccard",
    "sentence_jaccard",
    "cosine",
]

AUX_TARGET_TO_FIELD: dict[str, str] = {
    "lir": "lir",
    "jaccard": "jaccard_distance",
    "sentence_jaccard": "sentence_jaccard",
    "cosine": "cosine_distance",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ENCODER_CACHE: dict[str, Any] = {}


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value_f = float(value)
        if not np.isfinite(value_f):
            return default
        return _clip01(value_f)
    except (TypeError, ValueError):
        return default


def _normalize_words(text: str) -> list[str]:
    cleaned = _PUNCT_RE.sub(" ", (text or "").lower())
    return cleaned.split()


def _get_token_encoder(encoding_name: str = "cl100k_base"):
    if encoding_name in _ENCODER_CACHE:
        encoder = _ENCODER_CACHE[encoding_name]
        if encoder is None:
            raise RuntimeError(f"tiktoken encoder unavailable: {encoding_name}")
        return encoder

    try:
        import tiktoken
    except Exception as exc:
        _ENCODER_CACHE[encoding_name] = None
        raise RuntimeError(
            f"Unable to load tiktoken encoder '{encoding_name}'. "
            "Exact LIR fallback requires tiktoken."
        ) from exc

    try:
        encoder = tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        _ENCODER_CACHE[encoding_name] = None
        raise RuntimeError(f"Unable to load tiktoken encoder: {encoding_name}") from exc

    _ENCODER_CACHE[encoding_name] = encoder
    return encoder


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        encoder = _get_token_encoder(encoding_name)
        return len(encoder.encode(text or ""))
    except RuntimeError:
        warnings.warn(
            "tiktoken is unavailable; falling back to whitespace token counts "
            "for LIR fallback reconstruction only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return len(_normalize_words(text or ""))


def jaccard_distance_text(a: str, b: str) -> float:
    """Set-based word Jaccard distance: 1 - |A cap B| / |A cup B|."""
    a_set = set(_normalize_words(a))
    b_set = set(_normalize_words(b))
    union = a_set | b_set
    if not union:
        return 0.0
    return round(_clip01(1.0 - (len(a_set & b_set) / len(union))), 6)


def split_sentences_rough(text: str) -> list[str]:
    """Lightweight sentence split used only for fallback target construction."""
    parts = _SENTENCE_SPLIT_RE.split(text or "")
    return [part.strip() for part in parts if part.strip()]


def length_involvement_ratio(
    record: dict[str, Any],
    encoding_name: str = "cl100k_base",
) -> float | None:
    """
    Standard LIR fallback: token mass of AI-labelled sentences divided by
    total token mass in the observed mixed document.
    """
    labels = record.get("sentence_labels") or []
    if not labels:
        return _safe_float(record.get("target_ai_ratio", 0.0))

    mixed_sentences = split_sentences_rough(record.get("mixed_text", ""))
    if not mixed_sentences:
        return _safe_float(sum(float(x) for x in labels) / max(len(labels), 1))

    full_text = " ".join(mixed_sentences)
    total_tokens = count_tokens(full_text, encoding_name)
    if total_tokens <= 0:
        return 0.0

    ai_tokens = 0
    for idx, label in enumerate(labels):
        if not label or idx >= len(mixed_sentences):
            continue
        ai_tokens += count_tokens(mixed_sentences[idx], encoding_name)
    return round(_clip01(ai_tokens / total_tokens), 6)


def sentence_jaccard_distance(record: dict[str, Any]) -> float | None:
    """
    Fallback sentence-level Jaccard distance over only edited sentence pairs.
    Unchanged sentences do not participate in this target.
    """
    labels = record.get("sentence_labels") or []
    if not labels:
        return 0.0

    original_sentences = split_sentences_rough(record.get("original_text", ""))
    mixed_sentences = split_sentences_rough(record.get("mixed_text", ""))

    if len(original_sentences) != len(mixed_sentences) or len(original_sentences) != len(labels):
        return None

    scores = [
        jaccard_distance_text(original_sentences[idx], mixed_sentences[idx])
        for idx, label in enumerate(labels)
        if int(label) == 1
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


def _ngram_tf(text: str, n: int = 2) -> Counter:
    words = _normalize_words(text)
    if len(words) < n:
        return Counter()
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def cosine_distance_text(a: str, b: str, n: int = 2) -> float:
    vec_a = _ngram_tf(a, n)
    vec_b = _ngram_tf(b, n)
    keys = set(vec_a) | set(vec_b)
    if not keys:
        return 0.0

    dot = sum(vec_a[key] * vec_b[key] for key in keys)
    norm_a = math.sqrt(sum(value ** 2 for value in vec_a.values()))
    norm_b = math.sqrt(sum(value ** 2 for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cosine_sim = dot / (norm_a * norm_b)
    return round(_clip01(1.0 - min(cosine_sim, 1.0)), 6)


def continuous_targets_from_record(record: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """
    Build physically meaningful continuous targets for multi-task supervision.

    Dataset-provided values are preferred. Missing values are reconstructed
    from original_text, mixed_text, and sentence_labels when possible.
    """
    lir = _safe_float(record.get("lir"), default=np.nan)
    mask_lir = 1.0
    if not np.isfinite(lir):
        lir = length_involvement_ratio(record)
    if lir is None or not np.isfinite(lir):
        lir = 0.0
        mask_lir = 0.0

    jaccard = _safe_float(record.get("jaccard_distance"), default=np.nan)
    mask_jaccard = 1.0
    if not np.isfinite(jaccard):
        jaccard = jaccard_distance_text(
            record.get("original_text", ""),
            record.get("mixed_text", ""),
        )
    if jaccard is None or not np.isfinite(jaccard):
        jaccard = 0.0
        mask_jaccard = 0.0

    sent_jaccard = _safe_float(record.get("sentence_jaccard"), default=np.nan)
    mask_sent_jaccard = 1.0
    if not np.isfinite(sent_jaccard):
        sent_jaccard = sentence_jaccard_distance(record)
    if sent_jaccard is None or not np.isfinite(sent_jaccard):
        sent_jaccard = 0.0
        mask_sent_jaccard = 0.0

    cosine = _safe_float(record.get("cosine_distance"), default=np.nan)
    mask_cosine = 1.0
    if not np.isfinite(cosine):
        cosine = cosine_distance_text(
            record.get("original_text", ""),
            record.get("mixed_text", ""),
        )
    if cosine is None or not np.isfinite(cosine):
        cosine = 0.0
        mask_cosine = 0.0

    targets = {
        "lir": _clip01(lir),
        "jaccard": _clip01(jaccard),
        "sentence_jaccard": _clip01(sent_jaccard),
        "cosine": _clip01(cosine),
    }
    masks = {
        "lir": mask_lir,
        "jaccard": mask_jaccard,
        "sentence_jaccard": mask_sent_jaccard,
        "cosine": mask_cosine,
    }
    return targets, masks


# ------------------------------------------------------------------ #
#  Class imbalance handling                                            #
# ------------------------------------------------------------------ #
def compute_class_weights(
    labels: list[int] | torch.Tensor | np.ndarray,
    num_classes: int = 6,
    method: str = "effective_number",
    zero_class_boost: float = 6.0,
    clip_min: float = 0.5,
    clip_max: float = 6.0,
) -> torch.Tensor:
    """Compute per-class CE/sampling weights for imbalanced ratio bins."""
    if isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    labels = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    balanced_count = class_counts.sum() / num_classes

    if method == "effective_number":
        beta = 0.9999
        effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
        weights = balanced_count / (effective_num + 1e-8)
    elif method == "inverse_freq":
        weights = balanced_count / (class_counts + 1e-8)
    elif method == "sqrt_inverse":
        weights = np.sqrt(balanced_count / (class_counts + 1e-8))
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    if zero_class_boost > 1.0:
        weights[0] *= zero_class_boost

    weights = np.clip(weights, clip_min, clip_max)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(
    labels: list[int] | torch.Tensor | np.ndarray,
    num_samples: int | None = None,
    zero_class_boost: float = 6.0,
    method: str = "sqrt_inverse",
) -> WeightedRandomSampler:
    """Create a replacement sampler that oversamples minority classes."""
    if isinstance(labels, torch.Tensor):
        label_tensor = labels.detach().cpu().long()
    else:
        label_tensor = torch.tensor(labels, dtype=torch.long)

    class_weights = compute_class_weights(
        label_tensor,
        num_classes=6,
        method=method,
        zero_class_boost=zero_class_boost,
    )
    sample_weights = class_weights[label_tensor]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples or len(label_tensor),
        replacement=True,
    )


_RATIO_ID_RE = re.compile(r"_r(?:0|20|40|60|80|100)_(?:block|scatter)$")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def base_sample_id(record: dict[str, Any]) -> str:
    """Return the human-source group id shared by all ratio variants."""
    sample_id = str(record.get("id", ""))
    stripped = _RATIO_ID_RE.sub("", sample_id)
    return stripped or sample_id


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Lightweight sentence span splitter.

    PACT already stores sentence-level labels.  For token supervision we only
    need approximate char spans in the final mixed text; this regex keeps the
    implementation dependency-free and falls back to a single span if needed.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    if not spans and text:
        spans.append((0, len(text)))
    return spans


def _token_ai_targets(
    text: str,
    sentence_labels: list[int],
    offset_mapping: list[tuple[int, int]],
) -> tuple[list[float], list[float]]:
    """Map sentence-level labels to token-level labels using token offsets."""
    token_labels = [-100.0] * len(offset_mapping)
    token_mask = [0.0] * len(offset_mapping)

    spans = _sentence_spans(text)
    if not spans or not sentence_labels:
        return token_labels, token_mask

    usable = min(len(spans), len(sentence_labels))
    spans = spans[:usable]
    labels = [float(int(v)) for v in sentence_labels[:usable]]

    for i, (start, end) in enumerate(offset_mapping):
        if end <= start:
            continue
        mid = (start + end) / 2
        sent_idx = None
        for idx, (s_start, s_end) in enumerate(spans):
            if s_start <= mid <= s_end:
                sent_idx = idx
                break
        if sent_idx is None:
            continue
        token_labels[i] = labels[sent_idx]
        token_mask[i] = 1.0

    return token_labels, token_mask


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #
class ProportionDataset(Dataset):
    """
    Pre-tokenized dataset for AI-proportion detection.

    Supports both classification (bin) and regression (exact percentage) targets.

    Args:
        records: List of data records from JSONL file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 512)
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 512,
        load_continuous_features: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_continuous_features = load_continuous_features

        texts = [r["mixed_text"] for r in records]
        ratios = [r["target_ai_ratio"] for r in records]

        try:
            enc = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offset_mapping = enc.pop("offset_mapping").tolist()
        except Exception:
            enc = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            offset_mapping = [[(0, 0)] * max_length for _ in texts]

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        self.bins = torch.tensor(
            [RATIO_TO_BIN.get(r, int(round(r * 5))) for r in ratios],
            dtype=torch.long,
        )
        self.pcts = torch.tensor(ratios, dtype=torch.float32)
        aux_pairs = [continuous_targets_from_record(r) for r in records]
        aux_dicts = [targets for targets, _ in aux_pairs]
        aux_masks = [masks for _, masks in aux_pairs]
        self.aux_targets = torch.tensor(
            [[d[name] for name in AUX_TARGET_NAMES] for d in aux_dicts],
            dtype=torch.float32,
        )
        self.aux_target_mask = torch.tensor(
            [[m[name] for name in AUX_TARGET_NAMES] for m in aux_masks],
            dtype=torch.float32,
        )
        self.lir = self.aux_targets[:, AUX_TARGET_NAMES.index("lir")]
        self.jaccard = self.aux_targets[:, AUX_TARGET_NAMES.index("jaccard")]
        self.sentence_jaccard = self.aux_targets[:, AUX_TARGET_NAMES.index("sentence_jaccard")]
        self.cosine = self.aux_targets[:, AUX_TARGET_NAMES.index("cosine")]

        token_labels: list[list[float]] = []
        token_masks: list[list[float]] = []
        for rec, offsets in zip(records, offset_mapping):
            labels, mask = _token_ai_targets(
                rec["mixed_text"],
                rec.get("sentence_labels", []),
                [tuple(x) for x in offsets],
            )
            token_labels.append(labels)
            token_masks.append(mask)

        self.token_ai_labels = torch.tensor(token_labels, dtype=torch.float32)
        self.token_label_mask = torch.tensor(token_masks, dtype=torch.float32)

    def __len__(self) -> int:
        return self.bins.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "proportion_bin": self.bins[idx],
            "exact_pct": self.pcts[idx],
            "aux_targets": self.aux_targets[idx],
            "aux_target_mask": self.aux_target_mask[idx],
            "target_lir": self.lir[idx],
            "target_jaccard": self.jaccard[idx],
            "target_sentence_jaccard": self.sentence_jaccard[idx],
            "target_cosine": self.cosine[idx],
            "jaccard_distance": self.jaccard[idx],
            "sentence_jaccard": self.sentence_jaccard[idx],
            "cosine_distance": self.cosine[idx],
            "lir": self.lir[idx],
            "token_ai_labels": self.token_ai_labels[idx],
            "token_label_mask": self.token_label_mask[idx],
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "sample_id": rec.get("id", f"sample_{idx}"),
        }

        if self.load_continuous_features:
            item["jaccard"] = self.jaccard[idx]
            item["sentence_jaccard"] = self.sentence_jaccard[idx]
            item["cosine"] = self.cosine[idx]
            item["lir"] = self.lir[idx]

        return item

    def get_class_distribution(self) -> dict[str, float]:
        """Return the class distribution as a dictionary."""
        unique, counts = self.bins.unique(return_counts=True)
        total = len(self.bins)
        return {
            BIN_NAMES[bin_id]: counts[i].item() / total
            for i, bin_id in enumerate(unique)
        }


# ------------------------------------------------------------------ #
#  Data loading & stratified split                                     #
# ------------------------------------------------------------------ #
def load_multiple_datasets(
    filepaths: list[str | Path],
    seed: int = 42,
) -> list[dict]:
    """
    Load and merge multiple JSONL datasets.

    Args:
        filepaths: List of paths to JSONL data files
        seed: Random seed for shuffling

    Returns:
        Combined list of records from all datasets
    """
    all_records = []

    for filepath in filepaths:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[警告] 数据文件不存在，跳过: {filepath}")
            continue

        with open(filepath, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        # 添加数据来源标记
        for r in records:
            r["_source_file"] = filepath.stem

        all_records.extend(records)
        print(f"  加载 {filepath.name}: {len(records)} 条")

    if not all_records:
        raise ValueError("没有加载到任何数据！")

    # 打乱数据
    rng = np.random.default_rng(seed)
    rng.shuffle(all_records)

    print(f"  合计加载: {len(all_records)} 条")
    return all_records


def load_and_split_data(
    filepath: str | Path | list[str | Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
    per_class_split: bool = False,
    output_split_path: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Load JSONL and return (train, val, test) record lists.

    Supports:
    1. Single file path - loads one dataset
    2. List of file paths - merges multiple datasets

    Supports two split modes:
    1. Global stratified split (stratify=True, per_class_split=False):
       Stratified by target_ai_ratio across the entire dataset.

    2. Per-class stratified split (per_class_split=True):
       Each class is split independently at train_ratio:val_ratio:test_ratio.
       This ensures every class has samples in train/val/test.

    Args:
        filepath: Path to the JSONL data file, or list of paths for multiple datasets
        train_ratio: Fraction of data for training (default: 0.8)
        val_ratio: Fraction of data for validation (default: 0.1)
        seed: Random seed for reproducibility
        stratify: Whether to stratify splits by target_ai_ratio
        per_class_split: If True, split each class independently
        output_split_path: If provided, save test set to this JSON file

    Returns:
        Tuple of (train_records, val_records, test_records)
    """
    # 支持多数据集合并
    if isinstance(filepath, (list, tuple)):
        records = load_multiple_datasets(filepath, seed)
    else:
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

    rng = np.random.default_rng(seed)

    if per_class_split:
        # Per-class stratified split: each class independently split
        train_recs, val_recs, test_recs = [], [], []

        for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            # Get all records for this class
            class_records = [r for r in records if r["target_ai_ratio"] == ratio]
            rng.shuffle(class_records)

            n = len(class_records)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            # Ensure we have samples in each split
            n_val = min(n_val, n - n_train)
            n_test = n - n_train - n_val

            train_recs.extend(class_records[:n_train])
            val_recs.extend(class_records[n_train:n_train + n_val])
            test_recs.extend(class_records[n_train + n_val:])

        # Shuffle each split to mix classes
        rng.shuffle(train_recs)
        rng.shuffle(val_recs)
        rng.shuffle(test_recs)
    elif stratify:
        strat = [r["target_ai_ratio"] for r in records]
        train_recs, tmp_recs, _, tmp_strat = train_test_split(
            records,
            strat,
            test_size=1 - train_ratio,
            random_state=seed,
            stratify=strat,
        )
        val_recs, test_recs = train_test_split(
            tmp_recs,
            test_size=0.5,
            random_state=seed,
            stratify=tmp_strat,
        )
    else:
        train_recs, tmp_recs = train_test_split(
            records,
            test_size=1 - train_ratio,
            random_state=seed,
        )
        val_size = val_ratio / (1 - train_ratio)
        val_recs, test_recs = train_test_split(
            tmp_recs,
            test_size=1 - val_size,
            random_state=seed,
        )

    # Save test set to JSON if path is provided
    if output_split_path is not None:
        output_split_path = Path(output_split_path)
        output_split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_split_path, "w", encoding="utf-8") as f:
            for rec in test_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] Test set saved to {output_split_path}")

    return train_recs, val_recs, test_recs


def split_records_by_group(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split records by source-document group to prevent leakage.

    PACT emits multiple ratio variants from the same human source.  A random
    record-level split leaks the source document across train/val/test and can
    make the detector memorize topic/content instead of AI proportion cues.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(base_sample_id(rec), []).append(rec)

    group_ids = np.array(sorted(groups))
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train:n_train + n_val])
    test_ids = set(group_ids[n_train + n_val:])

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    test_recs: list[dict] = []
    for gid, items in groups.items():
        if gid in train_ids:
            train_recs.extend(items)
        elif gid in val_ids:
            val_recs.extend(items)
        elif gid in test_ids:
            test_recs.extend(items)

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)
    return train_recs, val_recs, test_recs


def create_kfold_splits(
    filepath: str | Path,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[dict], list[dict]]]:
    """
    Create K-fold cross-validation splits.

    Args:
        filepath: Path to the JSONL data file
        n_folds: Number of folds (default: 5)
        seed: Random seed for reproducibility

    Returns:
        List of (train_records, val_records) tuples for each fold
    """
    filepath = Path(filepath)
    with open(filepath, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    strat = [r["target_ai_ratio"] for r in records]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []

    for train_idx, val_idx in skf.split(records, strat):
        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        splits.append((train_recs, val_recs))

    return splits


# ------------------------------------------------------------------ #
#  Early stopping                                                      #
# ------------------------------------------------------------------ #
class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self._is_best = False

    def step(self, metric: float) -> bool:
        """
        Feed latest metric. Returns ``True`` when training should stop.

        Args:
            metric: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        self._is_best = False

        if self.best_value is None:
            self.best_value = metric
            self._is_best = True
            return False

        if self.mode == "min":
            improved = metric < self.best_value - self.min_delta
        else:
            improved = metric > self.best_value + self.min_delta

        if improved:
            self.best_value = metric
            self.counter = 0
            self._is_best = True
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def is_best(self) -> bool:
        """Return True if the last step was a new best."""
        return self._is_best

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self._is_best = False


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #
def compute_metrics(
    cls_preds: list | np.ndarray,
    cls_labels: list | np.ndarray,
    reg_preds: list | np.ndarray,
    reg_targets: list | np.ndarray,
    bin_names: list[str] | None = None,
    record_lirs: list | np.ndarray | None = None,
    record_jaccards: list | np.ndarray | None = None,
    record_sentence_jaccards: list | np.ndarray | None = None,
    cls_probs: list | np.ndarray | None = None,
    aux_preds: list | np.ndarray | None = None,
    aux_targets: list | np.ndarray | None = None,
    aux_masks: list | np.ndarray | None = None,
    aux_names: list[str] | tuple[str, ...] | None = None,
    all_jaccard_preds: list | np.ndarray | None = None,
    all_sent_jaccard_preds: list | np.ndarray | None = None,
    all_cosine_preds: list | np.ndarray | None = None,
    all_lir_preds: list | np.ndarray | None = None,
    all_jaccard_targets: list | np.ndarray | None = None,
    all_sent_jaccard_targets: list | np.ndarray | None = None,
    all_cosine_targets: list | np.ndarray | None = None,
    all_lir_targets: list | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        cls_preds: Classification predictions (class indices)
        cls_labels: Classification labels (class indices)
        reg_preds: Regression predictions (float values in [0, 1])
        reg_targets: Regression targets (float values in [0, 1])
        bin_names: Optional list of bin names for metric keys
        record_lirs: Optional dataset-provided LIR values for comparable metrics
        record_jaccards: Optional dataset-provided Jaccard distances
        record_sentence_jaccards: Optional dataset-provided sentence Jaccard distances
        cls_probs: Optional class probabilities for multi-class AUROC
        aux_preds: Optional model predictions for continuous auxiliary targets
        aux_targets: Optional gold values for continuous auxiliary targets
        aux_masks: Optional mask indicating valid targets per auxiliary column
        aux_names: Names for aux target columns

    Returns:
        Dictionary containing:
        - Per-class accuracy (acc_*)
        - Per-class F1 score (f1_*)
        - Macro F1 score
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - AUROC (binary: 0% vs >0%)
        - TPR at FPR=0.05 (binary)
        - Confusion matrix
    """
    if bin_names is None:
        bin_names = BIN_NAMES

    cp = np.asarray(cls_preds)
    cl = np.asarray(cls_labels)
    rp = np.asarray(reg_preds)
    rt = np.asarray(reg_targets)

    valid_main = np.isfinite(rp) & np.isfinite(rt)
    if not np.all(valid_main):
        cp = cp[valid_main]
        cl = cl[valid_main]
        rp = rp[valid_main]
        rt = rt[valid_main]
        if cls_probs is not None:
            cls_probs = np.asarray(cls_probs)[valid_main]

    n_cls = len(bin_names)
    m: dict[str, Any] = {}

    if cp.size == 0 or cl.size == 0 or rp.size == 0 or rt.size == 0:
        m["per_class_acc"] = {name: 0.0 for name in bin_names}
        m["per_class_f1"] = {name: 0.0 for name in bin_names}
        m["per_class_precision"] = {name: 0.0 for name in bin_names}
        m["per_class_recall"] = {name: 0.0 for name in bin_names}
        m["per_class_support"] = {name: 0 for name in bin_names}
        m["macro_f1"] = 0.0
        m["macro_F1"] = 0.0
        m["mae"] = 0.0
        m["rmse"] = 0.0
        m["mse"] = 0.0
        m["auroc"] = 0.0
        m["tpr_at_fpr005"] = 0.0
        m["macro_auroc"] = 0.0
        m["multi_class_AUROC"] = 0.0
        m["confusion_matrix"] = np.zeros((len(bin_names), len(bin_names)), dtype=int).tolist()
        m["pred_class_counts"] = {name: 0 for name in bin_names}
        m["true_class_counts"] = {name: 0 for name in bin_names}
        for name in bin_names:
            m[f"acc_{name}"] = 0.0
            m[f"f1_{name}"] = 0.0
            m[f"precision_{name}"] = 0.0
            m[f"recall_{name}"] = 0.0
            m[f"support_{name}"] = 0
            m[f"mae_{name}"] = 0.0
            m[f"auroc_{name}"] = 0.0
        return m
    bin_thresholds = [0, 20, 40, 60, 80, 100]
    m["pred_class_counts"] = {
        name: int((cp == i).sum()) for i, name in enumerate(bin_names)
    }
    m["true_class_counts"] = {
        name: int((cl == i).sum()) for i, name in enumerate(bin_names)
    }
    if cls_probs is not None:
        prob_arr = np.asarray(cls_probs, dtype=float)
        if prob_arr.ndim == 2 and prob_arr.shape[1] >= n_cls:
            m["mean_pred_prob_by_class"] = {
                name: float(prob_arr[:, i].mean()) for i, name in enumerate(bin_names)
            }
            m["mean_max_pred_prob"] = float(prob_arr.max(axis=1).mean())

    # Per-class accuracy
    per_class_acc: dict[str, float] = {}
    for i, name in enumerate(bin_names):
        mask = cl == i
        per_class_acc[name] = float((cp[mask] == i).mean()) if mask.any() else 0.0
        m[f"acc_{name}"] = per_class_acc[name]

    # Per-class F1 & macro F1
    f1s = f1_score(
        cl, cp, labels=list(range(n_cls)), average=None, zero_division=0,
    )
    per_class_f1: dict[str, float] = {}
    for i, name in enumerate(bin_names):
        per_class_f1[name] = float(f1s[i])
        m[f"f1_{name}"] = per_class_f1[name]
        m[f"F1@{bin_thresholds[i]}"] = per_class_f1[name]
    m["per_class_acc"] = per_class_acc
    m["per_class_f1"] = per_class_f1
    m["macro_f1"] = float(f1s.mean())
    m["macro_F1"] = float(f1s.mean())
    m["micro_f1"] = float(f1_score(
        cl, cp, labels=list(range(n_cls)), average="micro", zero_division=0,
    ))
    m["weighted_f1"] = float(f1_score(
        cl, cp, labels=list(range(n_cls)), average="weighted", zero_division=0,
    ))
    m["overall_f1"] = m["micro_f1"]

    # Regression metrics
    m["mae"] = float(np.abs(rp - rt).mean())
    m["mse"] = float(((rp - rt) ** 2).mean())
    m["rmse"] = float(np.sqrt(((rp - rt) ** 2).mean()))
    m["mse"] = float(((rp - rt) ** 2).mean())

    if aux_preds is None and all_lir_preds is not None:
        aux_preds = np.stack(
            [
                np.asarray(all_lir_preds, dtype=float),
                np.asarray(all_jaccard_preds, dtype=float),
                np.asarray(all_sent_jaccard_preds, dtype=float),
                np.asarray(all_cosine_preds, dtype=float),
            ],
            axis=1,
        )
        aux_targets = np.stack(
            [
                np.asarray(all_lir_targets, dtype=float),
                np.asarray(all_jaccard_targets, dtype=float),
                np.asarray(all_sent_jaccard_targets, dtype=float),
                np.asarray(all_cosine_targets, dtype=float),
            ],
            axis=1,
        )
        aux_names = AUX_TARGET_NAMES

    if aux_preds is not None and aux_targets is not None:
        ap = np.asarray(aux_preds, dtype=float)
        at = np.asarray(aux_targets, dtype=float)
        am = np.asarray(aux_masks, dtype=float) if aux_masks is not None else None
        valid_aux = np.isfinite(ap).all(axis=1) & np.isfinite(at).all(axis=1)
        if am is not None:
            valid_aux &= np.isfinite(am).all(axis=1)
        ap = ap[valid_aux]
        at = at[valid_aux]
        if am is not None:
            am = am[valid_aux]
        if ap.ndim == 1:
            ap = ap.reshape(-1, 1)
        if at.ndim == 1:
            at = at.reshape(-1, 1)
        if am is not None and am.ndim == 1:
            am = am.reshape(-1, 1)
        names = list(aux_names or AUX_TARGET_NAMES[: ap.shape[1]])
        n_aux = min(ap.shape[1], at.shape[1], len(names))
        aux_mae: dict[str, float] = {}
        aux_mse: dict[str, float] = {}
        metric_aliases = {
            "lir": "LIR",
            "jaccard": "Jaccard",
            "sentence_jaccard": "Sentence Jaccard",
            "cosine": "Cosine",
        }
        for idx in range(n_aux):
            name = names[idx]
            diff = ap[:, idx] - at[:, idx]
            if am is not None and idx < am.shape[1]:
                mask = am[:, idx] > 0.5
                if mask.any():
                    diff = diff[mask]
                else:
                    m[f"mae_{name}"] = 0.0
                    m[f"mse_{name}"] = 0.0
                    pretty = metric_aliases.get(name)
                    if pretty is not None:
                        m[f"MAE({pretty})"] = 0.0
                        m[f"MSE({pretty})"] = 0.0
                    continue
            mae = float(np.abs(diff).mean())
            mse = float((diff ** 2).mean())
            aux_mae[name] = mae
            aux_mse[name] = mse
            m[f"mae_{name}"] = mae
            m[f"mse_{name}"] = mse
            pretty = metric_aliases.get(name)
            if pretty is not None:
                m[f"MAE({pretty})"] = mae
                m[f"MSE({pretty})"] = mse
        m["aux_mae"] = aux_mae
        m["aux_mse"] = aux_mse
    else:
        if record_lirs is not None:
            lir_targets = np.asarray(record_lirs, dtype=float)
            m["MAE(LIR_target_vs_ratio_pred)"] = float(np.abs(rp - lir_targets).mean())
            m["MSE(LIR_target_vs_ratio_pred)"] = float(((rp - lir_targets) ** 2).mean())
        if record_jaccards is not None:
            j_targets = np.asarray(record_jaccards, dtype=float)
            m["MAE(Jaccard_target_vs_ratio_pred)"] = float(np.abs(rp - j_targets).mean())
            m["MSE(Jaccard_target_vs_ratio_pred)"] = float(((rp - j_targets) ** 2).mean())
        if record_sentence_jaccards is not None:
            sj_targets = np.asarray(record_sentence_jaccards, dtype=float)
            m["MAE(Sentence_Jaccard_target_vs_ratio_pred)"] = float(np.abs(rp - sj_targets).mean())
            m["MSE(Sentence_Jaccard_target_vs_ratio_pred)"] = float(((rp - sj_targets) ** 2).mean())

    # Mean Absolute Error per bin
    for i, name in enumerate(bin_names):
        mask = cl == i
        if mask.any():
            m[f"mae_{name}"] = float(np.abs(rp[mask] - rt[mask]).mean())
        else:
            m[f"mae_{name}"] = 0.0

    prob_arr = None
    if cls_probs is not None:
        prob_arr = np.asarray(cls_probs)

    # ── Per-class AUROC ─────────────────────────────────────────────
    # One-vs-Rest AUROC for each class.  Middle-class AUROC must use the
    # class probability, not the monotonic pct score.
    for i, name in enumerate(bin_names):
        binary_labels = (cl == i).astype(int)
        if len(np.unique(binary_labels)) == 2:
            try:
                score = prob_arr[:, i] if prob_arr is not None and prob_arr.ndim == 2 else rp
                m[f"auroc_{name}"] = float(roc_auc_score(binary_labels, score))
            except ValueError:
                m[f"auroc_{name}"] = 0.0
        else:
            m[f"auroc_{name}"] = 0.0
    m["macro_auroc"] = float(np.mean([m[f"auroc_{name}"] for name in bin_names]))
    m["multi_class_AUROC"] = m["macro_auroc"]

    # Binary view: pure-human (0%) vs any-AI (>0%)
    bt = (cl > 0).astype(int)
    if len(np.unique(bt)) == 2:
        m["auroc"] = float(roc_auc_score(bt, rp))
        fpr, tpr, _ = roc_curve(bt, rp)
        valid = np.where(fpr <= 0.05 + 1e-8)[0]
        m["tpr_at_fpr005"] = float(tpr[valid[-1]]) if valid.size else 0.0
    else:
        m["auroc"] = 0.0
        m["tpr_at_fpr005"] = 0.0

    # Confusion matrix
    m["confusion_matrix"] = confusion_matrix(
        cl, cp, labels=list(range(n_cls)),
    ).tolist()

    bin_step = 1.0 / max(n_cls - 1, 1)
    if np.any(~np.isfinite(rp)):
        n_nan = int(np.sum(np.isnan(rp)))
        n_inf = int(np.sum(np.isinf(rp)))
        warnings.warn(
            f"Non-finite predictions detected: {n_nan} NaN, {n_inf} inf; replacing with 0.5",
            RuntimeWarning,
            stacklevel=2,
        )
        rp = np.where(np.isfinite(rp), rp, 0.5)
    threshold_preds = np.clip(np.round(rp / bin_step), 0, n_cls - 1).astype(int)
    f1s_thresh = f1_score(
        cl, threshold_preds, labels=list(range(n_cls)), average=None, zero_division=0,
    )
    per_class_f1_thresh: dict[str, float] = {}
    for i, name in enumerate(bin_names):
        per_class_f1_thresh[name] = float(f1s_thresh[i])
        m[f"f1_{name}_thresh"] = per_class_f1_thresh[name]
    m["per_class_f1_thresh"] = per_class_f1_thresh
    m["macro_f1_thresh"] = float(f1s_thresh.mean())
    m["threshold_confusion_matrix"] = confusion_matrix(
        cl, threshold_preds, labels=list(range(n_cls)),
    ).tolist()
    m["threshold_classification_report"] = classification_report(
        cl, threshold_preds, labels=list(range(n_cls)),
        target_names=bin_names, output_dict=True, zero_division=0,
    )

    # Precision, Recall, F1, Support
    precision, recall, f1, support = precision_recall_fscore_support(
        cl, cp, labels=list(range(n_cls)), average=None, zero_division=0,
    )
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_support: dict[str, int] = {}
    for i, name in enumerate(bin_names):
        per_class_precision[name] = float(precision[i])
        per_class_recall[name] = float(recall[i])
        per_class_support[name] = int(support[i])
        m[f"precision_{name}"] = per_class_precision[name]
        m[f"recall_{name}"] = per_class_recall[name]
        m[f"support_{name}"] = per_class_support[name]

    m["per_class_precision"] = per_class_precision
    m["per_class_recall"] = per_class_recall
    m["per_class_support"] = per_class_support

    return m


def compute_calibration_metrics(
    cls_probs: np.ndarray,
    cls_labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute calibration metrics (Expected Calibration Error).

    Args:
        cls_probs: Classification probabilities (N, num_classes)
        cls_labels: True labels (N,)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with ECE and other calibration metrics
    """
    probs = np.array(cls_probs)
    labels = np.array(cls_labels)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += bin_size * abs(bin_accuracy - bin_confidence)

    ece /= total_samples

    return {
        "ece": float(ece),
        "n_bins": n_bins,
    }


def print_metrics_table(metrics: dict[str, Any], prefix: str = "") -> None:
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics from compute_metrics
        prefix: Optional prefix for metric keys (e.g., "test/")
    """
    sep = "=" * 60
    print(f"\n{sep}")
    if prefix:
        print(f"  Metrics: {prefix}")
    print(sep)

    print("  Per-class metrics:")
    for name in BIN_NAMES:
        acc = metrics.get(f"{prefix}acc_{name}", 0.0)
        f1 = metrics.get(f"{prefix}f1_{name}", 0.0)
        prec = metrics.get(f"{prefix}precision_{name}", 0.0)
        rec = metrics.get(f"{prefix}recall_{name}", 0.0)
        support = metrics.get(f"{prefix}support_{name}", 0)
        print(f"    {name:>8s}: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  (n={support})")

    print(f"\n  Aggregate metrics:")
    print(f"    {'Macro F1':>12s}: {metrics.get(f'{prefix}macro_f1', 0.0):.4f}")
    print(f"    {'MAE':>12s}: {metrics.get(f'{prefix}mae', 0.0):.4f}")
    print(f"    {'RMSE':>12s}: {metrics.get(f'{prefix}rmse', 0.0):.4f}")
    print(f"    {'AUROC':>12s}: {metrics.get(f'{prefix}auroc', 0.0):.4f}")
    print(f"    {'TPR@FPR=0.05':>12s}: {metrics.get(f'{prefix}tpr_at_fpr005', 0.0):.4f}")
    print(sep)


# ------------------------------------------------------------------ #
#  Data statistics                                                     #
# ------------------------------------------------------------------ #
def analyze_dataset(
    filepath: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Analyze dataset statistics.

    Args:
        filepath: Path to the JSONL data file
        verbose: Whether to print statistics

    Returns:
        Dictionary containing dataset statistics
    """
    filepath = Path(filepath)
    with open(filepath, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    n_samples = len(records)
    ratios = [r["target_ai_ratio"] for r in records]
    text_lengths = [len(r["mixed_text"].split()) for r in records]

    # Class distribution
    unique_ratios, counts = np.unique(ratios, return_counts=True)
    class_dist = {
        BIN_NAMES[RATIO_TO_BIN.get(r, int(round(r * 5)))]: c / n_samples
        for r, c in zip(unique_ratios, counts)
    }

    stats = {
        "n_samples": n_samples,
        "class_distribution": class_dist,
        "text_length_mean": float(np.mean(text_lengths)),
        "text_length_std": float(np.std(text_lengths)),
        "text_length_min": int(np.min(text_lengths)),
        "text_length_max": int(np.max(text_lengths)),
        "n_per_bin": {
            BIN_NAMES[RATIO_TO_BIN.get(r, int(round(r * 5)))]: int(c)
            for r, c in zip(unique_ratios, counts)
        },
    }

    if verbose:
        print("\n" + "=" * 50)
        print("  Dataset Statistics")
        print("=" * 50)
        print(f"  Total samples: {n_samples}")
        print(f"\n  Text length:")
        print(f"    Mean: {stats['text_length_mean']:.1f} words")
        print(f"    Std:  {stats['text_length_std']:.1f}")
        print(f"    Min:  {stats['text_length_min']} words")
        print(f"    Max:  {stats['text_length_max']} words")
        print(f"\n  Class distribution:")
        for name in BIN_NAMES:
            pct = class_dist.get(name, 0) * 100
            n = stats["n_per_bin"].get(name, 0)
            bar = "#" * int(pct / 2)
            print(f"    {name:>8s}: {pct:5.1f}% {bar} (n={n})")
        print("=" * 50)

    return stats


# ------------------------------------------------------------------ #
#  Checkpoint utilities                                                #
# ------------------------------------------------------------------ #
def load_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint into a model.

    Args:
        filepath: Path to the checkpoint file
        model: Model to load weights into
        device: Device to load the checkpoint to

    Returns:
        Checkpoint dictionary (excluding model state dict)
    """
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    info = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return info


def save_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    epoch: int,
    metrics: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> None:
    """
    Save a checkpoint.

    Args:
        filepath: Path to save the checkpoint
        model: Model to save
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        args: Optional training arguments to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": args,
    }

    torch.save(checkpoint, filepath)
