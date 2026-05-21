"""
Evaluate a saved RATIO Stage-1 checkpoint on a rewrite split.

The rewrite split stores both the original mixed document and an adversarially
rewritten version.  This evaluator replaces mixed_text with rewritten_text by
default and maps rewrite_* continuous targets onto RATIO target field names so
metrics remain comparable with eval_lir_joint_checkpoint.py.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ratio_detector.eval_lir_joint_checkpoint import namespace_from_checkpoint_args
from ratio_detector.model import MultiHeadDetector
from ratio_detector.train import evaluate_lir
from ratio_detector.training_utils import count_bins, load_jsonl_records
from ratio_detector.utils import AUX_TARGET_NAMES, ProportionDataset, compute_class_weights


TARGET_FIELD_MAPS: dict[str, dict[str, str]] = {
    "rewritten_text": {
        "lir": "rewrite_lir",
        "jaccard_distance": "rewrite_jaccard_distance",
        "sentence_jaccard": "rewrite_sentence_jaccard",
        "cosine_distance": "rewrite_cosine_distance",
    },
    "mixed_text": {
        "lir": "mixed_lir",
        "jaccard_distance": "mixed_jaccard_distance",
        "sentence_jaccard": "mixed_sentence_jaccard",
        "cosine_distance": "mixed_cosine_distance",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved RATIO checkpoint on a rewrite split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/ratio_stage1/best_model.pt",
    )
    parser.add_argument(
        "--test_data",
        default="data/rewrite/test.jsonl",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument(
        "--text_field",
        default="rewritten_text",
        choices=["rewritten_text", "mixed_text"],
        help="Which text field to feed as model input.",
    )
    parser.add_argument(
        "--target_prefix",
        default="auto",
        choices=["auto", "rewrite", "mixed"],
        help="Which continuous-target prefix to map onto lir/jaccard/cosine fields.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed_precision", default="no", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def target_fields_for(text_field: str, target_prefix: str) -> dict[str, str]:
    if target_prefix == "auto":
        return TARGET_FIELD_MAPS[text_field]
    if target_prefix == "rewrite":
        return TARGET_FIELD_MAPS["rewritten_text"]
    if target_prefix == "mixed":
        return TARGET_FIELD_MAPS["mixed_text"]
    raise ValueError(f"Unsupported target_prefix: {target_prefix}")


def prepare_rewrite_records(
    records: list[dict[str, Any]],
    text_field: str,
    target_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    field_map = target_fields_for(text_field, target_prefix)
    prepared: list[dict[str, Any]] = []
    missing_text = 0
    missing_targets = {name: 0 for name in field_map}

    for idx, record in enumerate(records):
        text = record.get(text_field)
        if not text:
            missing_text += 1
            continue

        copied = dict(record)
        copied["mixed_text"] = text

        for target_name, source_name in field_map.items():
            value = record.get(source_name)
            if value is None:
                missing_targets[target_name] += 1
            else:
                copied[target_name] = value

        copied["_eval_original_index"] = idx
        copied["_eval_text_field"] = text_field
        copied["_eval_target_field_map"] = field_map
        prepared.append(copied)

    stats = {"missing_text": missing_text}
    stats.update({f"missing_{name}": count for name, count in missing_targets.items()})
    return prepared, stats


def model_name_from_checkpoint(ckpt: dict[str, Any], args: Namespace) -> str:
    model_name = getattr(args, "model_name", None)
    if model_name:
        return str(model_name)
    model_config = ckpt.get("model_config", {})
    return str(model_config.get("model_name", "microsoft/deberta-v3-large"))


def main() -> None:
    cli_args = parse_args()
    checkpoint_path = Path(cli_args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = namespace_from_checkpoint_args(ckpt.get("args", {}))
    model_name = model_name_from_checkpoint(ckpt, train_args)

    raw_records = load_jsonl_records(cli_args.test_data)
    eval_records, prep_stats = prepare_rewrite_records(
        raw_records,
        text_field=cli_args.text_field,
        target_prefix=cli_args.target_prefix,
    )
    if not eval_records:
        raise ValueError(
            f"No evaluable records found in {cli_args.test_data} with text_field={cli_args.text_field!r}"
        )

    accelerator = Accelerator(mixed_precision=cli_args.mixed_precision)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_ds = ProportionDataset(
        eval_records,
        tokenizer,
        max_length=int(train_args.max_seq_len),
        load_continuous_features=bool(train_args.use_continuous_features),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
    )

    model = MultiHeadDetector(
        model_name=model_name,
        num_classes=int(train_args.num_classes),
        dropout=float(train_args.dropout),
        use_continuous_features=bool(train_args.use_continuous_features),
    )
    model.set_special_token_ids(tokenizer)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    class_weights = None
    if bool(train_args.use_class_weights):
        class_weights = compute_class_weights(
            test_ds.bins.numpy(),
            method=str(train_args.weighting_method),
            zero_class_boost=float(train_args.zero_class_boost),
        )

    model, test_dl = accelerator.prepare(model, test_dl)
    metrics = evaluate_lir(
        accelerator.unwrap_model(model),
        test_dl,
        cont_features_enabled=bool(train_args.use_continuous_features),
        accelerator=accelerator,
        args=train_args,
        class_weights=class_weights,
    )

    if accelerator.is_main_process:
        field_map = target_fields_for(cli_args.text_field, cli_args.target_prefix)
        output_json = cli_args.output_json
        if not output_json:
            suffix = f"{cli_args.text_field}_rewrite_eval_results.json"
            output_json = str(checkpoint_path.parent / suffix)

        result = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)) + 1,
            "test_data": cli_args.test_data,
            "text_field": cli_args.text_field,
            "target_prefix": cli_args.target_prefix,
            "target_field_map": field_map,
            "model_name": model_name,
            "detector_input_mode": "metadata_inputs" if bool(train_args.use_continuous_features) else "text_only",
            "inference_requires_continuous_features": bool(train_args.use_continuous_features),
            "use_continuous_features": bool(train_args.use_continuous_features),
            "use_aux_targets": bool(train_args.use_aux_targets),
            "aux_target_names": AUX_TARGET_NAMES,
            "raw_records": len(raw_records),
            "eval_records": len(eval_records),
            "prepare_stats": prep_stats,
            "data_counts": {
                "raw": count_bins(raw_records),
                "eval": count_bins(eval_records),
            },
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "metrics": metrics,
        }

        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"records: {len(eval_records)} / {len(raw_records)}")
        print(f"text_field: {cli_args.text_field}")
        print(f"target_field_map: {field_map}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"Macro F1: {metrics.get('macro_f1', 0.0):.4f}")
        print(f"MAE(LIR): {metrics.get('mae', 0.0):.4f}")
        print(f"AUROC: {metrics.get('auroc', 0.0):.4f}")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
