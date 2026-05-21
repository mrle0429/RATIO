"""Evaluate a saved RATIO Stage-1 checkpoint without retraining."""

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

from ratio_detector.model import MultiHeadDetector
from ratio_detector.train import evaluate_lir
from ratio_detector.training_utils import load_jsonl_records
from ratio_detector.utils import ProportionDataset, compute_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved RATIO Stage-1 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/ratio_stage1/best_model.pt",
    )
    parser.add_argument("--test_data", default="data/pact/test.jsonl")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def namespace_from_checkpoint_args(raw_args: dict[str, Any]) -> Namespace:
    defaults = {
        "use_continuous_features": False,
        "use_aux_targets": False,
        "use_class_weights": False,
        "weighting_method": "sqrt_inverse",
        "zero_class_boost": 6.0,
        "label_smoothing": 0.0,
        "w_ce": 1.0,
        "w_mae": 1.0,
        "w_mse": 0.0,
        "w_huber": 0.0,
        "w_consistency": 0.0,
        "w_aux": 0.0,
        "w_aux_mse": 0.0,
        "w_aux_lir": 0.0,
        "w_aux_jaccard": 1.0,
        "w_aux_sentence_jaccard": 1.0,
        "w_aux_cosine": 0.5,
        "strict_finite_outputs": True,
        "num_classes": 6,
        "dropout": 0.2,
        "max_seq_len": 512,
    }
    merged = {**defaults, **raw_args}
    return Namespace(**merged)


def main() -> None:
    cli_args = parse_args()
    checkpoint_path = Path(cli_args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = namespace_from_checkpoint_args(ckpt.get("args", {}))

    model_name = getattr(args, "model_name", None)
    if not model_name:
        model_config = ckpt.get("model_config", {})
        model_name = model_config.get("model_name", "microsoft/deberta-v3-large")

    accelerator = Accelerator(mixed_precision="no")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_records = load_jsonl_records(cli_args.test_data)
    test_ds = ProportionDataset(
        test_records,
        tokenizer,
        max_length=int(args.max_seq_len),
        load_continuous_features=bool(args.use_continuous_features),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
    )

    model = MultiHeadDetector(
        model_name=model_name,
        num_classes=int(args.num_classes),
        dropout=float(args.dropout),
        use_continuous_features=bool(args.use_continuous_features),
    )
    model.set_special_token_ids(tokenizer)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    class_weights = None
    if bool(args.use_class_weights):
        class_weights = compute_class_weights(
            test_ds.bins.numpy(),
            method=str(args.weighting_method),
            zero_class_boost=float(args.zero_class_boost),
        )

    model, test_dl = accelerator.prepare(model, test_dl)
    metrics = evaluate_lir(
        accelerator.unwrap_model(model),
        test_dl,
        cont_features_enabled=bool(args.use_continuous_features),
        accelerator=accelerator,
        args=args,
        class_weights=class_weights,
    )

    if accelerator.is_main_process:
        result = {
            "checkpoint": str(checkpoint_path),
            "test_data": cli_args.test_data,
            "checkpoint_epoch": int(ckpt.get("epoch", -1)) + 1,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "metrics": metrics,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        output_json = cli_args.output_json
        if not output_json:
            output_json = str(checkpoint_path.parent / "retest_best_model_results.json")
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
