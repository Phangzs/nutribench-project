#!/usr/bin/env python
"""Utility script for evaluating saved checkpoints and running ad-hoc predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or query a saved transformer checkpoint.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run MAE/Acc@7.5 on a dataset split.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to a HuggingFace checkpoint directory.")
    eval_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate (default: test).",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during batched inference (default: 32).",
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV to store predictions. Columns: query,label,prediction",
    )

    predict_parser = subparsers.add_parser("predict", help="Load a checkpoint and answer free-form prompts.")
    predict_parser.add_argument("--checkpoint", required=True, help="Path to a HuggingFace checkpoint directory.")
    predict_parser.add_argument(
        "--text",
        default=None,
        help="Optional single description to score. If omitted, enters interactive mode.",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batched CLI inputs (default: 32).",
    )

    return parser.parse_args()


def load_model(checkpoint: str):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint '{checkpoint}' was not found.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


@torch.inference_mode()
def predict_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    preds: List[float] = []
    for batch in batched(texts, batch_size):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits.squeeze(-1).detach().cpu().numpy()
        preds.extend(logits.tolist())
    return np.asarray(preds, dtype=np.float32)


def evaluate_split(args: argparse.Namespace) -> None:
    split_file = Path("data") / f"{args.split}.csv"
    if not split_file.exists():
        raise SystemExit(f"Missing data split '{split_file}'. Run the data step first.")

    df = pd.read_csv(split_file)
    if "query" not in df.columns or "label" not in df.columns:
        raise SystemExit(f"{split_file} must contain 'query' and 'label' columns.")

    tokenizer, model, device = load_model(args.checkpoint)
    predictions = predict_texts(model, tokenizer, device, df["query"].tolist(), args.batch_size)
    labels = df["label"].to_numpy(dtype=np.float32)

    mae = np.mean(np.abs(predictions - labels))
    acc = np.mean(np.abs(predictions - labels) <= 7.5)

    print("=" * 72)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print("-" * 72)
    print(f"MAE        : {mae:.4f}")
    print(f"Accuracy@7.5 : {acc * 100:.2f}%")
    print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df = df.copy()
        out_df["prediction"] = predictions
        out_df.to_csv(args.output, index=False)
        print(f"[+] Saved predictions to {args.output}")


def interactive_predict(args: argparse.Namespace) -> None:
    tokenizer, model, device = load_model(args.checkpoint)

    def score_inputs(batch: List[str]) -> np.ndarray:
        preds = predict_texts(model, tokenizer, device, batch, args.batch_size)
        return preds

    if args.text:
        pred = float(score_inputs([args.text])[0])
        print(f"Input: {args.text}")
        print(f"Predicted carbs: {pred:.2f} g")
        return

    print("Enter a meal description (empty line to exit):")
    buffer: List[str] = []
    try:
        while True:
            text = input("> ").strip()
            if not text:
                if buffer:
                    outputs = score_inputs(buffer)
                    for in_text, out_val in zip(buffer, outputs):
                        print(f"• {out_val:.2f} g\t<= {in_text}")
                    buffer.clear()
                else:
                    break
                continue

            buffer.append(text)
            if len(buffer) >= args.batch_size:
                outputs = score_inputs(buffer)
                for in_text, out_val in zip(buffer, outputs):
                    print(f"• {out_val:.2f} g\t<= {in_text}")
                buffer.clear()

    except (EOFError, KeyboardInterrupt):
        pass

    if buffer:
        outputs = score_inputs(buffer)
        for in_text, out_val in zip(buffer, outputs):
            print(f"• {out_val:.2f} g\t<= {in_text}")


def main() -> None:
    args = parse_args()
    if args.mode == "evaluate":
        evaluate_split(args)
    elif args.mode == "predict":
        interactive_predict(args)
    else:
        raise SystemExit(f"Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()
