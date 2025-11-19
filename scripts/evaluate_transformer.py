#!/usr/bin/env python
"""Utility script for running inference from saved checkpoints."""

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
    parser = argparse.ArgumentParser(description="Run inference with a saved transformer checkpoint.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run inference on a dataset split.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to a HuggingFace checkpoint directory.")
    eval_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to run inference on (default: test).",
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
        help="Optional CSV to store predictions. Defaults to results/<split>_predictions.csv",
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


def run_inference(args: argparse.Namespace) -> None:
    split_file = Path("data") / f"{args.split}.csv"
    if not split_file.exists():
        raise SystemExit(f"Missing data split '{split_file}'. Run the data step first.")

    df = pd.read_csv(split_file)
    if "query" not in df.columns:
        raise SystemExit(f"{split_file} must contain a 'query' column.")

    queries = df["query"].fillna("").astype(str).tolist()
    tokenizer, model, device = load_model(args.checkpoint)
    predictions = predict_texts(model, tokenizer, device, queries, args.batch_size)

    output = df.copy()
    output["prediction"] = predictions
    output_path = args.output or Path("results") / f"{args.split}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"[+] Wrote {len(output)} predictions to {output_path}")


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
        run_inference(args)
    elif args.mode == "predict":
        interactive_predict(args)
    else:
        raise SystemExit(f"Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main()
