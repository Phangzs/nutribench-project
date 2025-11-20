#!/usr/bin/env python
"""Inference utilities for saved RNN checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from train_rnn import RNNRegressor, TextVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved RNN checkpoint.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Run inference on a dataset split.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to an RNN checkpoint directory.")
    eval_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to run inference on (default: test).",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during batched inference (default: 64).",
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV to store predictions. Defaults to results/<split>_predictions_rnn.csv",
    )

    predict_parser = subparsers.add_parser("predict", help="Load a checkpoint and answer free-form prompts.")
    predict_parser.add_argument("--checkpoint", required=True, help="Path to an RNN checkpoint directory.")
    predict_parser.add_argument(
        "--text",
        default=None,
        help="Optional single description to score. If omitted, enters interactive mode.",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for batched CLI inputs (default: 64).",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint: Path, device: torch.device) -> Tuple[TextVectorizer, RNNRegressor]:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint '{checkpoint}' was not found.")

    config_path = checkpoint / "config.json"
    vectorizer_path = checkpoint / "vectorizer.json"
    model_path = checkpoint / "model.pt"

    if not config_path.exists() or not vectorizer_path.exists() or not model_path.exists():
        raise SystemExit(f"Checkpoint '{checkpoint}' is missing config/vectorizer/model files.")

    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)
    with open(vectorizer_path, "r", encoding="utf-8") as fp:
        vectorizer = TextVectorizer.from_dict(json.load(fp))

    model = RNNRegressor(
        vocab_size=len(vectorizer.vocab),
        embedding_dim=int(config["embedding_dim"]),
        rnn1_units=int(config["rnn_units"]),
        rnn2_units=int(config["rnn2_units"]),
        lstm_units=int(config["lstm_units"]),
        dropout=float(config["dropout"]),
        pad_idx=vectorizer.pad_idx,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return vectorizer, model


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def vectorize_batch(vectorizer: TextVectorizer, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [vectorizer.encode(t) for t in texts]
    input_ids = torch.stack([item[0] for item in encoded])
    lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
    return input_ids, lengths


@torch.inference_mode()
def predict_texts(
    model: RNNRegressor,
    vectorizer: TextVectorizer,
    device: torch.device,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    preds: List[float] = []
    for batch in batched(texts, batch_size):
        input_ids, lengths = vectorize_batch(vectorizer, batch)
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        outputs = model(input_ids, lengths).detach().cpu().numpy()
        preds.extend(outputs.tolist())
    return np.asarray(preds, dtype=np.float32)


def maybe_compute_metrics(df: pd.DataFrame, preds: np.ndarray) -> None:
    if "label" not in df.columns:
        return
    label_values = df["label"].to_numpy(dtype=np.float32)
    mae = float(np.mean(np.abs(preds - label_values)))
    acc = float(np.mean(np.abs(preds - label_values) <= 7.5) * 100.0)
    print(f"[metrics] MAE={mae:.3f} | Accuracy@7.5={acc:.2f}%")


def run_inference(args: argparse.Namespace) -> None:
    split_file = DATA_DIR / f"{args.split}.csv"
    if not split_file.exists():
        raise SystemExit(f"Missing data split '{split_file}'. Run the data step first.")

    df = pd.read_csv(split_file)
    if "query" not in df.columns:
        raise SystemExit(f"{split_file} must contain a 'query' column.")

    queries = df["query"].fillna("").astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectorizer, model = load_checkpoint(Path(args.checkpoint), device)
    predictions = predict_texts(model, vectorizer, device, queries, args.batch_size)

    output = df.copy()
    output["prediction"] = predictions
    output_path = args.output or RESULTS_DIR / f"{args.split}_predictions_rnn.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"[+] Wrote {len(output)} predictions to {output_path}")
    maybe_compute_metrics(df, predictions)


def interactive_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectorizer, model = load_checkpoint(Path(args.checkpoint), device)

    def score(batch: List[str]) -> np.ndarray:
        return predict_texts(model, vectorizer, device, batch, args.batch_size)

    if args.text:
        pred = float(score([args.text])[0])
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
                    outputs = score(buffer)
                    for in_text, out_val in zip(buffer, outputs):
                        print(f"• {out_val:.2f} g\t<= {in_text}")
                    buffer.clear()
                else:
                    break
                continue

            buffer.append(text)
            if len(buffer) >= args.batch_size:
                outputs = score(buffer)
                for in_text, out_val in zip(buffer, outputs):
                    print(f"• {out_val:.2f} g\t<= {in_text}")
                buffer.clear()

    except (EOFError, KeyboardInterrupt):
        pass

    if buffer:
        outputs = score(buffer)
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
