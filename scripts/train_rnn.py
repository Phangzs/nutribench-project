#!/usr/bin/env python
"""Train a lightweight 2-layer RNN + LSTM regressor with Optuna (PyTorch)."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    """Lightweight tokenizer: lowercase and split on alphanumerics/punctuation."""
    return re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9\s]", text.lower())


@dataclass
class TextVectorizer:
    vocab: Dict[str, int]
    max_length: int
    pad_idx: int = 0
    unk_idx: int = 1

    @classmethod
    def build(cls, texts: List[str], max_vocab: int = 10000, max_length_cap: int = 128) -> "TextVectorizer":
        tokens = [tokenize(t) for t in texts]
        counter: Counter = Counter()
        for seq in tokens:
            counter.update(seq)

        vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for token, _ in counter.most_common(max_vocab - len(vocab)):
            if token not in vocab:
                vocab[token] = len(vocab)

        observed_max = max((len(seq) for seq in tokens), default=1)
        max_length = max(1, min(max_length_cap, observed_max))
        return cls(vocab=vocab, max_length=max_length)

    def encode(self, text: str) -> Tuple[torch.Tensor, int]:
        tokens = tokenize(text)
        indices = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        if not indices:
            indices = [self.unk_idx]

        length = min(len(indices), self.max_length)
        indices = indices[: self.max_length]
        if len(indices) < self.max_length:
            indices = indices + [self.pad_idx] * (self.max_length - len(indices))

        return torch.tensor(indices, dtype=torch.long), max(1, length)

    def to_dict(self) -> Dict:
        return {
            "vocab": self.vocab,
            "max_length": self.max_length,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TextVectorizer":
        return cls(
            vocab=data["vocab"],
            max_length=int(data["max_length"]),
            pad_idx=int(data.get("pad_idx", 0)),
            unk_idx=int(data.get("unk_idx", 1)),
        )


class RNNDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer: TextVectorizer):
        self.samples: List[Tuple[torch.Tensor, int, float]] = []
        for query, label in zip(dataframe["query"].fillna(""), dataframe["label"]):
            encoded, length = vectorizer.encode(str(query))
            self.samples.append((encoded, length, float(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        return self.samples[idx]


def collate_batch(batch):
    input_ids = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "lengths": lengths, "labels": labels}


class RNNRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn1_units: int,
        rnn2_units: int,
        lstm_units: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn1 = nn.RNN(embedding_dim, rnn1_units, batch_first=True, nonlinearity="relu")
        self.rnn2 = nn.RNN(rnn1_units, rnn2_units, batch_first=True, nonlinearity="relu")
        self.lstm = nn.LSTM(rnn2_units, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_units, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn1(packed)
        packed, _ = self.rnn2(packed)
        padded, _ = pad_packed_sequence(packed, batch_first=True)
        repacked = pack_padded_sequence(self.dropout(padded), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(repacked)
        return self.head(h_n[-1]).squeeze(-1)


def load_split(split: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split '{csv_path}'. Run './run_project.sh data' first.")
    df = pd.read_csv(csv_path)
    required_cols = {"query", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {required_cols}.")
    return df


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(preds - labels)))
    acc = float((np.abs(preds - labels) <= 7.5).mean() * 100.0)
    return {"mae": mae, "accuracy@7.5": acc}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    losses: List[float] = []
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(input_ids, lengths)
            loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(loss.detach().cpu().item())
            preds.append(outputs.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(labels, axis=0)
    metrics = compute_metrics(y_pred, y_true)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_train_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion, optimizer=None)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae']:.3f} "
            f"val_mae={val_metrics['mae']:.3f} "
            f"val_acc@7.5={val_metrics['accuracy@7.5']:.2f}%"
        )

        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = deepcopy(model.state_dict())
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics | {"epoch": epoch}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            print(f"[Early stop] No improvement for {patience} epoch(s).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_train_metrics, best_val_metrics, {"learning_rate": learning_rate}


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    vectorizer: TextVectorizer,
    params: Dict,
    metrics: Dict[str, float],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2)
    with open(checkpoint_dir / "vectorizer.json", "w", encoding="utf-8") as fp:
        json.dump(vectorizer.to_dict(), fp, indent=2)
    with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"[+] Saved checkpoint -> {checkpoint_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stacked RNN + LSTM regressor with Optuna.")
    parser.add_argument("--study-name", default="rnn_optuna", help="Name for the Optuna study/checkpoint folder.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials to run (default: 30).")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs per trial (default: 30).")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on val MAE (default: 3).")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Maximum vocabulary size (default: 10000).")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence length cap (default: 128).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_df = load_split("train")
    val_df = load_split("val")

    vectorizer = TextVectorizer.build(
        texts=train_df["query"].fillna("").astype(str).tolist(),
        max_vocab=args.vocab_size,
        max_length_cap=args.max_len,
    )

    train_dataset = RNNDataset(train_df, vectorizer)
    val_dataset = RNNDataset(val_df, vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    search_space = {
        "embedding_dim": [64, 128, 256],
        "rnn_units": (16, 128),
        "lstm_units": [16, 32, 64],
        "dropout": (0.2, 0.5),
        "learning_rate": (1e-4, 1e-2),
    }

    def objective(trial: optuna.Trial) -> float:
        embedding_dim = trial.suggest_categorical("embedding_dim", search_space["embedding_dim"])
        rnn_units = trial.suggest_int("rnn_units", search_space["rnn_units"][0], search_space["rnn_units"][1])
        lstm_units = trial.suggest_categorical("lstm_units", search_space["lstm_units"])
        dropout = trial.suggest_float("dropout", search_space["dropout"][0], search_space["dropout"][1])
        learning_rate = trial.suggest_float(
            "learning_rate", search_space["learning_rate"][0], search_space["learning_rate"][1], log=True
        )

        rnn2_units = max(1, rnn_units // 2)  # second layer = floor(R/2)

        model = RNNRegressor(
            vocab_size=len(vectorizer.vocab),
            embedding_dim=embedding_dim,
            rnn1_units=rnn_units,
            rnn2_units=rnn2_units,
            lstm_units=lstm_units,
            dropout=dropout,
            pad_idx=vectorizer.pad_idx,
        ).to(device)

        train_metrics, val_metrics, extra = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=learning_rate,
            max_epochs=args.epochs,
            patience=args.patience,
        )

        all_params = {
            "embedding_dim": embedding_dim,
            "rnn_units": rnn_units,
            "rnn2_units": rnn2_units,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "vocab_size": len(vectorizer.vocab),
            "max_length": vectorizer.max_length,
            "pad_idx": vectorizer.pad_idx,
            "unk_idx": vectorizer.unk_idx,
            "trial_number": trial.number,
        }
        metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "extra": extra,
        }

        checkpoint_dir = CHECKPOINT_ROOT / args.study_name / str(trial.number)
        save_checkpoint(checkpoint_dir, model, vectorizer, all_params, metrics)

        trial.set_user_attr("val_mae", val_metrics["mae"])
        return val_metrics["mae"]

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, timeout=None, show_progress_bar=True)

    print("Best MAE:", study.best_value)
    print("Best hyperparameters:", study.best_params)

    trials_csv = CHECKPOINT_ROOT / f"{args.study_name}_trials.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)
    print(f"[+] Saved trial history -> {trials_csv}")


if __name__ == "__main__":
    main()
