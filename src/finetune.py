# -*- coding: utf-8 -*-
"""Sentiment fine-tuning utilities for NSMC-style data."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Read NSMC TSV files and return train/validation/test rows.

    Returns:
        [{"text": review_text, "label": 0_or_1}, ...]
    """

    def read_nsmc_tsv(path: str | Path | None) -> list[dict]:
        if path is None:
            return []

        rows: list[dict] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                label = row.get("label")
                if not text or label not in {"0", "1"}:
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    def write_tsv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")

    train_rows = read_nsmc_tsv(train_tsv_path)
    test_data = read_nsmc_tsv(test_tsv_path)

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        write_tsv(output_path / "sentiment_train.tsv", train_data)
        write_tsv(output_path / "sentiment_val.tsv", val_data)
        write_tsv(output_path / "sentiment_test.tsv", test_data)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """Dataset that returns padded token ids and one sentiment label."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        text = item["text"]
        label = int(item["label"])

        try:
            token_ids = self.tokenizer.encode(text, add_bos_eos=True)
        except TypeError:
            token_ids = self.tokenizer.encode(text)

        token_ids = token_ids[: self.max_length]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long), label


class GPTForSequenceClassification(nn.Module):
    """GPT backbone with a classification head for sentiment labels."""

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = pad_id
        emb_dim = gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.gpt.embedding(input_ids)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x)

        non_pad = input_ids.ne(self.pad_id)
        last_token_indices = non_pad.long().sum(dim=1).sub(1).clamp(min=0)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_vector = x[batch_indices, last_token_indices]
        logits = self.classifier(self.dropout(sentence_vector))

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(logits, labels.long())
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train one epoch and return (average_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate and return (average_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels)
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, total_correct / total_examples
