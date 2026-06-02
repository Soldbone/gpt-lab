# -*- coding: utf-8 -*-
"""Shared helpers for the report-oriented Colab pipeline."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import download_data  # noqa: E402
from bpe import BPETokenizer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_data(lm_char_limit: int | None = None, val_ratio: float = 0.08, seed: int = 42) -> dict[str, str]:
    os.makedirs(download_data.DATA_DIR, exist_ok=True)
    download_data._download(download_data.NSMC_TRAIN_URL, download_data.RAW_TRAIN_PATH)
    download_data._download(download_data.NSMC_TEST_URL, download_data.RAW_TEST_PATH)
    kwargs = {"val_ratio": val_ratio, "seed": seed}
    if lm_char_limit is not None and lm_char_limit > 0:
        kwargs["lm_char_limit"] = lm_char_limit
    return download_data.prepare_data(**kwargs)


def read_text(path: str | Path, char_limit: int | None = None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if char_limit is not None and char_limit > 0:
        return text[:char_limit]
    return text


def artifact_path(artifact_dir: str | Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(artifact_dir) / path


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_tokenizer(vocab_path: str | Path) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(vocab_path)
    tokenizer.vocab_size = len(tokenizer.token_to_id)
    return tokenizer


def save_token_ids(path: str | Path, ids: list[int], source_path: str | Path, char_count: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ids,
            "source_path": str(source_path),
            "char_count": char_count,
            "token_count": len(ids),
        },
        path,
    )


def load_token_ids(path: str | Path) -> list[int]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "ids" in payload:
        return list(payload["ids"])
    return list(payload)


def compact_ids(ids: Iterable[int], max_items: int = 32) -> list[int | str]:
    values = list(ids)
    if len(values) <= max_items:
        return values
    half = max_items // 2
    return values[:half] + ["..."] + values[-half:]


def safe_decode(tokenizer: BPETokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids)
    except UnicodeDecodeError:
        pieces: list[bytes] = []
        for token_id in ids:
            token = tokenizer.id_to_token.get(int(token_id))
            if isinstance(token, bytes):
                pieces.append(token)
        return b"".join(pieces).decode("utf-8", errors="replace")


def build_epoch_summary(metrics: list[dict]) -> list[dict]:
    rows: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    prev_val = None

    for row in metrics:
        if row.get("phase") != "epoch_end":
            continue
        epoch = int(row["epoch"])
        train_loss = float(row["train_loss"])
        val_loss = float(row["val_loss"])
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
        rows.append(
            {
                "epoch": epoch,
                "global_step": int(row["global_step"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "generalization_gap": val_loss - train_loss,
                "val_loss_delta": "" if prev_val is None else val_loss - prev_val,
                "best_val_loss_so_far": best_val,
                "best_epoch_so_far": best_epoch,
                "lr": float(row["lr"]),
                "elapsed_sec": float(row["elapsed_sec"]),
            }
        )
        prev_val = val_loss
    return rows


def write_epoch_markdown(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return

    lines = [
        "| epoch | step | train loss | val loss | gap | val delta | best val so far | best epoch | lr | elapsed sec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row["val_loss_delta"]
        delta_text = "" if delta == "" else f"{delta:.4f}"
        lines.append(
            "| {epoch} | {global_step} | {train_loss:.4f} | {val_loss:.4f} | "
            "{generalization_gap:.4f} | {delta_text} | {best_val_loss_so_far:.4f} | "
            "{best_epoch_so_far} | {lr:.8f} | {elapsed_sec:.1f} |".format(
                delta_text=delta_text,
                **row,
            )
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
