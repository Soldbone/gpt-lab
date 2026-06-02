# -*- coding: utf-8 -*-
"""Replot saved NSMC losses with epoch on the x-axis."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "nsmc_report"


def read_config() -> dict[str, str]:
    config = {}
    for line in (OUT_DIR / "config.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key] = value
    return config


def main() -> None:
    config = read_config()
    train_tokens = int(config["train_tokens"])
    context_length = int(config["context_length"])
    batch_size = int(config["batch_size"])
    eval_freq = int(config["eval_freq"])

    samples_per_epoch = (train_tokens - context_length + context_length - 1) // context_length
    steps_per_epoch = samples_per_epoch // batch_size

    rows = []
    with (OUT_DIR / "losses.csv").open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_index = int(row["eval_index"])
            global_step = eval_index * eval_freq
            epoch = global_step / steps_per_epoch
            rows.append(
                {
                    "eval_index": eval_index,
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                }
            )

    with (OUT_DIR / "losses_by_epoch.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_index", "global_step", "epoch", "train_loss", "val_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]

    plt.figure(figsize=(10, 5.5))
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NSMC Pretraining Loss by Epoch")
    plt.xlim(0, max(epochs))
    plt.xticks(range(0, int(max(epochs)) + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / "losses_by_epoch.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(OUT_DIR / "losses_by_epoch.csv")
    print(OUT_DIR / "losses_by_epoch.png")


if __name__ == "__main__":
    main()
