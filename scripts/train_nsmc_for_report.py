# -*- coding: utf-8 -*-
"""Full NSMC pretraining run for report artifacts.

Outputs:
- outputs/nsmc_report/losses.csv
- outputs/nsmc_report/losses.png
- outputs/nsmc_report/final_checkpoint.pt
- outputs/nsmc_report/config.txt
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import download_data  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
import train  # noqa: E402


OUT_DIR = ROOT / "outputs" / "nsmc_report"

CONFIG = {
    "vocab_size": 300,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

BATCH_SIZE = 16
NUM_EPOCHS = 5
EVAL_FREQ = 100
EVAL_ITER = 20
LR = 3e-4
WEIGHT_DECAY = 0.1
START_CONTEXT = "이 영화"


def ensure_data() -> tuple[str, str]:
    paths = download_data.main()
    return paths["lm_train"], paths["lm_val"]


def save_loss_artifacts(train_losses: list[float], val_losses: list[float]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "losses.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_index", "train_loss", "val_loss"])
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, train_loss, val_loss])

    plt.figure(figsize=(10, 5.5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Evaluation step")
    plt.ylabel("Loss")
    plt.title("NSMC Pretraining Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_DIR / "losses.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=prepare_data", flush=True)
    train_path, val_path = ensure_data()
    print("stage=read_text", flush=True)
    train_text = Path(train_path).read_text(encoding="utf-8")
    val_text = Path(val_path).read_text(encoding="utf-8")

    print("stage=train_bpe", flush=True)
    tokenizer = BPETokenizer(vocab_size=CONFIG["vocab_size"])
    tokenizer.train(train_text)

    print("stage=encode_text", flush=True)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    print("stage=create_loaders", flush=True)
    train_loader = create_dataloader(
        train_ids,
        context_length=CONFIG["context_length"],
        batch_size=BATCH_SIZE,
        stride=CONFIG["context_length"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=CONFIG["context_length"],
        batch_size=BATCH_SIZE,
        stride=CONFIG["context_length"],
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage=create_model device={device}", flush=True)
    model = GPTModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    config_text = "\n".join(
        [
            f"device={device}",
            f"train_chars={len(train_text)}",
            f"val_chars={len(val_text)}",
            f"train_tokens={len(train_ids)}",
            f"val_tokens={len(val_ids)}",
            f"batch_size={BATCH_SIZE}",
            f"num_epochs={NUM_EPOCHS}",
            f"eval_freq={EVAL_FREQ}",
            f"eval_iter={EVAL_ITER}",
            f"lr={LR}",
            f"weight_decay={WEIGHT_DECAY}",
            *[f"{key}={value}" for key, value in CONFIG.items()],
        ]
    )
    (OUT_DIR / "config.txt").write_text(config_text + "\n", encoding="utf-8")

    # For report loss curves, avoid random sample generation overhead/noise.
    train.generate_and_print_sample = lambda *args, **kwargs: None

    print("stage=train_model", flush=True)
    train_losses, val_losses = train.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        start_context=START_CONTEXT,
        tokenizer=tokenizer,
    )

    print("stage=save_artifacts", flush=True)
    save_loss_artifacts(train_losses, val_losses)
    train.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=NUM_EPOCHS,
        global_step=len(train_loader) * NUM_EPOCHS,
        path=str(OUT_DIR / "final_checkpoint.pt"),
    )

    print(f"saved: {OUT_DIR / 'losses.csv'}")
    print(f"saved: {OUT_DIR / 'losses.png'}")
    print(f"saved: {OUT_DIR / 'final_checkpoint.pt'}")
    print(f"final_train_loss={train_losses[-1] if train_losses else None}")
    print(f"final_val_loss={val_losses[-1] if val_losses else None}")


if __name__ == "__main__":
    main()
