# -*- coding: utf-8 -*-
"""Train briefly on real text data and display the loss graph."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from download_data import LM_TRAIN_PATH, LM_VAL_PATH, main as download_data  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
import train as train_utils  # noqa: E402


def load_or_create_text() -> tuple[str, str]:
    train_path = Path(LM_TRAIN_PATH)
    val_path = Path(LM_VAL_PATH)

    if not train_path.exists() or not val_path.exists():
        download_data()

    train_text = train_path.read_text(encoding="utf-8")
    val_text = val_path.read_text(encoding="utf-8")
    return train_text, val_text


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_text, val_text = load_or_create_text()
    train_text = train_text[:30_000]
    val_text = val_text[:8_000]

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(train_text)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": 32,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    train_loader = create_dataloader(
        train_ids,
        context_length=config["context_length"],
        batch_size=8,
        stride=config["context_length"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=config["context_length"],
        batch_size=8,
        stride=config["context_length"],
        shuffle=False,
        drop_last=False,
    )

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    train_utils.generate_and_print_sample = lambda *args, **kwargs: None
    train_losses, val_losses = train_utils.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
        eval_freq=10,
        eval_iter=5,
        start_context="이 영화",
        tokenizer=tokenizer,
    )

    train_utils.plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
