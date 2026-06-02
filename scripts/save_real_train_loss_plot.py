# -*- coding: utf-8 -*-
"""Create a loss plot from real NSMC data and save it to outputs/."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpe import BPETokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
import train  # noqa: E402


def main() -> None:
    train_text = (ROOT / "data" / "nsmc_lm_train.txt").read_text(encoding="utf-8")[:30_000]
    val_text = (ROOT / "data" / "nsmc_lm_val.txt").read_text(encoding="utf-8")[:8_000]

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(train_text)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    config = {
        "vocab_size": 300,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    train.generate_and_print_sample = lambda *args, **kwargs: None
    train_losses, val_losses = train.train_model(
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

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "real_train_losses.png"

    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Evaluation step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("NSMC Training / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(out_path)
    print("train_losses=", train_losses)
    print("val_losses=", val_losses)


if __name__ == "__main__":
    main()
