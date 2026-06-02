# -*- coding: utf-8 -*-
"""Run mini GPT pretraining on the prepared NSMC language-modeling text."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.train import load_checkpoint, save_checkpoint, train_model

ROOT = Path(__file__).resolve().parent

DEFAULT_HYPERPARAMS = {
    "train_path": ROOT / "data" / "nsmc_lm_train.txt",
    "val_path": ROOT / "data" / "nsmc_lm_val.txt",
    "tokenizer_path": ROOT / "checkpoints" / "bpe_vocab_3000.json",
    "output_dir": ROOT / "checkpoints",
    "vocab_size": 3000, #vocab_size=3000, 
    "context_length": 128, #context_length=128
    "stride": None,
    "emb_dim": 64,#emb_dim: 64, 128, 192
    "n_heads": 4,
    "n_layers": 1,#n_layers: 1, 2, 4  default:2
    "drop_rate": 0.0,#drop_rate: 0.0, 0.1, 0.2 default:0.1
    "batch_size": 8, #batch_size: 2, 4, 8, 16 default:8
    "num_workers": 0,
    "epochs": 50,
    "lr": 1e-4,#learning_rate: 1e-4, 3e-4, 5e-4 default:3e-4
    "weight_decay": 0.1,
    "eval_freq": 500,
    "eval_iter": 20,
    "ckpt_freq": 1000,
    "resume": None,
    "device": "auto",
    "start_context": "영화",
    "max_train_chars": None,
    "max_val_chars": None,
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_text(path: Path, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if max_chars is not None:
        return text[:max_chars]
    return text


def build_tokenizer(args: argparse.Namespace, train_text: str) -> BPETokenizer:
    tokenizer_path = args.tokenizer_path
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    if tokenizer_path.exists() and not args.retrain_tokenizer:
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer: {tokenizer_path}")
        return tokenizer

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training tokenizer on {len(train_text):,} characters...")
    tokenizer.train(train_text)
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer: {tokenizer_path}")
    return tokenizer


def parse_args() -> argparse.Namespace:
    defaults = DEFAULT_HYPERPARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=defaults["train_path"])
    parser.add_argument("--val-path", type=Path, default=defaults["val_path"])
    parser.add_argument("--tokenizer-path", type=Path, default=defaults["tokenizer_path"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=defaults["vocab_size"])
    parser.add_argument("--context-length", type=int, default=defaults["context_length"])
    parser.add_argument("--stride", type=int, default=defaults["stride"])
    parser.add_argument("--emb-dim", type=int, default=defaults["emb_dim"])
    parser.add_argument("--n-heads", type=int, default=defaults["n_heads"])
    parser.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    parser.add_argument("--drop-rate", type=float, default=defaults["drop_rate"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--eval-freq", type=int, default=defaults["eval_freq"])
    parser.add_argument("--eval-iter", type=int, default=defaults["eval_iter"])
    parser.add_argument("--ckpt-freq", type=int, default=defaults["ckpt_freq"])
    parser.add_argument("--resume", type=Path, default=defaults["resume"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--start-context", type=str, default=defaults["start_context"])
    parser.add_argument("--max-train-chars", type=int, default=defaults["max_train_chars"])
    parser.add_argument("--max-val-chars", type=int, default=defaults["max_val_chars"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def print_summary(args: argparse.Namespace, train_losses: list[float], val_losses: list[float]) -> None:
    print("\n=== Training summary ===")
    print(
        "Hyperparameters: "
        f"vocab={args.vocab_size}, context={args.context_length}, "
        f"emb={args.emb_dim}, layers={args.n_layers}, heads={args.n_heads}, "
        f"drop={args.drop_rate}, batch={args.batch_size}, epochs={args.epochs}, lr={args.lr}"
    )
    if not train_losses or not val_losses:
        print("No evaluation losses were recorded. Lower --eval-freq or train longer.")
        return

    last_train = train_losses[-1]
    last_val = val_losses[-1]
    best_val = min(val_losses)
    print(f"Eval points: {len(train_losses)}")
    print(f"Final train loss: {last_train:.3f}")
    print(f"Final val loss: {last_val:.3f}")
    print(f"Best val loss: {best_val:.3f}")
    print(f"Final val perplexity: {math.exp(last_val):.2f}")
    print(f"Best val perplexity: {math.exp(best_val):.2f}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.vocab_size = min(args.vocab_size, 300)
        args.tokenizer_path = args.output_dir / f"bpe_vocab_{args.vocab_size}_smoke.json"
        args.max_train_chars = args.max_train_chars or 3_000
        args.max_val_chars = args.max_val_chars or 1_000
        args.context_length = min(args.context_length, 32)
        args.emb_dim = min(args.emb_dim, 32)
        args.n_layers = 1
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.eval_freq = min(args.eval_freq, 20)
        args.eval_iter = min(args.eval_iter, 1)
        args.ckpt_freq = 0

    train_text = read_text(args.train_path, args.max_train_chars)
    val_text = read_text(args.val_path, args.max_val_chars)
    tokenizer = build_tokenizer(args, train_text)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    if len(train_ids) <= args.context_length or len(val_ids) <= args.context_length:
        raise ValueError("Not enough tokens for the chosen context length.")

    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
    }
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    device = pick_device(args.device)
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from {args.resume}: epoch={start_epoch}, step={global_step}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Tokenizer vocab: {config['vocab_size']}")
    print(f"Model params: {num_params:,}")
    print(f"Batches: train={len(train_loader):,}, val={len(val_loader):,}")

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
        ckpt_freq=args.ckpt_freq if args.ckpt_freq > 0 else None,
        start_epoch=start_epoch,
        global_step=global_step,
        checkpoint_dir=args.output_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "final_pretrain.pt"
    final_step = global_step + len(train_loader) * args.epochs
    save_checkpoint(model, optimizer, start_epoch + args.epochs, final_step, str(final_path))
    print(f"Saved final checkpoint: {final_path}")
    print_summary(args, train_losses, val_losses)


if __name__ == "__main__":
    main()
