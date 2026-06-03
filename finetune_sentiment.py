# -*- coding: utf-8 -*-
"""Run NSMC sentiment fine-tuning with a pretrained mini GPT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.bpe import BPETokenizer
from src.finetune import (
    GPTForSequenceClassification,
    ReviewSentimentDataset,
    evaluate_sentiment,
    train_epoch_sentiment,
)
from src.model import GPTModel
from src.reproducibility import DEFAULT_SEED, make_torch_generator, seed_worker, set_global_seed
from pretrain import DEFAULT_HYPERPARAMS, pick_device

ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            text = (item.get("text") or "").strip()
            label = item.get("label")
            if text and label in {0, 1, "0", "1"}:
                rows.append({"text": text, "label": int(label)})
    return rows


def default_checkpoint_path() -> Path:
    candidates = [
        ROOT / "checkpoints" / "final_pretrain.pt",
        ROOT / "checkpoints" / "pretrain_from_scratch" / "final_pretrain.pt",
    ]
    candidates.extend(
        sorted(
            (ROOT / "checkpoints" / "pretrain_from_scratch").glob("ckpt_step_*.pt"),
            reverse=True,
        )
    )
    candidates.extend(sorted((ROOT / "checkpoints").glob("ckpt_step_*.pt"), reverse=True))
    for path in candidates:
        if path.exists():
            return path
    return ROOT / "checkpoints" / "final_pretrain.pt"


def default_tokenizer_path() -> Path:
    candidates = [
        ROOT / "checkpoints" / f"bpe_vocab_{DEFAULT_HYPERPARAMS['vocab_size']}.json",
        DEFAULT_HYPERPARAMS["output_dir"] / f"bpe_vocab_{DEFAULT_HYPERPARAMS['vocab_size']}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def infer_config_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    fallback: argparse.Namespace,
) -> dict:
    token_weight = state_dict["embedding.token_embedding_layer.weight"]
    pos_weight = state_dict["embedding.position_layer.weight"]
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("trf_blocks.") and key.split(".")[1].isdigit()
    }
    return {
        "vocab_size": token_weight.shape[0],
        "context_length": pos_weight.shape[0],
        "emb_dim": token_weight.shape[1],
        "n_heads": fallback.n_heads,
        "n_layers": max(layer_ids) + 1 if layer_ids else fallback.n_layers,
        "drop_rate": fallback.drop_rate,
        "qkv_bias": False,
    }


def load_pretrained_gpt(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> GPTModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    config = infer_config_from_state_dict(state_dict, args)
    gpt = GPTModel(config)
    gpt.load_state_dict(state_dict)
    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    print(
        "Backbone config: "
        f"vocab={config['vocab_size']}, context={config['context_length']}, "
        f"emb={config['emb_dim']}, layers={config['n_layers']}, heads={config['n_heads']}"
    )
    return gpt


def build_random_gpt(args: argparse.Namespace, tokenizer: BPETokenizer) -> GPTModel:
    config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
    }
    print("No pretrained checkpoint loaded; using random GPT backbone.")
    return GPTModel(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "nsmc_sentiment_train.jsonl")
    parser.add_argument("--val-path", type=Path, default=ROOT / "data" / "nsmc_sentiment_val.jsonl")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "nsmc_sentiment_test.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=default_tokenizer_path())
    parser.add_argument("--pretrained-checkpoint", type=Path, default=default_checkpoint_path())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "finetune_sentiment")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_HYPERPARAMS["n_heads"])
    parser.add_argument("--n-layers", type=int, default=DEFAULT_HYPERPARAMS["n_layers"])
    parser.add_argument("--emb-dim", type=int, default=DEFAULT_HYPERPARAMS["emb_dim"])
    parser.add_argument("--context-length", type=int, default=DEFAULT_HYPERPARAMS["context_length"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = pick_device(args.device)
    tokenizer = BPETokenizer(vocab_size=DEFAULT_HYPERPARAMS["vocab_size"])
    tokenizer.load(args.tokenizer_path)

    train_data = read_jsonl(args.train_path, args.max_train_samples)
    val_data = read_jsonl(args.val_path, args.max_val_samples)
    test_data = read_jsonl(args.test_path, args.max_test_samples)

    train_dataset = ReviewSentimentDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = ReviewSentimentDataset(val_data, tokenizer, max_length=args.max_length)
    test_dataset = ReviewSentimentDataset(test_data, tokenizer, max_length=args.max_length)
    train_generator = make_torch_generator(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    if args.pretrained_checkpoint.exists():
        gpt = load_pretrained_gpt(args.pretrained_checkpoint, args, device)
    elif args.allow_random_init:
        gpt = build_random_gpt(args, tokenizer)
    else:
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {args.pretrained_checkpoint}. "
            "Pass --pretrained-checkpoint or --allow-random-init."
        )

    model = GPTForSequenceClassification(
        gpt_model=gpt,
        num_labels=2,
        drop_rate=args.drop_rate,
        pad_id=tokenizer.get_pad_id(),
    ).to(device)

    if args.freeze_backbone:
        for param in model.gpt.parameters():
            param.requires_grad = False
        print("Backbone frozen: training classifier head only.")

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Data: train={len(train_data):,}, val={len(val_data):,}, test={len(test_data):,}")
    print(f"Batches: train={len(train_loader):,}, val={len(val_loader):,}, test={len(test_loader):,}")

    best_val_acc = -1.0
    best_path = args.output_dir / "best_sentiment.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch_sentiment(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_sentiment(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}: "
            f"train loss {train_loss:.4f}, train acc {train_acc:.4f}, "
            f"val loss {val_loss:.4f}, val acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone_config": model.gpt.config,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "args": serializable_args(args),
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate_sentiment(model, test_loader, device)
    print("\n=== Fine-tuning summary ===")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
