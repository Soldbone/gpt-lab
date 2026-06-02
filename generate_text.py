# -*- coding: utf-8 -*-
"""Generate text from a trained mini GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.bpe import BPETokenizer
from src.model import GPTModel
from src.train import generate


ROOT = Path(__file__).resolve().parent


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(path: Path | None) -> Path:
    if path is not None:
        return path

    final_path = ROOT / "checkpoints" / "final_pretrain.pt"
    if final_path.exists():
        return final_path

    candidates = sorted((ROOT / "checkpoints").glob("ckpt_step_*.pt"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found under checkpoints/.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "checkpoints" / "bpe_vocab_3000.json")
    parser.add_argument("--prompt", type=str, default="영화")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = find_checkpoint(args.checkpoint)
    device = pick_device(args.device)

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)

    config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
    }
    model = GPTModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        context_size=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(token_ids.squeeze(0).tolist(), errors="replace")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}, Step: {checkpoint.get('global_step')}")
    print(text.replace("\n", " "))


if __name__ == "__main__":
    main()
