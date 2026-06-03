# -*- coding: utf-8 -*-
"""Train and save only the NSMC byte-level BPE vocabulary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pretrain import DEFAULT_HYPERPARAMS
from src.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_HYPERPARAMS["train_path"],
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_HYPERPARAMS["vocab_size"])
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_HYPERPARAMS["max_train_chars"])
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train_path.resolve()
    if args.output_path is None:
        output_path = (
            DEFAULT_HYPERPARAMS["output_dir"] / f"bpe_vocab_{args.vocab_size}.json"
        ).resolve()
    else:
        output_path = args.output_path.resolve()

    overwrite = not args.no_overwrite
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Remove --no-overwrite or choose --output-path."
        )

    text = train_path.read_text(encoding="utf-8")
    if args.max_chars is not None:
        text = text[: args.max_chars]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Tokenizer mode: train-from-scratch", flush=True)
    print(f"Train path: {train_path}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    print(f"Vocab size: {args.vocab_size:,}", flush=True)
    print(f"Train chars: {len(text):,}", flush=True)
    print(f"Train bytes: {len(text.encode('utf-8')):,}", flush=True)
    if output_path.exists() and overwrite:
        print(f"Overwrite existing tokenizer: {output_path}", flush=True)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    started_at = time.perf_counter()
    tokenizer.train(text, progress_interval=args.progress_interval)
    elapsed_seconds = time.perf_counter() - started_at
    tokenizer.save(output_path)

    metadata = {
        "train_path": str(train_path),
        "output_path": str(output_path),
        "vocab_size": args.vocab_size,
        "train_chars": len(text),
        "train_bytes": len(text.encode("utf-8")),
        "id_to_token_count": len(tokenizer.id_to_token),
        "merge_count": len(tokenizer.merges),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved tokenizer: {output_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)
    print(f"Elapsed: {elapsed_seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
