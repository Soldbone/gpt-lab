# -*- coding: utf-8 -*-
"""Step 1: train and save a reusable BPE tokenizer for report experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from report_pipeline_common import (
    ROOT,
    compact_ids,
    ensure_data,
    read_text,
    save_json,
    save_token_ids,
    set_seed,
)

from bpe import BPETokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/report_pipeline")
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--lm-char-limit", type=int, default=None)
    parser.add_argument("--train-chars", type=int, default=None)
    parser.add_argument("--val-chars", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifact_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    bpe_dir = artifact_dir / "bpe"
    token_dir = artifact_dir / "tokens"
    bpe_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    paths = ensure_data(lm_char_limit=args.lm_char_limit, val_ratio=args.val_ratio, seed=args.seed)
    train_text = read_text(paths["lm_train"], args.train_chars)
    val_text = read_text(paths["lm_val"], args.val_chars)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(train_text)
    actual_vocab_size = len(tokenizer.token_to_id)

    vocab_path = bpe_dir / f"bpe_vocab_{args.vocab_size}.json"
    tokenizer.save(vocab_path)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    train_ids_path = token_dir / "train_ids.pt"
    val_ids_path = token_dir / "val_ids.pt"
    save_token_ids(train_ids_path, train_ids, paths["lm_train"], len(train_text))
    save_token_ids(val_ids_path, val_ids, paths["lm_val"], len(val_text))

    sample_text = train_text[:80]
    sample_ids = tokenizer.encode(sample_text)
    manifest = {
        "stage": "bpe",
        "artifact_dir": str(artifact_dir),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "vocab_path": str(vocab_path.relative_to(artifact_dir)),
        "train_ids_path": str(train_ids_path.relative_to(artifact_dir)),
        "val_ids_path": str(val_ids_path.relative_to(artifact_dir)),
        "data_paths": paths,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "special_ids": {
            "pad": tokenizer.get_pad_id(),
            "unk": tokenizer.get_unk_id(),
            "bos": tokenizer.get_bos_id(),
            "eos": tokenizer.get_eos_id(),
        },
        "sample": {
            "text": sample_text,
            "ids": compact_ids(sample_ids),
            "decoded": tokenizer.decode(sample_ids),
        },
        "elapsed_sec": time.time() - start_time,
        "seed": args.seed,
    }
    manifest_path = artifact_dir / "bpe_manifest.json"
    save_json(manifest_path, manifest)

    print(f"artifact_dir={artifact_dir}", flush=True)
    print(f"vocab_path={vocab_path}", flush=True)
    print(f"manifest_path={manifest_path}", flush=True)
    print(f"actual_vocab_size={actual_vocab_size}", flush=True)
    print(f"train_tokens={len(train_ids)} val_tokens={len(val_ids)}", flush=True)


if __name__ == "__main__":
    main()
