# -*- coding: utf-8 -*-
"""Write data-flow examples for the report.

This script documents:
- raw NSMC text examples
- tokenizer input/output examples
- GPTDataset input/target shifting
- DataLoader batch shapes
- GPTModel logits shape
- optional generation sample from a checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import download_data  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from dataset import GPTDataset, create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
from train import generate, load_checkpoint  # noqa: E402


OUT_DIR = ROOT / "outputs" / "data_flow"
VOCAB_PATHS = [
    ROOT / "outputs" / "local_sweep_v3000_e100" / "final" / "bpe_vocab_3000.json",
    ROOT / "outputs" / "local_sweep_full" / "final" / "bpe_vocab_500.json",
    ROOT / "outputs" / "nsmc_report" / "bpe_vocab_300.json",
]
CHECKPOINT_PATHS = [
    ROOT
    / "outputs"
    / "local_sweep_v3000_e100"
    / "final"
    / "final_ctx128_layers4_lr3e-4"
    / "best_checkpoint.pt",
    ROOT
    / "outputs"
    / "local_sweep_full"
    / "final"
    / "final_ctx128_layers4_lr3e-4"
    / "best_checkpoint.pt",
]


def pick_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def compact_ids(ids: list[int], n: int = 24) -> list[int | str]:
    if len(ids) <= n:
        return ids
    half = n // 2
    return [*ids[:half], "...", *ids[-half:]]


def safe_decode(tokenizer: BPETokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids)
    except UnicodeDecodeError:
        byte_chunks = []
        for token_id in ids:
            token = tokenizer.id_to_token.get(token_id)
            if isinstance(token, bytes):
                byte_chunks.append(token)
        return b"".join(byte_chunks).decode("utf-8", errors="replace")


def make_tokenizer(train_text: str) -> tuple[BPETokenizer, Path | None]:
    vocab_path = pick_existing(VOCAB_PATHS)
    tokenizer = BPETokenizer(vocab_size=3000)
    if vocab_path is not None:
        tokenizer.load(vocab_path)
        tokenizer.vocab_size = len(tokenizer.token_to_id)
        return tokenizer, vocab_path

    tokenizer.train(train_text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = OUT_DIR / "bpe_vocab_3000.json"
    tokenizer.save(vocab_path)
    return tokenizer, vocab_path


def build_model_config(vocab_size: int, checkpoint_path: Path | None) -> dict:
    if checkpoint_path and "local_sweep_full" in str(checkpoint_path):
        return {
            "vocab_size": vocab_size,
            "context_length": 128,
            "emb_dim": 192,
            "n_heads": 4,
            "n_layers": 4,
            "drop_rate": 0.1,
            "qkv_bias": False,
        }
    return {
        "vocab_size": vocab_size,
        "context_length": 128,
        "emb_dim": 192,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = download_data.main()
    train_text = Path(paths["lm_train"]).read_text(encoding="utf-8")
    val_text = Path(paths["lm_val"]).read_text(encoding="utf-8")

    tokenizer, vocab_path = make_tokenizer(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    context_length = 128
    batch_size = 4
    dataset = GPTDataset(train_ids, context_length=context_length, stride=context_length)
    input_ids, target_ids = dataset[0]
    loader = create_dataloader(
        train_ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=context_length,
        shuffle=False,
        drop_last=True,
    )
    batch_inputs, batch_targets = next(iter(loader))

    checkpoint_path = pick_existing(CHECKPOINT_PATHS)
    config = build_model_config(len(tokenizer.token_to_id), checkpoint_path)
    model = GPTModel(config)
    checkpoint_info = None
    if checkpoint_path is not None:
        try:
            epoch, step = load_checkpoint(model, None, str(checkpoint_path), torch.device("cpu"))
            checkpoint_info = {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "epoch": epoch,
                "global_step": step,
            }
        except RuntimeError as exc:
            checkpoint_info = {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "load_error": str(exc),
            }

    model.eval()
    with torch.no_grad():
        logits = model(batch_inputs)
        predicted_ids = torch.argmax(logits[0], dim=-1).tolist()

    prompt = "이 영화"
    prompt_ids = tokenizer.encode(prompt)
    generation = None
    if prompt_ids:
        idx = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            generated_ids = generate(
                model,
                idx,
                max_new_tokens=40,
                context_size=config["context_length"],
                temperature=0.8,
                top_k=min(40, len(tokenizer.token_to_id)),
            )[0].tolist()
        generation = {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "generated_ids": compact_ids(generated_ids, 40),
            "generated_text": safe_decode(tokenizer, generated_ids),
        }

    sample_texts = train_text.splitlines()[:5]
    token_example_text = sample_texts[0] if sample_texts else train_text[:80]
    token_example_ids = tokenizer.encode(token_example_text)

    report = {
        "data": {
            "train_path": str(Path(paths["lm_train"]).relative_to(ROOT)),
            "val_path": str(Path(paths["lm_val"]).relative_to(ROOT)),
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "train_tokens": len(train_ids),
            "val_tokens": len(val_ids),
            "sample_raw_reviews": sample_texts,
        },
        "tokenizer": {
            "type": "UTF-8 byte-level BPE",
            "vocab_path": str(vocab_path.relative_to(ROOT)) if vocab_path else None,
            "actual_vocab_size": len(tokenizer.token_to_id),
            "special_tokens": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
            "example_text": token_example_text,
            "example_token_ids": compact_ids(token_example_ids),
            "example_decoded_text": safe_decode(tokenizer, token_example_ids),
        },
        "dataset": {
            "task": "next-token prediction",
            "context_length": context_length,
            "stride": context_length,
            "num_train_samples": len(dataset),
            "input_ids_shape": list(input_ids.shape),
            "target_ids_shape": list(target_ids.shape),
            "input_ids_example": compact_ids(input_ids.tolist()),
            "target_ids_example": compact_ids(target_ids.tolist()),
            "input_decoded": safe_decode(tokenizer, input_ids.tolist()),
            "target_decoded": safe_decode(tokenizer, target_ids.tolist()),
        },
        "batch": {
            "batch_size": batch_size,
            "input_batch_shape": list(batch_inputs.shape),
            "target_batch_shape": list(batch_targets.shape),
        },
        "model_io": {
            "config": config,
            "checkpoint": checkpoint_info,
            "logits_shape": list(logits.shape),
            "prediction_rule": "argmax(logits, dim=-1) or sampling after temperature/top-k",
            "predicted_ids_first_sample": compact_ids(predicted_ids),
            "predicted_text_first_sample": safe_decode(tokenizer, predicted_ids),
        },
        "generation": generation,
    }

    (OUT_DIR / "data_flow.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Data Flow Report",
        "",
        "## Raw Data",
        "",
        f"- train path: `{report['data']['train_path']}`",
        f"- validation path: `{report['data']['val_path']}`",
        f"- train chars: `{report['data']['train_chars']}`",
        f"- validation chars: `{report['data']['val_chars']}`",
        f"- train tokens: `{report['data']['train_tokens']}`",
        f"- validation tokens: `{report['data']['val_tokens']}`",
        "",
        "Sample raw reviews:",
        "",
    ]
    for text in sample_texts:
        lines.append(f"- {text}")

    lines.extend(
        [
            "",
            "## Tokenizer",
            "",
            f"- type: `{report['tokenizer']['type']}`",
            f"- vocab path: `{report['tokenizer']['vocab_path']}`",
            f"- vocab size: `{report['tokenizer']['actual_vocab_size']}`",
            "",
            "| text | token ids | decoded |",
            "| --- | --- | --- |",
            "| {text} | `{ids}` | {decoded} |".format(
                text=report["tokenizer"]["example_text"].replace("|", "\\|"),
                ids=report["tokenizer"]["example_token_ids"],
                decoded=report["tokenizer"]["example_decoded_text"].replace("|", "\\|"),
            ),
            "",
            "## Dataset Shift",
            "",
            "| item | shape | example ids | decoded |",
            "| --- | --- | --- | --- |",
            "| input | `{}` | `{}` | {} |".format(
                report["dataset"]["input_ids_shape"],
                report["dataset"]["input_ids_example"],
                report["dataset"]["input_decoded"].replace("|", "\\|"),
            ),
            "| target | `{}` | `{}` | {} |".format(
                report["dataset"]["target_ids_shape"],
                report["dataset"]["target_ids_example"],
                report["dataset"]["target_decoded"].replace("|", "\\|"),
            ),
            "",
            "## Batch And Model IO",
            "",
            f"- input batch shape: `{report['batch']['input_batch_shape']}`",
            f"- target batch shape: `{report['batch']['target_batch_shape']}`",
            f"- logits shape: `{report['model_io']['logits_shape']}`",
            f"- checkpoint: `{report['model_io']['checkpoint']}`",
            "",
            "The model outputs logits with shape `(batch, context_length, vocab_size)`. "
            "During training, cross entropy compares every position's logits with the shifted target token.",
        ]
    )

    if generation is not None:
        lines.extend(
            [
                "",
                "## Generation Example",
                "",
                f"- prompt: `{generation['prompt']}`",
                f"- prompt ids: `{generation['prompt_ids']}`",
                f"- generated ids: `{generation['generated_ids']}`",
                f"- generated text: {generation['generated_text']}",
            ]
        )

    (OUT_DIR / "data_flow.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "data_flow.md")
    print(OUT_DIR / "data_flow.json")


if __name__ == "__main__":
    main()
