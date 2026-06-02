# -*- coding: utf-8 -*-
"""Step 3: load saved BPE/model artifacts and run evaluation/generation tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from report_pipeline_common import (  # noqa: E402
    ROOT,
    artifact_path,
    compact_ids,
    load_json,
    load_token_ids,
    load_tokenizer,
    safe_decode,
    save_json,
    save_token_ids,
    write_csv,
)
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
import train as train_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="outputs/report_pipeline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def default_checkpoint(artifact_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        runs_dir = artifact_dir / "runs"
        candidates = sorted(runs_dir.glob("*/best_checkpoint.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No best_checkpoint.pt found. Pass --run-name or --checkpoint.")
        return candidates[0]
    return artifact_dir / "runs" / run_name / "best_checkpoint.pt"


def read_test_text(manifest: dict) -> str:
    test_path = Path(manifest["data_paths"]["sentiment_test"])
    texts: list[str] = []
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text:
                texts.append(text)
    return "\n".join(texts)


def get_split_ids(artifact_dir: Path, manifest: dict, tokenizer, split: str) -> list[int]:
    if split == "train":
        return load_token_ids(artifact_path(artifact_dir, manifest["train_ids_path"]))
    if split == "val":
        return load_token_ids(artifact_path(artifact_dir, manifest["val_ids_path"]))
    if split == "test":
        test_ids_path = artifact_dir / "tokens" / "test_ids.pt"
        if test_ids_path.exists():
            return load_token_ids(test_ids_path)
        test_text = read_test_text(manifest)
        test_ids = tokenizer.encode(test_text)
        save_token_ids(test_ids_path, test_ids, manifest["data_paths"]["sentiment_test"], len(test_text))
        return test_ids
    raise ValueError(f"unknown split: {split}")


def evaluate_split(
    model: GPTModel,
    ids: list[int],
    split: str,
    context_length: int,
    batch_size: int,
    device: torch.device,
    eval_batches: int,
) -> dict:
    loader = create_dataloader(
        ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=context_length,
        shuffle=False,
        drop_last=False,
    )
    loss = train_utils.calc_loss_loader(loader, model, device, num_batches=eval_batches)
    return {
        "split": split,
        "loss": loss,
        "num_batches": len(loader),
        "eval_batches": min(eval_batches, len(loader)),
        "token_count": len(ids),
    }


def inspect_io(
    model: GPTModel,
    ids: list[int],
    tokenizer,
    context_length: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    loader = create_dataloader(
        ids,
        context_length=context_length,
        batch_size=batch_size,
        stride=context_length,
        shuffle=False,
        drop_last=False,
    )
    input_batch, target_batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        logits = model(input_batch.to(device))
    pred_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
    return {
        "input_shape": list(input_batch.shape),
        "target_shape": list(target_batch.shape),
        "logits_shape": list(logits.shape),
        "first_input_ids": compact_ids(input_batch[0].tolist()),
        "first_target_ids": compact_ids(target_batch[0].tolist()),
        "first_prediction_ids": compact_ids(pred_ids),
        "first_input_text": safe_decode(tokenizer, input_batch[0].tolist()),
        "first_target_text": safe_decode(tokenizer, target_batch[0].tolist()),
        "first_prediction_text": safe_decode(tokenizer, pred_ids),
    }


def generate_samples(
    model: GPTModel,
    tokenizer,
    prompts: list[str],
    context_length: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[dict]:
    rows: list[dict] = []
    model.eval()
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        idx = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = train_utils.generate(
                model,
                idx,
                max_new_tokens=max_new_tokens,
                context_size=context_length,
                temperature=temperature,
                top_k=min(top_k, len(tokenizer.token_to_id)),
                eos_id=tokenizer.get_eos_id(),
            )
        output_ids = output[0].detach().cpu().tolist()
        rows.append(
            {
                "prompt": prompt,
                "prompt_ids": compact_ids(input_ids),
                "generated_ids": compact_ids(output_ids),
                "generated_text": safe_decode(tokenizer, output_ids),
                "temperature": temperature,
                "top_k": min(top_k, len(tokenizer.token_to_id)),
            }
        )
    return rows


def write_generation_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "prompt_ids", "generated_ids", "generated_text", "temperature", "top_k"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, report: dict) -> None:
    lines = [
        "# Loaded Model Test Report",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- tokenizer: `{report['tokenizer_path']}`",
        f"- device: `{report['device']}`",
        f"- checkpoint epoch: `{report['checkpoint_epoch']}`",
        f"- checkpoint step: `{report['checkpoint_step']}`",
        f"- best val loss in checkpoint: `{report['best_val_loss']}`",
        "",
        "## Split Loss",
        "",
        "| split | loss | token count | eval batches | total batches |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["split_metrics"]:
        lines.append(
            f"| {row['split']} | {row['loss']:.4f} | {row['token_count']} | "
            f"{row['eval_batches']} | {row['num_batches']} |"
        )

    io = report["io"]
    lines.extend(
        [
            "",
            "## Input / Output Check",
            "",
            f"- input shape: `{io['input_shape']}`",
            f"- target shape: `{io['target_shape']}`",
            f"- logits shape: `{io['logits_shape']}`",
            f"- first input ids: `{io['first_input_ids']}`",
            f"- first target ids: `{io['first_target_ids']}`",
            f"- first prediction ids: `{io['first_prediction_ids']}`",
            "",
            "## Generations",
            "",
        ]
    )
    for row in report["generations"]:
        lines.append(f"### Prompt: {row['prompt']}")
        lines.append("")
        lines.append(row["generated_text"].replace("\n", " "))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = (ROOT / args.artifact_dir).resolve() if not Path(args.artifact_dir).is_absolute() else Path(args.artifact_dir)
    manifest = load_json(artifact_dir / "bpe_manifest.json")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else default_checkpoint(artifact_dir, args.run_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    context_length = args.context_length or int(config["context_length"])
    batch_size = args.batch_size or int(checkpoint.get("training_args", {}).get("batch_size", 16))

    tokenizer_path = Path(checkpoint.get("tokenizer_path", artifact_path(artifact_dir, manifest["vocab_path"])))
    if not tokenizer_path.exists():
        tokenizer_path = artifact_path(artifact_dir, manifest["vocab_path"])
    tokenizer = load_tokenizer(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_metrics = []
    requested_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    for split in requested_splits:
        ids = get_split_ids(artifact_dir, manifest, tokenizer, split)
        split_metrics.append(
            evaluate_split(
                model,
                ids,
                split,
                context_length,
                batch_size,
                device,
                args.eval_batches,
            )
        )

    io_ids = get_split_ids(artifact_dir, manifest, tokenizer, requested_splits[0])
    io_report = inspect_io(model, io_ids, tokenizer, context_length, batch_size, device)

    prompts = args.prompt or ["이 영화", "정말", "배우의 연기가"]
    generations = generate_samples(
        model,
        tokenizer,
        prompts,
        context_length,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.run_name is not None:
        out_dir = artifact_dir / "runs" / args.run_name / "test"
    else:
        out_dir = checkpoint_path.parent / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
        "device": str(device),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_step": int(checkpoint.get("global_step", 0)),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "config": config,
        "split_metrics": split_metrics,
        "io": io_report,
        "generations": generations,
    }
    save_json(out_dir / "test_report.json", report)
    write_csv(out_dir / "split_metrics.csv", split_metrics)
    write_generation_csv(out_dir / "generations.csv", generations)
    write_report(out_dir / "test_report.md", report)

    print(f"checkpoint={checkpoint_path}", flush=True)
    for row in split_metrics:
        print(f"{row['split']}_loss={row['loss']:.4f}", flush=True)
    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
