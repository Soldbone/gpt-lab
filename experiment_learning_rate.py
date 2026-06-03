# -*- coding: utf-8 -*-
"""Run learning-rate comparison experiments for mini GPT pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from analyze_learning_rate_results import format_learning_rate, write_analysis_outputs
from experiment_batch_size import (
    choose_sample_sentences,
    current_memory,
    evaluate_epoch,
    log_line,
    make_loaders,
    make_model_config,
    metric_value,
    parse_int_list,
    parse_optional_int,
    prompt_and_target,
    score_text,
    simple_generation_comment,
    write_csv,
)
from generate_learning_rate_report_summary import write_report
from pretrain import DEFAULT_HYPERPARAMS, generate_sample_text, pick_device, read_text
from src.bpe import BPETokenizer
from src.model import GPTModel
from src.reproducibility import DEFAULT_SEED, set_global_seed
from src.train import calc_loss_batch, load_checkpoint, save_checkpoint
from visualize_learning_rate_results import write_visualizations


ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_PATH = ROOT / "checkpoints_dropout01_vocab3000" / "bpe_vocab_3000.json"


def parse_float_list(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return items


def make_run_root(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_dir / f"learning_rate_experiment_{timestamp}"
    counter = 1
    while root.exists():
        root = base_dir / f"learning_rate_experiment_{timestamp}_{counter}"
        counter += 1
    root.mkdir(parents=True)
    for child in ("checkpoints", "graphs", "tables", "metrics", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def write_metrics(root: Path, learning_rate: float, rows: list[dict]) -> None:
    metrics_dir = root / "metrics"
    lr_label = format_learning_rate(learning_rate)
    write_csv(metrics_dir / f"lr_{lr_label}_metrics.csv", rows)
    (metrics_dir / f"lr_{lr_label}_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip()


def tokenizer_blame_authors(path: Path) -> list[str]:
    relative_path = _relative(path)
    try:
        result = subprocess.run(
            ["git", "blame", "--line-porcelain", "--", relative_path],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return []
    authors = sorted({
        line.removeprefix("author ").strip()
        for line in result.stdout.splitlines()
        if line.startswith("author ")
    })
    return authors


def load_existing_tokenizer(args: argparse.Namespace, run_root: Path) -> tuple[BPETokenizer, Path, Path, str]:
    tokenizer_path = args.tokenizer_path.resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.load(tokenizer_path)
    actual_vocab_size = len(tokenizer.id_to_token)
    if actual_vocab_size != args.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size mismatch: expected {args.vocab_size}, got {actual_vocab_size} from {tokenizer_path}"
        )

    run_vocab_path = run_root / "checkpoints" / f"bpe_vocab_{args.vocab_size}.json"
    run_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_path, run_vocab_path)
    return tokenizer, tokenizer_path, run_vocab_path, "loaded-existing"


def data_manifest(path: Path, text: str, token_ids: list[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "used_chars": len(text),
        "used_text_sha256": text_sha256(text),
        "tokens": len(token_ids),
    }
    if path.exists():
        payload.update({
            "source_file_bytes": path.stat().st_size,
            "source_file_sha256": file_sha256(path),
        })
    return payload


def environment_manifest(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
    }
    if torch.cuda.is_available():
        payload["cuda_version"] = torch.version.cuda
        payload["cuda_device_count"] = torch.cuda.device_count()
        payload["cuda_device_name"] = torch.cuda.get_device_name(device if device.type == "cuda" else 0)
    return payload


def save_run_config(
    args: argparse.Namespace,
    run_root: Path,
    tokenizer_path: Path,
    run_tokenizer_path: Path,
    tokenizer_mode: str,
    device: torch.device,
    train_text: str,
    val_text: str,
    test_text: str,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
) -> None:
    payload = {
        "experiment": "learning_rate",
        "command": {
            "argv": sys.argv,
            "python_executable": sys.executable,
        },
        "git": {
            "branch": git_output("branch", "--show-current"),
            "commit": git_output("rev-parse", "HEAD"),
            "status_short": git_output("status", "--short"),
        },
        "environment": environment_manifest(device),
        "seed": args.seed,
        "learning_rates": args.learning_rates,
        "epochs": args.epochs,
        "corpus_size": args.corpus_size,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "n_heads": args.n_heads,
        "drop_rate": args.drop_rate,
        "weight_decay": args.weight_decay,
        "stride": args.stride,
        "eval_batches": args.eval_batches,
        "accuracy_eval_batches": args.accuracy_eval_batches,
        "top_n_values": args.top_n_values,
        "tokenizer": {
            "path": str(tokenizer_path),
            "run_copy_path": str(run_tokenizer_path),
            "mode": tokenizer_mode,
            "sha256": file_sha256(tokenizer_path),
            "blame_authors": tokenizer_blame_authors(tokenizer_path),
        },
        "device": str(device),
        "data": {
            "train": data_manifest(args.train_path, train_text, train_ids),
            "val": data_manifest(args.val_path, val_text, val_ids),
            "test": data_manifest(args.test_path, test_text, test_ids),
        },
    }
    (run_root / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_one_learning_rate(
    learning_rate: float,
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    device: torch.device,
    run_root: Path,
) -> list[dict]:
    lr_label = format_learning_rate(learning_rate)
    log_path = run_root / "logs" / f"lr_{lr_label}.log"
    log_line(log_path, f"Starting learning_rate={lr_label} with seed={args.seed}")
    set_global_seed(args.seed)

    loaders = make_loaders(train_ids, val_ids, test_ids, args.batch_size, args)
    config = make_model_config(tokenizer, args)
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    checkpoint_dir = run_root / "checkpoints" / f"lr_{lr_label}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    metrics: list[dict] = []
    best_val_loss = float("inf")
    global_step = 0
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        for input_batch, target_batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

        epoch_time = time.perf_counter() - epoch_start
        eval_row = evaluate_epoch(
            model=model,
            loaders=loaders,
            device=device,
            loss_batches=args.eval_batches,
            accuracy_batches=args.accuracy_eval_batches,
            top_n_values=args.top_n_values,
        )
        total_time = time.perf_counter() - total_start
        allocated, reserved, memory_status = current_memory(device)
        row = {
            "learning_rate": learning_rate,
            "learning_rate_label": lr_label,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "epoch": epoch,
            "global_step": global_step,
            **{key: metric_value(value) for key, value in eval_row.items()},
            "epoch_time": epoch_time,
            "total_training_time": total_time,
            "average_epoch_time": total_time / epoch,
            "max_gpu_memory_allocated": allocated,
            "max_gpu_memory_reserved": reserved,
            "memory_status": memory_status,
        }
        metrics.append(row)
        write_metrics(run_root, learning_rate, metrics)

        val_loss = float(eval_row["val_loss"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, global_step, str(checkpoint_dir / "best.pt"))
            log_line(log_path, f"Saved best checkpoint at epoch={epoch}, val_loss={val_loss:.6f}")

        log_line(
            log_path,
            (
                f"Epoch {epoch:03d}: train_loss={eval_row['train_loss']:.4f}, "
                f"val_loss={eval_row['val_loss']:.4f}, test_loss={eval_row['test_loss']:.4f}, "
                f"epoch_time={epoch_time:.2f}s"
            ),
        )

    save_checkpoint(model, optimizer, args.epochs, global_step, str(checkpoint_dir / "final.pt"))
    log_line(log_path, f"Saved final checkpoint: {checkpoint_dir / 'final.pt'}")
    return metrics


def generate_samples(
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    samples: list[str],
    device: torch.device,
    run_root: Path,
) -> list[dict]:
    rows: list[dict] = []
    config = make_model_config(tokenizer, args)
    for sentence in samples:
        prompt, target = prompt_and_target(sentence)
        for learning_rate in args.learning_rates:
            lr_label = format_learning_rate(learning_rate)
            model = GPTModel(config).to(device)
            load_checkpoint(model, None, str(run_root / "checkpoints" / f"lr_{lr_label}" / "best.pt"), device)
            output = generate_sample_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                context_length=args.context_length,
                max_new_tokens=args.report_sample_tokens,
            )
            loss_score = score_text(model, tokenizer, target, device, args.context_length)
            rows.append({
                "input_sentence": prompt,
                "learning_rate": learning_rate,
                "learning_rate_label": lr_label,
                "batch_size": args.batch_size,
                "generated_output": output,
                "target_sentence": target,
                "expected_output": target,
                "loss": loss_score,
                "score": loss_score,
                "simple_evaluation": simple_generation_comment(output),
            })
    return rows


def parse_args() -> argparse.Namespace:
    defaults = DEFAULT_HYPERPARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rates", type=parse_float_list, default=[1e-4, 3e-4, 5e-4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--corpus-size", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-path", type=Path, default=defaults["train_path"])
    parser.add_argument("--val-path", type=Path, default=defaults["val_path"])
    parser.add_argument("--test-path", type=Path, default=defaults["test_path"])
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-batches", type=parse_optional_int, default=None)
    parser.add_argument("--accuracy-eval-batches", type=parse_optional_int, default=None)
    parser.add_argument("--top-n", type=parse_int_list, default=[1, 3, 5])
    parser.add_argument("--start-context", type=str, default="영화")
    parser.add_argument("--report-sample-tokens", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.top_n_values = args.top_n
    if args.smoke:
        args.learning_rates = [1e-4, 3e-4]
        args.epochs = 1
        args.corpus_size = min(args.corpus_size, 3_000)
        args.context_length = min(args.context_length, 32)
        args.emb_dim = min(args.emb_dim, 32)
        args.n_layers = 1
        args.batch_size = min(args.batch_size, 2)
        args.eval_batches = 1
        args.accuracy_eval_batches = 1
        args.report_sample_tokens = min(args.report_sample_tokens, 8)
    return args


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    run_root = make_run_root(args.output_dir)
    device = pick_device(args.device)

    train_text = read_text(args.train_path, args.corpus_size)
    val_text = read_text(args.val_path, None)
    test_text = read_text(args.test_path, None) if args.test_path.exists() else val_text
    tokenizer, tokenizer_path, run_tokenizer_path, tokenizer_mode = load_existing_tokenizer(args, run_root)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    test_ids = tokenizer.encode(test_text)
    if min(len(train_ids), len(val_ids), len(test_ids)) <= args.context_length:
        raise ValueError("Not enough tokens for the chosen context length.")

    save_run_config(
        args=args,
        run_root=run_root,
        tokenizer_path=tokenizer_path,
        run_tokenizer_path=run_tokenizer_path,
        tokenizer_mode=tokenizer_mode,
        device=device,
        train_text=train_text,
        val_text=val_text,
        test_text=test_text,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    print(f"Run root: {run_root}")
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_path} ({tokenizer_mode})")

    for learning_rate in args.learning_rates:
        train_one_learning_rate(
            learning_rate=learning_rate,
            args=args,
            tokenizer=tokenizer,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            device=device,
            run_root=run_root,
        )

    sample_rows = generate_samples(
        args=args,
        tokenizer=tokenizer,
        samples=choose_sample_sentences(test_text),
        device=device,
        run_root=run_root,
    )
    write_analysis_outputs(run_root, sample_rows=sample_rows)
    write_visualizations(run_root)
    write_report(run_root)
    print(f"Completed learning rate experiment: {run_root}")


if __name__ == "__main__":
    main()
