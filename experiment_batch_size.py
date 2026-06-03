# -*- coding: utf-8 -*-
"""Run batch-size comparison experiments for mini GPT pretraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from analyze_results import write_analysis_outputs
from generate_report_summary import write_report
from pretrain import DEFAULT_HYPERPARAMS, compute_topk_accuracy, generate_sample_text, pick_device, read_text, safe_ppl
from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.reproducibility import DEFAULT_SEED, make_torch_generator, seed_worker, set_global_seed
from src.train import calc_loss_batch, calc_loss_loader, load_checkpoint, save_checkpoint
from visualize_results import write_visualizations


ROOT = Path(__file__).resolve().parent


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return items


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "all", ""}:
        return None
    return int(value)


def make_run_root(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_dir / f"batch_size_experiment_{timestamp}"
    counter = 1
    while root.exists():
        root = base_dir / f"batch_size_experiment_{timestamp}_{counter}"
        counter += 1
    root.mkdir(parents=True)
    for child in ("checkpoints", "graphs", "tables", "metrics", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(root: Path, batch_size: int, rows: list[dict]) -> None:
    metrics_dir = root / "metrics"
    write_csv(metrics_dir / f"batch_{batch_size}_metrics.csv", rows)
    (metrics_dir / f"batch_{batch_size}_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def log_line(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def default_tokenizer_candidates(vocab_size: int) -> list[Path]:
    return [
        ROOT / "checkpoints" / f"bpe_vocab_{vocab_size}.json",
        ROOT / "checkpoints_dropout01_vocab3000" / f"bpe_vocab_{vocab_size}.json",
        DEFAULT_HYPERPARAMS["output_dir"] / f"bpe_vocab_{vocab_size}.json",
    ]


def build_or_load_tokenizer(args: argparse.Namespace, train_text: str, run_root: Path) -> tuple[BPETokenizer, Path, str]:
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    selected_path: Path | None = None
    mode = "loaded-existing"

    if args.tokenizer_path is not None:
        selected_path = args.tokenizer_path
    else:
        for candidate in default_tokenizer_candidates(args.vocab_size):
            if candidate.exists():
                selected_path = candidate
                break

    if selected_path is not None and selected_path.exists() and not args.retrain_tokenizer:
        tokenizer.load(selected_path)
    else:
        mode = "trained-new"
        selected_path = selected_path or (run_root / "checkpoints" / f"bpe_vocab_{args.vocab_size}.json")
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.train(train_text)
        tokenizer.save(selected_path)

    run_vocab_path = run_root / "checkpoints" / f"bpe_vocab_{args.vocab_size}.json"
    if selected_path.resolve() != run_vocab_path.resolve():
        run_vocab_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_path, run_vocab_path)
    return tokenizer, selected_path, mode


def make_model_config(tokenizer: BPETokenizer, args: argparse.Namespace) -> dict:
    return {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
    }


def metric_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return value


def current_memory(device: torch.device) -> tuple[Any, Any, str]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "N/A", "N/A", "CUDA not available"
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
        "available",
    )


def evaluate_epoch(
    model: GPTModel,
    loaders: dict[str, Any],
    device: torch.device,
    loss_batches: int | None,
    accuracy_batches: int | None,
    top_n_values: list[int],
) -> dict:
    model.eval()
    losses = {
        split: calc_loss_loader(loader, model, device, num_batches=loss_batches)
        for split, loader in loaders.items()
    }
    accuracies: dict[str, dict[int, float]] = {}
    for split, loader in loaders.items():
        accuracies[split] = compute_topk_accuracy(
            model=model,
            data_loader=loader,
            device=device,
            top_k_values=top_n_values,
            num_batches=accuracy_batches,
        )
    model.train()

    row = {
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "test_loss": losses["test"],
        "train_ppl": safe_ppl(losses["train"]),
        "val_ppl": safe_ppl(losses["val"]),
        "test_ppl": safe_ppl(losses["test"]),
    }
    for split in ("train", "val", "test"):
        for top_n in top_n_values:
            row[f"{split}_top_{top_n}_accuracy"] = accuracies[split][top_n]
    for top_n in top_n_values:
        row[f"top_{top_n}_accuracy"] = accuracies["test"][top_n]
    return row


def make_loaders(
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_generator = make_torch_generator(args.seed)
    return {
        "train": create_dataloader(
            train_ids,
            context_length=args.context_length,
            batch_size=batch_size,
            stride=args.stride,
            drop_last=True,
            shuffle=True,
            num_workers=args.num_workers,
            generator=train_generator,
            worker_init_fn=seed_worker,
        ),
        "val": create_dataloader(
            val_ids,
            context_length=args.context_length,
            batch_size=batch_size,
            stride=args.stride,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
        ),
        "test": create_dataloader(
            test_ids,
            context_length=args.context_length,
            batch_size=batch_size,
            stride=args.stride,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
        ),
    }


def train_one_batch_size(
    batch_size: int,
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    device: torch.device,
    run_root: Path,
) -> list[dict]:
    log_path = run_root / "logs" / f"batch_{batch_size}.log"
    log_line(log_path, f"Starting batch_size={batch_size} with seed={args.seed}")
    set_global_seed(args.seed)

    loaders = make_loaders(train_ids, val_ids, test_ids, batch_size, args)
    config = make_model_config(tokenizer, args)
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = run_root / "checkpoints" / f"batch_{batch_size}"
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
            "batch_size": batch_size,
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
        write_metrics(run_root, batch_size, metrics)

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


def choose_sample_sentences(test_text: str, count: int = 3) -> list[str]:
    candidates = [line.strip() for line in test_text.splitlines() if line.strip()]
    if not candidates:
        return []
    ordered = sorted(candidates, key=len)
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    unique: list[str] = []
    for item in picks:
        if item not in unique:
            unique.append(item)
    return unique[:count]


def prompt_and_target(sentence: str) -> tuple[str, str]:
    if len(sentence) <= 12:
        return sentence, sentence
    cut = max(6, min(len(sentence) - 1, len(sentence) // 2))
    return sentence[:cut], sentence


def score_text(model: GPTModel, tokenizer: BPETokenizer, text: str, device: torch.device, context_length: int) -> float | str:
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        return "N/A"
    token_ids = token_ids[: context_length + 1]
    input_ids = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    target_ids = torch.tensor(token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        loss = calc_loss_batch(input_ids, target_ids, model, device)
    model.train()
    return float(loss.item())


def simple_generation_comment(output: str) -> str:
    words = output.split()
    if not output.strip():
        return "No generation was produced."
    if len(words) >= 8:
        repeats = sum(1 for left, right in zip(words, words[1:]) if left == right)
        if repeats / max(len(words) - 1, 1) > 0.25:
            return "Immediate repetition is visible."
    return "Automatic check found no strong immediate repetition."


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
        for batch_size in args.batch_sizes:
            model = GPTModel(config).to(device)
            load_checkpoint(model, None, str(run_root / "checkpoints" / f"batch_{batch_size}" / "best.pt"), device)
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
                "batch_size": batch_size,
                "generated_output": output,
                "target_sentence": target,
                "expected_output": target,
                "loss": loss_score,
                "score": loss_score,
                "simple_evaluation": simple_generation_comment(output),
            })
    return rows


def save_run_config(
    args: argparse.Namespace,
    run_root: Path,
    tokenizer_path: Path,
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
        "seed": args.seed,
        "batch_sizes": args.batch_sizes,
        "epochs": args.epochs,
        "corpus_size": args.corpus_size,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "drop_rate": args.drop_rate,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stride": args.stride,
        "eval_batches": args.eval_batches,
        "accuracy_eval_batches": args.accuracy_eval_batches,
        "top_n_values": args.top_n_values,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_mode": tokenizer_mode,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "test_chars": len(test_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "test_tokens": len(test_ids),
    }
    (run_root / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    defaults = DEFAULT_HYPERPARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[4, 8, 16])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--corpus-size", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-path", type=Path, default=defaults["train_path"])
    parser.add_argument("--val-path", type=Path, default=defaults["val_path"])
    parser.add_argument("--test-path", type=Path, default=defaults["test_path"])
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        args.batch_sizes = [2, 4]
        args.epochs = 1
        args.corpus_size = min(args.corpus_size, 3_000)
        args.vocab_size = min(args.vocab_size, 300)
        args.context_length = min(args.context_length, 32)
        args.emb_dim = min(args.emb_dim, 32)
        args.n_layers = 1
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
    tokenizer, tokenizer_path, tokenizer_mode = build_or_load_tokenizer(args, train_text, run_root)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    test_ids = tokenizer.encode(test_text)
    if min(len(train_ids), len(val_ids), len(test_ids)) <= args.context_length:
        raise ValueError("Not enough tokens for the chosen context length.")

    save_run_config(
        args=args,
        run_root=run_root,
        tokenizer_path=tokenizer_path,
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

    for batch_size in args.batch_sizes:
        train_one_batch_size(
            batch_size=batch_size,
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
    print(f"Completed batch size experiment: {run_root}")


if __name__ == "__main__":
    main()
