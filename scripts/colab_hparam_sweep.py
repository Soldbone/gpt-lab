# -*- coding: utf-8 -*-
"""Run Colab-friendly NSMC pretraining experiments.

Examples:
    python scripts/colab_hparam_sweep.py --preset pilot
    python scripts/colab_hparam_sweep.py --preset final
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import download_data  # noqa: E402
from bpe import BPETokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
import train  # noqa: E402


@dataclass(frozen=True)
class Experiment:
    name: str
    lr: float
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    batch_size: int
    num_epochs: int
    eval_freq: int
    eval_iter: int
    weight_decay: float = 0.1
    qkv_bias: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_text(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    return text[:limit]


def get_experiments(preset: str) -> list[Experiment]:
    if preset == "pilot":
        return [
            Experiment("lr_1e-4", 1e-4, 64, 128, 4, 2, 0.1, 16, 2, 100, 20),
            Experiment("lr_3e-4", 3e-4, 64, 128, 4, 2, 0.1, 16, 2, 100, 20),
            Experiment("lr_5e-4", 5e-4, 64, 128, 4, 2, 0.1, 16, 2, 100, 20),
            Experiment("ctx_128", 3e-4, 128, 128, 4, 2, 0.1, 12, 2, 100, 20),
            Experiment("layers_4", 3e-4, 64, 128, 4, 4, 0.1, 12, 2, 100, 20),
            Experiment("dropout_0", 3e-4, 64, 128, 4, 2, 0.0, 16, 2, 100, 20),
        ]
    if preset == "final":
        return [
            Experiment("final_ctx128_layers4_lr3e-4", 3e-4, 128, 192, 4, 4, 0.1, 16, 5, 100, 20),
        ]
    raise ValueError(f"unknown preset: {preset}")


def make_tokenizer(train_text: str, vocab_size: int, out_dir: Path) -> BPETokenizer:
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(train_text)
    tokenizer.save(out_dir / f"bpe_vocab_{vocab_size}.json")
    return tokenizer


def evaluate_for_run(model, train_loader, val_loader, device, eval_iter: int) -> tuple[float, float]:
    return train.evaluate_model(model, train_loader, val_loader, device, eval_iter)


def run_one(
    exp: Experiment,
    train_ids: list[int],
    val_ids: list[int],
    vocab_size: int,
    device: torch.device,
    out_dir: Path,
    seed: int,
    grad_clip: float,
    save_checkpoint: bool,
) -> tuple[list[dict], dict]:
    set_seed(seed)

    run_dir = out_dir / exp.name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "vocab_size": vocab_size,
        "context_length": exp.context_length,
        "emb_dim": exp.emb_dim,
        "n_heads": exp.n_heads,
        "n_layers": exp.n_layers,
        "drop_rate": exp.drop_rate,
        "qkv_bias": exp.qkv_bias,
    }

    train_loader = create_dataloader(
        train_ids,
        context_length=exp.context_length,
        batch_size=exp.batch_size,
        stride=exp.context_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=exp.context_length,
        batch_size=exp.batch_size,
        stride=exp.context_length,
        shuffle=False,
        drop_last=False,
    )

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    param_count = sum(p.numel() for p in model.parameters())

    metrics: list[dict] = []
    global_step = 0
    best_val = float("inf")
    best_step = 0
    start_time = time.time()
    steps_per_epoch = len(train_loader)

    for epoch_idx in range(exp.num_epochs):
        model.train()
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = train.calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1

            is_eval_step = global_step % exp.eval_freq == 0
            is_epoch_end = batch_idx == steps_per_epoch
            if is_eval_step or is_epoch_end:
                train_loss, val_loss = evaluate_for_run(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    exp.eval_iter,
                )
                epoch_float = epoch_idx + (batch_idx / steps_per_epoch)
                metrics.append(
                    {
                        "run": exp.name,
                        "epoch": epoch_float,
                        "global_step": global_step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                print(
                    f"{exp.name} epoch={epoch_float:.3f} "
                    f"step={global_step:06d} train={train_loss:.4f} val={val_loss:.4f}",
                    flush=True,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_step = global_step
                    if save_checkpoint:
                        train.save_checkpoint(
                            model,
                            optimizer,
                            epoch_idx + 1,
                            global_step,
                            str(run_dir / "best_checkpoint.pt"),
                        )
                if is_epoch_end:
                    epoch_summary = build_epoch_summary(metrics)
                    write_csv(run_dir / "metrics.csv", metrics)
                    write_csv(run_dir / "epoch_summary.csv", epoch_summary)
                    write_epoch_markdown(run_dir / "epoch_summary.md", epoch_summary)

    elapsed_sec = time.time() - start_time
    summary = {
        **asdict(exp),
        "vocab_size": vocab_size,
        "param_count": param_count,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": global_step,
        "best_val_loss": best_val,
        "best_step": best_step,
        "final_train_loss": metrics[-1]["train_loss"] if metrics else None,
        "final_val_loss": metrics[-1]["val_loss"] if metrics else None,
        "elapsed_sec": elapsed_sec,
    }

    epoch_summary = build_epoch_summary(metrics)
    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "epoch_summary.csv", epoch_summary)
    write_csv(run_dir / "summary.csv", [summary])
    write_epoch_markdown(run_dir / "epoch_summary.md", epoch_summary)
    return metrics, summary


def build_epoch_summary(metrics: list[dict]) -> list[dict]:
    epoch_rows = []
    best_val = float("inf")
    best_epoch = 0
    prev_val = None

    for row in metrics:
        epoch_value = float(row["epoch"])
        rounded_epoch = round(epoch_value)
        if rounded_epoch < 1 or abs(epoch_value - rounded_epoch) > 1e-6:
            continue

        train_loss = float(row["train_loss"])
        val_loss = float(row["val_loss"])
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = rounded_epoch

        epoch_rows.append(
            {
                "epoch": rounded_epoch,
                "global_step": int(row["global_step"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "generalization_gap": val_loss - train_loss,
                "val_loss_delta": "" if prev_val is None else val_loss - prev_val,
                "best_val_loss_so_far": best_val,
                "best_epoch_so_far": best_epoch,
            }
        )
        prev_val = val_loss

    return epoch_rows


def write_epoch_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    lines = [
        "| epoch | step | train loss | val loss | gap | val delta | best val so far | best epoch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = row["val_loss_delta"]
        delta_text = "" if delta == "" else f"{delta:.4f}"
        lines.append(
            "| {epoch} | {global_step} | {train_loss:.4f} | {val_loss:.4f} | "
            "{generalization_gap:.4f} | {delta_text} | {best_val_loss_so_far:.4f} | "
            "{best_epoch_so_far} |".format(delta_text=delta_text, **row)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_all_metrics(out_dir: Path, all_metrics: list[dict]) -> None:
    if not all_metrics:
        return

    runs = sorted({row["run"] for row in all_metrics})

    plt.figure(figsize=(11, 6))
    for run in runs:
        rows = [row for row in all_metrics if row["run"] == run]
        plt.plot(
            [row["epoch"] for row in rows],
            [row["val_loss"] for row in rows],
            label=run,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss by Hyperparameter Setting")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "val_loss_by_epoch.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 6))
    for run in runs:
        rows = [row for row in all_metrics if row["run"] == run]
        plt.plot(
            [row["epoch"] for row in rows],
            [row["train_loss"] for row in rows],
            label=f"{run} train",
            alpha=0.8,
        )
        plt.plot(
            [row["epoch"] for row in rows],
            [row["val_loss"] for row in rows],
            linestyle="--",
            label=f"{run} val",
            alpha=0.8,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss Curves")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "train_val_loss_by_epoch.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_all_epoch_summary(out_dir: Path, all_metrics: list[dict]) -> None:
    rows = []
    for run in sorted({row["run"] for row in all_metrics}):
        for epoch_row in build_epoch_summary([row for row in all_metrics if row["run"] == run]):
            rows.append({"run": run, **epoch_row})
    write_csv(out_dir / "epoch_summary.csv", rows)


def write_report_snippet(out_dir: Path, summaries: list[dict]) -> None:
    if not summaries:
        return
    best = min(summaries, key=lambda row: row["best_val_loss"])
    lines = [
        "# Colab Sweep Summary",
        "",
        f"- Best run: `{best['name']}`",
        f"- Best validation loss: `{best['best_val_loss']:.4f}` at step `{best['best_step']}`",
        "",
        "| run | lr | context | layers | emb | dropout | best val | final val | params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: item["best_val_loss"]):
        lines.append(
            "| {name} | {lr:.0e} | {context_length} | {n_layers} | {emb_dim} | "
            "{drop_rate} | {best_val_loss:.4f} | {final_val_loss:.4f} | {param_count} |".format(**row)
        )
    lines.append("")
    lines.append("Generated plots:")
    lines.append("- `val_loss_by_epoch.png`")
    lines.append("- `train_val_loss_by_epoch.png`")
    (out_dir / "report_snippet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["pilot", "final"], default="pilot")
    parser.add_argument("--out-dir", default="outputs/colab_sweep")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--train-chars", type=int, default=None)
    parser.add_argument("--val-chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = ROOT / args.out_dir / args.preset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_chars = args.train_chars
    val_chars = args.val_chars
    if train_chars is None:
        train_chars = 300_000 if args.preset == "pilot" else 0
    if val_chars is None:
        val_chars = 60_000 if args.preset == "pilot" else 0

    paths = download_data.main()
    train_text = limit_text(Path(paths["lm_train"]).read_text(encoding="utf-8"), train_chars)
    val_text = limit_text(Path(paths["lm_val"]).read_text(encoding="utf-8"), val_chars)

    print(f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
    print(f"preset={args.preset}", flush=True)
    print(f"train_chars={len(train_text)} val_chars={len(val_text)}", flush=True)

    tokenizer = make_tokenizer(train_text, args.vocab_size, out_dir)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    actual_vocab_size = len(tokenizer.token_to_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = get_experiments(args.preset)
    if args.num_epochs is not None:
        experiments = [replace(exp, num_epochs=args.num_epochs) for exp in experiments]
    all_metrics: list[dict] = []
    summaries: list[dict] = []

    for exp in experiments:
        metrics, summary = run_one(
            exp=exp,
            train_ids=train_ids,
            val_ids=val_ids,
            vocab_size=actual_vocab_size,
            device=device,
            out_dir=out_dir,
            seed=args.seed,
            grad_clip=args.grad_clip,
            save_checkpoint=args.save_checkpoint or args.preset == "final",
        )
        all_metrics.extend(metrics)
        summaries.append(summary)
        write_csv(out_dir / "all_metrics.csv", all_metrics)
        write_csv(out_dir / "summary.csv", summaries)
        write_all_epoch_summary(out_dir, all_metrics)
        plot_all_metrics(out_dir, all_metrics)
        write_report_snippet(out_dir, summaries)

    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
