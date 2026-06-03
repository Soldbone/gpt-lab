# -*- coding: utf-8 -*-
"""Create report-ready figures for the batch-size experiment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from analyze_results import best_row, load_metrics
from pretrain import get_pyplot


COLORS = {
    4: "#2563eb",
    8: "#dc2626",
    16: "#059669",
}


def _color(batch_size: int) -> str:
    return COLORS.get(batch_size, "#7c3aed")


def _as_float(value: Any) -> float:
    if value in {None, "", "N/A"}:
        return float("nan")
    return float(value)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def _best_epoch(metrics: list[dict]) -> int:
    return int(best_row(metrics)["epoch"])


def _plot_batch_metric(metrics: list[dict], batch_size: int, metric: str, ylabel: str, path: Path) -> None:
    plt = get_pyplot(path)
    epochs = [int(row["epoch"]) for row in metrics]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for split, color in [("train", "#2563eb"), ("val", "#dc2626"), ("test", "#059669")]:
        key = f"{split}_{metric}"
        values = [_as_float(row[key]) for row in metrics]
        ax.plot(epochs, values, marker="o", linewidth=2.4, color=color, label=key)

    best = best_row(metrics)
    best_epoch = int(best["epoch"])
    best_value = _as_float(best[f"val_{metric}"])
    ax.scatter([best_epoch], [best_value], s=140, color="#f59e0b", zorder=5, label="best val epoch")
    ax.annotate(
        f"best epoch {best_epoch}",
        xy=(best_epoch, best_value),
        xytext=(12, 16),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#92400e"},
        fontsize=10,
        color="#92400e",
    )
    ax.set_title(f"Batch Size {batch_size} {ylabel}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.55)
    if metric == "ppl":
        finite = [value for row in metrics for split in ("train", "val", "test") for value in [_as_float(row[f"{split}_{metric}"])] if not math.isnan(value)]
        if finite and max(finite) / max(min(finite), 1e-9) > 20:
            ax.set_yscale("log")
            ax.set_ylabel(f"{ylabel} (log scale)")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_comparison(metrics_by_batch: dict[int, list[dict]], metric: str, split: str, ylabel: str, path: Path) -> None:
    plt = get_pyplot(path)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        epochs = [int(row["epoch"]) for row in metrics]
        values = [_as_float(row[f"{split}_{metric}"]) for row in metrics]
        ax.plot(
            epochs,
            values,
            marker="o",
            linewidth=2.5,
            color=_color(batch_size),
            label=f"batch {batch_size}",
        )
        best = best_row(metrics)
        if split == "val":
            ax.scatter([int(best["epoch"])], [_as_float(best[f"{split}_{metric}"])], s=120, color="#f59e0b", zorder=5)
    ax.set_title(f"{split.title()} {ylabel} Comparison")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_topn(metrics_by_batch: dict[int, list[dict]], mode: str, path: Path) -> None:
    plt = get_pyplot(path)
    top_values = [1, 3, 5]
    batch_sizes = sorted(metrics_by_batch)
    width = 0.22
    x_positions = list(range(len(top_values)))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for batch_idx, batch_size in enumerate(batch_sizes):
        source = best_row(metrics_by_batch[batch_size]) if mode == "best" else metrics_by_batch[batch_size][-1]
        offsets = [x + (batch_idx - (len(batch_sizes) - 1) / 2) * width for x in x_positions]
        scores = [_as_float(source.get(f"test_top_{top_n}_accuracy")) * 100 for top_n in top_values]
        bars = ax.bar(offsets, scores, width=width, color=_color(batch_size), label=f"batch {batch_size}")
        for bar, score in zip(bars, scores):
            label = "N/A" if math.isnan(score) else f"{score:.2f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, 0 if math.isnan(score) else bar.get_height(), label, ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Top-N Accuracy Comparison ({mode} epoch)")
    ax.set_xlabel("top-N")
    ax.set_ylabel("accuracy (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"top-{top_n}" for top_n in top_values])
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_best_epoch(metrics_by_batch: dict[int, list[dict]], path: Path) -> None:
    plt = get_pyplot(path)
    batch_sizes = sorted(metrics_by_batch)
    best_epochs = [_best_epoch(metrics_by_batch[batch_size]) for batch_size in batch_sizes]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.bar([str(batch_size) for batch_size in batch_sizes], best_epochs, color=[_color(batch_size) for batch_size in batch_sizes])
    for bar, epoch in zip(bars, best_epochs):
        ax.text(bar.get_x() + bar.get_width() / 2, epoch, str(epoch), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Best Epoch by Batch Size")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("best epoch")
    ax.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_gap_by_epoch(metrics_by_batch: dict[int, list[dict]], metric: str, path: Path) -> None:
    plt = get_pyplot(path)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        epochs = [int(row["epoch"]) for row in metrics]
        values = [_as_float(row[f"val_{metric}"]) - _as_float(row[f"train_{metric}"]) for row in metrics]
        ax.plot(epochs, values, marker="o", linewidth=2.4, color=_color(batch_size), label=f"batch {batch_size}")
    ax.axhline(0, color="#64748b", linewidth=1.2, linestyle="--")
    ax.set_title(f"Validation - Train {metric.upper()} Gap by Epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{metric} gap")
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_final_loss_gap(metrics_by_batch: dict[int, list[dict]], path: Path) -> None:
    plt = get_pyplot(path)
    batch_sizes = sorted(metrics_by_batch)
    gaps = [
        _as_float(metrics_by_batch[batch_size][-1]["val_loss"]) - _as_float(metrics_by_batch[batch_size][-1]["train_loss"])
        for batch_size in batch_sizes
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.bar([str(batch_size) for batch_size in batch_sizes], gaps, color=[_color(batch_size) for batch_size in batch_sizes])
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, gap, f"{gap:.4f}", ha="center", va="bottom" if gap >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#64748b", linewidth=1.2, linestyle="--")
    ax.set_title("Final Loss Gap Comparison")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("val_loss - train_loss")
    ax.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_time(metrics_by_batch: dict[int, list[dict]], key: str, title: str, ylabel: str, path: Path) -> None:
    plt = get_pyplot(path)
    batch_sizes = sorted(metrics_by_batch)
    values = [_as_float(metrics_by_batch[batch_size][-1].get(key)) for batch_size in batch_sizes]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.bar([str(batch_size) for batch_size in batch_sizes], values, color=[_color(batch_size) for batch_size in batch_sizes])
    for bar, value in zip(bars, values):
        label = "N/A" if math.isnan(value) else f"{value:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, 0 if math.isnan(value) else value, label, ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("batch_size")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_memory(metrics_by_batch: dict[int, list[dict]], path: Path) -> None:
    plt = get_pyplot(path)
    batch_sizes = sorted(metrics_by_batch)
    allocated = [_as_float(metrics_by_batch[batch_size][-1].get("max_gpu_memory_allocated")) for batch_size in batch_sizes]
    reserved = [_as_float(metrics_by_batch[batch_size][-1].get("max_gpu_memory_reserved")) for batch_size in batch_sizes]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    if all(math.isnan(value) for value in allocated + reserved):
        ax.axis("off")
        ax.text(0.5, 0.55, "GPU memory metrics: N/A", ha="center", va="center", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.42, "CUDA was not available during this run.", ha="center", va="center", fontsize=12)
    else:
        x_positions = list(range(len(batch_sizes)))
        width = 0.35
        ax.bar([x - width / 2 for x in x_positions], allocated, width=width, label="allocated", color="#2563eb")
        ax.bar([x + width / 2 for x in x_positions], reserved, width=width, label="reserved", color="#dc2626")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
        ax.set_xlabel("batch_size")
        ax.set_ylabel("bytes")
        ax.grid(True, axis="y", alpha=0.5)
        ax.legend(loc="best")
    ax.set_title("GPU Memory Comparison")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _plot_dashboard(metrics_by_batch: dict[int, list[dict]], path: Path) -> None:
    plt = get_pyplot(path)
    batch_sizes = sorted(metrics_by_batch)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.flatten()

    for batch_size in batch_sizes:
        metrics = metrics_by_batch[batch_size]
        epochs = [int(row["epoch"]) for row in metrics]
        axes[0].plot(epochs, [_as_float(row["val_loss"]) for row in metrics], marker="o", color=_color(batch_size), label=f"batch {batch_size}")
        axes[1].plot(epochs, [_as_float(row["val_ppl"]) for row in metrics], marker="o", color=_color(batch_size), label=f"batch {batch_size}")

    best_losses = [_as_float(best_row(metrics_by_batch[batch_size])["val_loss"]) for batch_size in batch_sizes]
    best_ppl = [_as_float(best_row(metrics_by_batch[batch_size])["val_ppl"]) for batch_size in batch_sizes]
    top1 = [_as_float(best_row(metrics_by_batch[batch_size]).get("test_top_1_accuracy")) * 100 for batch_size in batch_sizes]
    best_epochs = [_best_epoch(metrics_by_batch[batch_size]) for batch_size in batch_sizes]
    total_time = [_as_float(metrics_by_batch[batch_size][-1].get("total_training_time")) for batch_size in batch_sizes]
    final_gap = [
        _as_float(metrics_by_batch[batch_size][-1]["val_loss"]) - _as_float(metrics_by_batch[batch_size][-1]["train_loss"])
        for batch_size in batch_sizes
    ]

    axes[0].set_title("Validation Loss")
    axes[1].set_title("Validation Perplexity")
    axes[2].bar([str(batch_size) for batch_size in batch_sizes], best_losses, color=[_color(batch_size) for batch_size in batch_sizes])
    axes[2].set_title("Best Val Loss")
    axes[3].bar([str(batch_size) for batch_size in batch_sizes], top1, color=[_color(batch_size) for batch_size in batch_sizes])
    axes[3].set_title("Best-Epoch Test Top-1 Accuracy (%)")
    axes[4].bar([str(batch_size) for batch_size in batch_sizes], best_epochs, color=[_color(batch_size) for batch_size in batch_sizes])
    axes[4].set_title("Best Epoch")
    axes[5].plot([str(batch_size) for batch_size in batch_sizes], total_time, marker="o", color="#2563eb", label="training time")
    axes[5].plot([str(batch_size) for batch_size in batch_sizes], final_gap, marker="o", color="#dc2626", label="final loss gap")
    axes[5].set_title("Time and Final Gap")
    axes[5].legend(loc="best")

    for idx, ax in enumerate(axes):
        ax.grid(True, axis="y", alpha=0.45)
        if idx in {0, 1}:
            ax.set_xlabel("epoch")
            ax.legend(loc="best")
        else:
            ax.set_xlabel("batch_size")

    fig.suptitle("Batch Size Experiment Dashboard", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)
    plt.close(fig)


def write_visualizations(root: str | Path) -> None:
    root_path = Path(root)
    graph_dir = root_path / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_batch = load_metrics(root_path)

    for batch_size, metrics in sorted(metrics_by_batch.items()):
        _plot_batch_metric(metrics, batch_size, "loss", "Loss", graph_dir / f"loss_batch_{batch_size}.png")
        _plot_batch_metric(metrics, batch_size, "ppl", "Perplexity", graph_dir / f"ppl_batch_{batch_size}.png")

    for split in ("train", "val", "test"):
        _plot_comparison(metrics_by_batch, "loss", split, "Loss", graph_dir / f"loss_comparison_{split}.png")
        _plot_comparison(metrics_by_batch, "ppl", split, "Perplexity", graph_dir / f"ppl_comparison_{split}.png")

    _plot_topn(metrics_by_batch, "best", graph_dir / "top_n_accuracy_comparison.png")
    _plot_topn(metrics_by_batch, "best", graph_dir / "top_n_accuracy_best_epoch.png")
    _plot_topn(metrics_by_batch, "final", graph_dir / "top_n_accuracy_final_epoch.png")
    _plot_best_epoch(metrics_by_batch, graph_dir / "best_epoch_comparison.png")
    _plot_gap_by_epoch(metrics_by_batch, "loss", graph_dir / "loss_gap_by_epoch.png")
    _plot_gap_by_epoch(metrics_by_batch, "ppl", graph_dir / "ppl_gap_by_epoch.png")
    _plot_final_loss_gap(metrics_by_batch, graph_dir / "final_loss_gap_comparison.png")
    _plot_time(metrics_by_batch, "total_training_time", "Training Time Comparison", "seconds", graph_dir / "training_time_comparison.png")
    _plot_time(metrics_by_batch, "average_epoch_time", "Average Epoch Time Comparison", "seconds", graph_dir / "average_epoch_time_comparison.png")
    _plot_memory(metrics_by_batch, graph_dir / "gpu_memory_comparison.png")
    _plot_dashboard(metrics_by_batch, graph_dir / "batch_size_experiment_dashboard.png")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    write_visualizations(args.root)


if __name__ == "__main__":
    main()
