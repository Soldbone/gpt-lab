# -*- coding: utf-8 -*-
"""Build comparison tables for the batch-size experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


TOP_N_VALUES = (1, 3, 5)


def _as_float(value: Any) -> float:
    if value in {None, "", "N/A"}:
        return float("nan")
    return float(value)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _as_float(value)
    if math.isnan(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        values = [str(row.get(name, "")).replace("\n", " ") for name in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_metrics(root: str | Path) -> dict[int, list[dict]]:
    metrics_dir = Path(root) / "metrics"
    metrics_by_batch: dict[int, list[dict]] = {}
    for path in sorted(metrics_dir.glob("batch_*_metrics.json")):
        batch_size = int(path.stem.split("_")[1])
        rows = json.loads(path.read_text(encoding="utf-8"))
        metrics_by_batch[batch_size] = sorted(rows, key=lambda row: int(row["epoch"]))
    return metrics_by_batch


def best_row(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: _as_float(row["val_loss"]))


def stable_epoch_range(rows: list[dict]) -> str:
    best = best_row(rows)
    threshold = _as_float(best["val_loss"]) * 1.01
    near_best = [row for row in rows if _as_float(row["val_loss"]) <= threshold]
    if not near_best:
        return "N/A"
    first_epoch = near_best[0]["epoch"]
    last_epoch = near_best[-1]["epoch"]
    return str(first_epoch) if first_epoch == last_epoch else f"{first_epoch}-{last_epoch}"


def build_summary_rows(metrics_by_batch: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        final = metrics[-1]
        best = best_row(metrics)
        rows.append({
            "batch_size": batch_size,
            "best_epoch": best["epoch"],
            "final_train_loss": _fmt(final["train_loss"]),
            "final_val_loss": _fmt(final["val_loss"]),
            "final_test_loss": _fmt(final["test_loss"]),
            "best_val_loss": _fmt(best["val_loss"]),
            "test_loss_at_best_epoch": _fmt(best["test_loss"]),
            "final_train_ppl": _fmt(final["train_ppl"]),
            "final_val_ppl": _fmt(final["val_ppl"]),
            "final_test_ppl": _fmt(final["test_ppl"]),
            "best_val_ppl": _fmt(best["val_ppl"]),
            "test_ppl_at_best_epoch": _fmt(best["test_ppl"]),
            "top_1_accuracy_at_best_epoch": _fmt(best.get("test_top_1_accuracy")),
            "top_3_accuracy_at_best_epoch": _fmt(best.get("test_top_3_accuracy")),
            "top_5_accuracy_at_best_epoch": _fmt(best.get("test_top_5_accuracy")),
            "final_top_1_accuracy": _fmt(final.get("test_top_1_accuracy")),
            "final_top_3_accuracy": _fmt(final.get("test_top_3_accuracy")),
            "final_top_5_accuracy": _fmt(final.get("test_top_5_accuracy")),
            "training_time": _fmt(final.get("total_training_time"), 3),
            "average_epoch_time": _fmt(final.get("average_epoch_time"), 3),
            "max_gpu_memory_allocated_if_available": final.get("max_gpu_memory_allocated", "N/A"),
            "max_gpu_memory_reserved_if_available": final.get("max_gpu_memory_reserved", "N/A"),
            "final_loss_gap": _fmt(_as_float(final["val_loss"]) - _as_float(final["train_loss"])),
            "final_ppl_gap": _fmt(_as_float(final["val_ppl"]) - _as_float(final["train_ppl"])),
        })
    return rows


def build_best_epoch_rows(metrics_by_batch: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        best = best_row(metrics)
        rows.append({
            "batch_size": batch_size,
            "best_epoch": best["epoch"],
            "best_val_loss": _fmt(best["val_loss"]),
            "best_val_ppl": _fmt(best["val_ppl"]),
            "test_loss_at_best_epoch": _fmt(best["test_loss"]),
            "test_ppl_at_best_epoch": _fmt(best["test_ppl"]),
            "top_1_accuracy_at_best_epoch": _fmt(best.get("test_top_1_accuracy")),
            "top_3_accuracy_at_best_epoch": _fmt(best.get("test_top_3_accuracy")),
            "top_5_accuracy_at_best_epoch": _fmt(best.get("test_top_5_accuracy")),
            "stable_epoch_range_if_available": stable_epoch_range(metrics),
        })
    return rows


def _rank_rows(rows: list[dict], key: str, descending: bool = True) -> dict[int, int]:
    ordered = sorted(rows, key=lambda row: _as_float(row[key]), reverse=descending)
    return {int(row["batch_size"]): rank for rank, row in enumerate(ordered, start=1)}


def build_top_n_ranking_rows(metrics_by_batch: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        best = best_row(metrics)
        final = metrics[-1]
        row = {
            "batch_size": batch_size,
            "best_epoch": best["epoch"],
        }
        for top_n in TOP_N_VALUES:
            row[f"top_{top_n}_accuracy_at_best_epoch"] = _fmt(best.get(f"test_top_{top_n}_accuracy"))
        for top_n in TOP_N_VALUES:
            row[f"final_top_{top_n}_accuracy"] = _fmt(final.get(f"test_top_{top_n}_accuracy"))
        rows.append(row)

    for top_n in TOP_N_VALUES:
        ranks = _rank_rows(rows, f"top_{top_n}_accuracy_at_best_epoch", descending=True)
        for row in rows:
            row[f"rank_by_top_{top_n}"] = ranks[int(row["batch_size"])]
    return rows


def build_training_time_rows(metrics_by_batch: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        final = metrics[-1]
        rows.append({
            "batch_size": batch_size,
            "total_training_time": _fmt(final.get("total_training_time"), 3),
            "average_epoch_time": _fmt(final.get("average_epoch_time"), 3),
        })
    return rows


def build_overfitting_rows(metrics_by_batch: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        final = metrics[-1]
        loss_gap = _as_float(final["val_loss"]) - _as_float(final["train_loss"])
        ppl_gap = _as_float(final["val_ppl"]) - _as_float(final["train_ppl"])
        if math.isnan(loss_gap):
            comment = "N/A"
        elif loss_gap > 0.2:
            comment = "Validation loss is meaningfully higher than train loss; monitor overfitting."
        elif loss_gap < -0.05:
            comment = "Validation loss is lower than train loss; check sampling/evaluation variance."
        else:
            comment = "Train and validation losses are close."
        rows.append({
            "batch_size": batch_size,
            "final_train_loss": _fmt(final["train_loss"]),
            "final_val_loss": _fmt(final["val_loss"]),
            "loss_gap": _fmt(loss_gap),
            "final_train_ppl": _fmt(final["train_ppl"]),
            "final_val_ppl": _fmt(final["val_ppl"]),
            "ppl_gap": _fmt(ppl_gap),
            "overfitting_comment": comment,
        })
    return rows


def write_sample_generation_rows(root: str | Path, sample_rows: list[dict] | None) -> None:
    if sample_rows is None:
        sample_rows = []
    table_dir = Path(root) / "tables"
    _write_csv(table_dir / "sample_generation_results.csv", sample_rows)
    _write_markdown(table_dir / "sample_generation_results.md", sample_rows)


def write_analysis_outputs(root: str | Path, sample_rows: list[dict] | None = None) -> dict[str, list[dict]]:
    root_path = Path(root)
    table_dir = root_path / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_batch = load_metrics(root_path)

    outputs = {
        "summary_by_batch_size": build_summary_rows(metrics_by_batch),
        "best_epoch_summary": build_best_epoch_rows(metrics_by_batch),
        "top_n_accuracy_ranking": build_top_n_ranking_rows(metrics_by_batch),
        "training_time_summary": build_training_time_rows(metrics_by_batch),
        "overfitting_analysis": build_overfitting_rows(metrics_by_batch),
    }

    for name, rows in outputs.items():
        _write_csv(table_dir / f"{name}.csv", rows)
        _write_markdown(table_dir / f"{name}.md", rows)
    write_sample_generation_rows(root_path, sample_rows)
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    write_analysis_outputs(args.root)


if __name__ == "__main__":
    main()
