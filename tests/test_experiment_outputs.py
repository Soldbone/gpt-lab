# -*- coding: utf-8 -*-
"""Smoke tests for analysis, visualization, and report artifact generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _metric_rows(batch_size: int) -> list[dict]:
    rows = []
    for epoch in (1, 2):
        base = 3.0 - 0.1 * epoch + batch_size * 0.005
        rows.append({
            "batch_size": batch_size,
            "seed": 42,
            "epoch": epoch,
            "global_step": epoch * 10,
            "train_loss": base - 0.08,
            "val_loss": base,
            "test_loss": base + 0.04,
            "train_ppl": 10.0 + batch_size - epoch,
            "val_ppl": 11.0 + batch_size - epoch,
            "test_ppl": 12.0 + batch_size - epoch,
            "train_top_1_accuracy": 0.10 + epoch * 0.01,
            "train_top_3_accuracy": 0.20 + epoch * 0.01,
            "train_top_5_accuracy": 0.30 + epoch * 0.01,
            "val_top_1_accuracy": 0.09 + epoch * 0.01,
            "val_top_3_accuracy": 0.19 + epoch * 0.01,
            "val_top_5_accuracy": 0.29 + epoch * 0.01,
            "test_top_1_accuracy": 0.08 + epoch * 0.01,
            "test_top_3_accuracy": 0.18 + epoch * 0.01,
            "test_top_5_accuracy": 0.28 + epoch * 0.01,
            "top_1_accuracy": 0.08 + epoch * 0.01,
            "top_3_accuracy": 0.18 + epoch * 0.01,
            "top_5_accuracy": 0.28 + epoch * 0.01,
            "epoch_time": 1.0 + batch_size * 0.1,
            "total_training_time": epoch * (1.0 + batch_size * 0.1),
            "average_epoch_time": 1.0 + batch_size * 0.1,
            "max_gpu_memory_allocated": "N/A",
            "max_gpu_memory_reserved": "N/A",
            "memory_status": "CUDA not available",
        })
    return rows


def _lr_metric_rows(learning_rate: float) -> list[dict]:
    rows = []
    lr_scale = learning_rate * 10_000
    for epoch in (1, 2):
        base = 3.0 - 0.1 * epoch + lr_scale * 0.01
        rows.append({
            "learning_rate": learning_rate,
            "learning_rate_label": f"{learning_rate:.0e}".replace("e-0", "e-"),
            "batch_size": 8,
            "seed": 42,
            "epoch": epoch,
            "global_step": epoch * 10,
            "train_loss": base - 0.08,
            "val_loss": base,
            "test_loss": base + 0.04,
            "train_ppl": 10.0 + lr_scale - epoch,
            "val_ppl": 11.0 + lr_scale - epoch,
            "test_ppl": 12.0 + lr_scale - epoch,
            "train_top_1_accuracy": 0.10 + epoch * 0.01,
            "train_top_3_accuracy": 0.20 + epoch * 0.01,
            "train_top_5_accuracy": 0.30 + epoch * 0.01,
            "val_top_1_accuracy": 0.09 + epoch * 0.01,
            "val_top_3_accuracy": 0.19 + epoch * 0.01,
            "val_top_5_accuracy": 0.29 + epoch * 0.01,
            "test_top_1_accuracy": 0.08 + epoch * 0.01,
            "test_top_3_accuracy": 0.18 + epoch * 0.01,
            "test_top_5_accuracy": 0.28 + epoch * 0.01,
            "top_1_accuracy": 0.08 + epoch * 0.01,
            "top_3_accuracy": 0.18 + epoch * 0.01,
            "top_5_accuracy": 0.28 + epoch * 0.01,
            "epoch_time": 1.0 + lr_scale * 0.1,
            "total_training_time": epoch * (1.0 + lr_scale * 0.1),
            "average_epoch_time": 1.0 + lr_scale * 0.1,
            "max_gpu_memory_allocated": "N/A",
            "max_gpu_memory_reserved": "N/A",
            "memory_status": "CUDA not available",
        })
    return rows


def test_analysis_visualization_and_report_outputs_from_synthetic_metrics(tmp_path):
    from analyze_results import write_analysis_outputs
    from generate_report_summary import write_report
    from visualize_results import write_visualizations

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for batch_size in (4, 8, 16):
        (metrics_dir / f"batch_{batch_size}_metrics.json").write_text(
            json.dumps(_metric_rows(batch_size), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    sample_rows = [
        {
            "input_sentence": "sample input",
            "batch_size": 4,
            "generated_output": "sample generated output",
            "target_sentence": "sample target",
            "expected_output": "sample target",
            "loss": 1.23,
            "score": 1.23,
            "simple_evaluation": "Automatic check found no strong immediate repetition.",
        }
    ]

    write_analysis_outputs(tmp_path, sample_rows=sample_rows)
    write_visualizations(tmp_path)
    write_report(tmp_path)

    assert (tmp_path / "tables" / "summary_by_batch_size.md").exists()
    assert (tmp_path / "tables" / "best_epoch_summary.csv").exists()
    assert (tmp_path / "graphs" / "loss_comparison_val.png").exists()
    assert (tmp_path / "graphs" / "batch_size_experiment_dashboard.svg").exists()
    assert (tmp_path / "report_visual_analysis.md").exists()


def test_learning_rate_analysis_visualization_and_report_outputs_from_synthetic_metrics(tmp_path):
    from analyze_learning_rate_results import format_learning_rate, write_analysis_outputs
    from generate_learning_rate_report_summary import write_report
    from visualize_learning_rate_results import write_visualizations

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for learning_rate in (1e-4, 3e-4, 5e-4):
        lr_label = format_learning_rate(learning_rate)
        (metrics_dir / f"lr_{lr_label}_metrics.json").write_text(
            json.dumps(_lr_metric_rows(learning_rate), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    sample_rows = [
        {
            "input_sentence": "sample input",
            "learning_rate": 1e-4,
            "learning_rate_label": "1e-4",
            "batch_size": 8,
            "generated_output": "sample generated output",
            "target_sentence": "sample target",
            "expected_output": "sample target",
            "loss": 1.23,
            "score": 1.23,
            "simple_evaluation": "Automatic check found no strong immediate repetition.",
        }
    ]

    write_analysis_outputs(tmp_path, sample_rows=sample_rows)
    write_visualizations(tmp_path)
    write_report(tmp_path)

    assert (tmp_path / "tables" / "summary_by_learning_rate.md").exists()
    assert (tmp_path / "tables" / "best_epoch_summary.csv").exists()
    assert (tmp_path / "graphs" / "loss_comparison_val.png").exists()
    assert (tmp_path / "graphs" / "learning_rate_experiment_dashboard.svg").exists()
    assert (tmp_path / "report_visual_analysis.md").exists()
