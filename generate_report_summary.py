# -*- coding: utf-8 -*-
"""Generate narrative summaries from actual experiment artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from analyze_results import best_row, load_metrics


def _as_float(value: Any) -> float:
    if value in {None, "", "N/A"}:
        return float("nan")
    return float(value)


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value)
    if math.isnan(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _winner(rows: list[tuple[int, float]], lower_is_better: bool = True) -> tuple[int, float] | None:
    clean = [(batch_size, value) for batch_size, value in rows if not math.isnan(value)]
    if not clean:
        return None
    return min(clean, key=lambda item: item[1]) if lower_is_better else max(clean, key=lambda item: item[1])


def _sample_quality_comment(output: str) -> str:
    words = output.split()
    if not output.strip():
        return "No generation was produced."
    if len(words) >= 8:
        repeated = sum(1 for left, right in zip(words, words[1:]) if left == right)
        if repeated / max(len(words) - 1, 1) > 0.25:
            return "The output shows noticeable immediate repetition."
    if len(output) < 8:
        return "The output is very short, so quality judgment is limited."
    return "No strong immediate repetition was detected by the automatic check."


def build_report_lines(root: str | Path) -> list[str]:
    root_path = Path(root)
    metrics_by_batch = load_metrics(root_path)
    sample_rows = _read_csv(root_path / "tables" / "sample_generation_results.csv")

    best_loss_candidates = [
        (batch_size, _as_float(best_row(metrics)["val_loss"]))
        for batch_size, metrics in sorted(metrics_by_batch.items())
    ]
    best_test_ppl_candidates = [
        (batch_size, _as_float(best_row(metrics)["test_ppl"]))
        for batch_size, metrics in sorted(metrics_by_batch.items())
    ]
    final_gap_candidates = [
        (
            batch_size,
            abs(_as_float(metrics[-1]["val_loss"]) - _as_float(metrics[-1]["train_loss"])),
        )
        for batch_size, metrics in sorted(metrics_by_batch.items())
    ]
    time_candidates = [
        (batch_size, _as_float(metrics[-1].get("total_training_time")))
        for batch_size, metrics in sorted(metrics_by_batch.items())
    ]

    best_loss = _winner(best_loss_candidates, lower_is_better=True)
    best_test_ppl = _winner(best_test_ppl_candidates, lower_is_better=True)
    most_stable_gap = _winner(final_gap_candidates, lower_is_better=True)
    fastest = _winner(time_candidates, lower_is_better=True)
    slowest = _winner(time_candidates, lower_is_better=False)

    lines = [
        "# Batch Size Experiment Report",
        "",
        "## Loss Graph Interpretation",
    ]
    if best_loss is not None:
        lines.append(
            f"The lowest validation loss was achieved by batch_size {best_loss[0]} "
            f"with best_val_loss={_fmt(best_loss[1])}."
        )
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        best = best_row(metrics)
        final = metrics[-1]
        lines.append(
            f"- batch_size {batch_size}: best_epoch={best['epoch']}, "
            f"best_val_loss={_fmt(best['val_loss'])}, final_val_loss={_fmt(final['val_loss'])}."
        )

    lines.extend(["", "## Perplexity Graph Interpretation"])
    if best_test_ppl is not None:
        lines.append(
            f"The lowest test perplexity at the best validation epoch was batch_size {best_test_ppl[0]} "
            f"with test_ppl_at_best_epoch={_fmt(best_test_ppl[1])}."
        )
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        best = best_row(metrics)
        final = metrics[-1]
        lines.append(
            f"- batch_size {batch_size}: best_val_ppl={_fmt(best['val_ppl'])}, "
            f"final_test_ppl={_fmt(final['test_ppl'])}."
        )

    lines.extend(["", "## Top-N Accuracy Interpretation"])
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        best = best_row(metrics)
        final = metrics[-1]
        lines.append(
            f"- batch_size {batch_size}: best-epoch test top-1/top-3/top-5="
            f"{_fmt(best.get('test_top_1_accuracy'))}/"
            f"{_fmt(best.get('test_top_3_accuracy'))}/"
            f"{_fmt(best.get('test_top_5_accuracy'))}; final="
            f"{_fmt(final.get('test_top_1_accuracy'))}/"
            f"{_fmt(final.get('test_top_3_accuracy'))}/"
            f"{_fmt(final.get('test_top_5_accuracy'))}."
        )

    lines.extend(["", "## Best Epoch And Stability"])
    if most_stable_gap is not None:
        lines.append(
            f"The smallest final validation-train loss gap was batch_size {most_stable_gap[0]} "
            f"with absolute gap={_fmt(most_stable_gap[1])}."
        )
    for batch_size, metrics in sorted(metrics_by_batch.items()):
        final = metrics[-1]
        gap = _as_float(final["val_loss"]) - _as_float(final["train_loss"])
        lines.append(
            f"- batch_size {batch_size}: final_loss_gap={_fmt(gap)}, "
            f"average_epoch_time={_fmt(final.get('average_epoch_time'), 2)} seconds."
        )

    lines.extend(["", "## Training Time And Memory"])
    if fastest is not None and slowest is not None:
        lines.append(
            f"The fastest total run was batch_size {fastest[0]} ({_fmt(fastest[1], 2)} seconds), "
            f"and the slowest was batch_size {slowest[0]} ({_fmt(slowest[1], 2)} seconds)."
        )
    memory_values = [
        metrics[-1].get("max_gpu_memory_allocated", "N/A")
        for metrics in metrics_by_batch.values()
    ]
    if all(value == "N/A" for value in memory_values):
        lines.append("GPU memory metrics are N/A because CUDA was not available or memory tracking was not supported.")

    lines.extend(["", "## Generation Sample Interpretation"])
    if not sample_rows:
        lines.append("No generation sample rows were available.")
    else:
        for row in sample_rows[:12]:
            comment = row.get("simple_evaluation") or _sample_quality_comment(row.get("generated_output", ""))
            lines.append(
                f"- input={row.get('input_sentence', '')[:50]!r}, batch_size={row.get('batch_size')}: {comment}"
            )

    lines.extend(["", "## Recommendation"])
    if best_loss is not None and best_test_ppl is not None:
        if best_loss[0] == best_test_ppl[0]:
            lines.append(
                f"Based on validation loss and test perplexity, batch_size {best_loss[0]} is the primary recommendation."
            )
        else:
            lines.append(
                f"Validation loss favors batch_size {best_loss[0]}, while test perplexity favors batch_size {best_test_ppl[0]}; "
                "use the summary table to choose based on the report's priority metric."
            )
    else:
        lines.append("A recommendation could not be computed because required metrics were missing.")

    return lines


def write_report(root: str | Path) -> None:
    root_path = Path(root)
    lines = build_report_lines(root_path)
    report_text = "\n".join(lines) + "\n"
    (root_path / "report_visual_analysis.md").write_text(report_text, encoding="utf-8")
    (root_path / "report_summary.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    write_report(args.root)


if __name__ == "__main__":
    main()
