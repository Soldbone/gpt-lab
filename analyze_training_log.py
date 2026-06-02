# -*- coding: utf-8 -*-
"""Summarize pretrain.py loss logs with perplexity."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


LOSS_PATTERN = re.compile(
    r"에포크 (\d+) \(Step (\d+)\): 훈련 손실 ([0-9.]+), 검증 손실 ([0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Parse the whole file instead of only the last pretrain.py run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log is None:
        default_log = Path("train.log")
        if default_log.exists():
            args.log = default_log
        else:
            raise SystemExit("로그 파일 경로를 지정하세요. 예: python analyze_training_log.py --log train.log")
    if not args.log.exists():
        raise SystemExit(f"로그 파일을 찾을 수 없습니다: {args.log}")

    text = args.log.read_text(encoding="utf-8")
    if not args.all_runs:
        start = text.rfind("python pretrain.py")
        if start != -1:
            text = text[start:]

    rows = [
        (int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)))
        for m in LOSS_PATTERN.finditer(text)
    ]
    if not rows:
        raise SystemExit("No pretraining loss rows found.")

    first = rows[0]
    last = rows[-1]
    best_val = min(rows, key=lambda row: row[3])
    best_train = min(rows, key=lambda row: row[2])

    print(f"eval points: {len(rows)}")
    print(f"first: epoch={first[0]}, step={first[1]:,}, train_loss={first[2]:.3f}, val_loss={first[3]:.3f}")
    print(f"last: epoch={last[0]}, step={last[1]:,}, train_loss={last[2]:.3f}, val_loss={last[3]:.3f}")
    print(f"best val: epoch={best_val[0]}, step={best_val[1]:,}, val_loss={best_val[3]:.3f}")
    print(f"best train: epoch={best_train[0]}, step={best_train[1]:,}, train_loss={best_train[2]:.3f}")
    print(f"last train perplexity: {math.exp(last[2]):.2f}")
    print(f"last val perplexity: {math.exp(last[3]):.2f}")
    print(f"best val perplexity: {math.exp(best_val[3]):.2f}")
    print(f"generalization gap: {last[3] - last[2]:.3f}")


if __name__ == "__main__":
    main()
