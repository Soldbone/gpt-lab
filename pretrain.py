# -*- coding: utf-8 -*-
"""Run mini GPT pretraining on the prepared NSMC language-modeling text."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import textwrap
from pathlib import Path
from datetime import datetime

import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.train import calc_loss_loader, generate, load_checkpoint, save_checkpoint, train_model

ROOT = Path(__file__).resolve().parent

DEFAULT_HYPERPARAMS = {
    #기본 파라미터:vocab_size:3000, epoch:10, context_length:128,
    #corphus:1,500,000, emb_dim:128, drop_rate:0.1
    # layer:2,4,6, lr:3e-4,1e-4,5e-4 batch_size:4,8,16 n_heads:3,4,6
    "train_path": ROOT / "data" / "nsmc_lm_train.txt",
    "val_path": ROOT / "data" / "nsmc_lm_val.txt",
    "test_path": ROOT / "data" / "nsmc_sentiment_test.jsonl",
    "tokenizer_path": ROOT / "checkpoints" / "bpe_vocab_3000.json",
    "output_dir": ROOT / "checkpoints" / "pretrain_from_scratch",
    "results_dir": ROOT / "results",
    "vocab_size": 3000,
    "context_length": 128,
    "stride": None,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "batch_size": 2,
    "num_workers": 0,
    "epochs": 1,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "eval_freq": 500,
    "eval_iter": 20,
    "ckpt_freq": 1000,
    "resume": None,
    "device": "auto",
    "seed": 42,
    "start_context": "영화",
    "max_train_chars": None,
    "max_val_chars": None,
    "max_test_chars": None,
    "report_eval_iter": None,
    "top_k": "1,5,10",
    "report_sample_tokens": 40,
}


# ─────────────────────────────────────────────────────────────────────────────
# Palette & style constants
# ─────────────────────────────────────────────────────────────────────────────

# Core palette
C_TRAIN  = "#3B82F6"   # blue-500
C_VAL    = "#10B981"   # emerald-500
C_TEST   = "#F59E0B"   # amber-400
C_BEST   = "#8B5CF6"   # violet-500
C_SPAN   = "#FCD34D"   # amber-300

# Neutral surface palette
BG_PAGE  = "#F8F9FA"
BG_AXES  = "#FFFFFF"
BG_GRID  = "#F1F3F5"
COL_EDGE = "#DEE2E6"
COL_TICK = "#6C757D"
COL_TEXT = "#212529"
COL_SUB  = "#6C757D"

SPLIT_COLORS = {"train": C_TRAIN, "val": C_VAL, "test": C_TEST}
SPLIT_LABELS = {"train": "Train", "val": "Val", "test": "Test"}

MARKER_STYLE = dict(markersize=5, markeredgewidth=0.8, markeredgecolor="white")
LINE_STYLE   = dict(linewidth=2.2, solid_capstyle="round", solid_joinstyle="round")
ANNOT_BOX    = dict(boxstyle="round,pad=0.45", fc="white", ec=COL_EDGE, alpha=0.95, lw=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def get_pyplot(path: Path):
    """Return a configured matplotlib.pyplot instance."""
    mpl_config_root = (
        path.parent.parent
        if path.parent.name.startswith("pretrain_")
        else path.parent
    )
    mpl_config_dir = mpl_config_root / ".matplotlib"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Korean font detection
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
            if name in available:
                plt.rcParams["font.family"] = name
                break
    except Exception:
        pass

    plt.rcParams.update({
        # Figure
        "figure.facecolor":     BG_PAGE,
        "figure.dpi":           150,
        # Axes
        "axes.facecolor":       BG_AXES,
        "axes.edgecolor":       COL_EDGE,
        "axes.linewidth":       0.8,
        "axes.labelcolor":      COL_TEXT,
        "axes.labelsize":       12,
        "axes.labelpad":        8,
        "axes.titlesize":       15,
        "axes.titleweight":     "semibold",
        "axes.titlepad":        14,
        "axes.titlecolor":      COL_TEXT,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        # Grid
        "axes.grid":            True,
        "grid.color":           BG_GRID,
        "grid.linewidth":       1.0,
        "grid.linestyle":       "-",
        # Ticks
        "xtick.color":          COL_TICK,
        "ytick.color":          COL_TICK,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.major.pad":      5,
        "ytick.major.pad":      5,
        # Legend
        "legend.fontsize":      10,
        "legend.framealpha":    0.95,
        "legend.edgecolor":     COL_EDGE,
        "legend.fancybox":      False,
        "legend.borderpad":     0.6,
        "legend.labelspacing":  0.4,
        # Lines
        "lines.antialiased":    True,
        # Misc
        "axes.unicode_minus":   False,
        "savefig.facecolor":    BG_PAGE,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.25,
    })
    return plt


def _finish(fig, path: Path, plt) -> None:
    """Save and close a figure."""
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / PPL curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_curve(
    history: list[dict],
    metric_prefix: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt = get_pyplot(path)
    steps  = [row["step"] for row in history]
    fig, ax = plt.subplots(figsize=(11, 6))

    use_log = False
    if metric_prefix == "ppl":
        finite = [
            row[f"{split}_{metric_prefix}"]
            for row in history
            for split in ("train", "val", "test")
            if not math.isnan(row[f"{split}_{metric_prefix}"])
        ]
        if finite and max(finite) / max(min(finite), 1e-9) > 20:
            use_log = True

    for split in ("train", "val", "test"):
        key    = f"{split}_{metric_prefix}"
        values = [row[key] for row in history]
        color  = SPLIT_COLORS[split]
        label  = SPLIT_LABELS[split]

        ax.plot(
            steps, values,
            label=label,
            color=color,
            marker="o",
            **MARKER_STYLE,
            **LINE_STYLE,
        )

        # End-point label
        last_v = values[-1]
        label_text = f"{label}: {last_v:.3f}"
        ax.annotate(
            label_text,
            xy=(steps[-1], last_v),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9.5,
            color=color,
            fontweight="semibold",
            annotation_clip=False,
        )

    if use_log:
        ax.set_yscale("log")
        ylabel = f"{ylabel} (log scale)"

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)

    # Subtle fill under train curve
    train_vals = [row[f"train_{metric_prefix}"] for row in history]
    ax.fill_between(steps, train_vals, alpha=0.06, color=C_TRAIN)

    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(left=0)

    # Right-margin space for end labels
    x_range = steps[-1] - steps[0] if len(steps) > 1 else steps[0]
    ax.set_xlim(right=steps[-1] + x_range * 0.14)

    fig.tight_layout()
    _finish(fig, path, plt)


# ─────────────────────────────────────────────────────────────────────────────
# Top-N accuracy bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_topn_accuracy(top_rows: list[dict], path: Path) -> None:
    plt = get_pyplot(path)
    datasets   = ["train", "val", "test"]
    top_values = sorted({int(row["top_n"]) for row in top_rows})
    score_map  = {
        (row["dataset"], int(row["top_n"])): float(row["accuracy"]) * 100
        for row in top_rows
    }

    n_groups  = len(top_values)
    n_splits  = len(datasets)
    width     = 0.22
    gap       = 0.06
    positions = [i * (n_splits * (width + gap) + 0.3) for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=(max(9, n_groups * 3.5), 6))
    all_scores: list[float] = []

    for di, dataset in enumerate(datasets):
        color  = SPLIT_COLORS[dataset]
        label  = SPLIT_LABELS[dataset]
        offset = [p + di * (width + gap) for p in positions]
        scores = [score_map.get((dataset, k), float("nan")) for k in top_values]
        plot_s = [0.0 if math.isnan(s) else s for s in scores]
        all_scores.extend(plot_s)

        bars = ax.bar(
            offset, plot_s,
            width=width,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        for bar, score in zip(bars, scores):
            txt = "n/a" if math.isnan(score) else f"{score:.1f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                txt,
                ha="center", va="bottom",
                fontsize=9, color=color, fontweight="semibold",
            )

    tick_centers = [p + (n_splits - 1) * (width + gap) / 2 for p in positions]
    ax.set_xticks(tick_centers)
    ax.set_xticklabels([f"Top-{k}" for k in top_values], fontsize=11)

    max_s = max(all_scores) if all_scores else 0
    ax.set_ylim(0, min(100, max(1.0, max_s * 1.25 + 1)))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-N Next-Token Accuracy")
    ax.legend(loc="upper left", frameon=True)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _finish(fig, path, plt)


# ─────────────────────────────────────────────────────────────────────────────
# Best epoch / overfit diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def plot_best_epoch_range(history: list[dict], path: Path) -> None:
    plt = get_pyplot(path)
    best      = min(history, key=lambda r: r["val_loss"])
    threshold = best["val_loss"] * 1.01
    near_best = [r for r in history if r["val_loss"] <= threshold]
    first, last = near_best[0], near_best[-1]
    overfit   = history[-1]["val_loss"] > best["val_loss"] * 1.02
    steps     = [r["step"] for r in history]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Near-best span
    if first["step"] == last["step"]:
        ax.axvline(best["step"], color=C_SPAN, linestyle="--", linewidth=2.0,
                   label="best checkpoint", zorder=2)
    else:
        ax.axvspan(first["step"], last["step"], color=C_SPAN, alpha=0.18,
                   label="within 1 % of best val", zorder=1)

    # Loss curves
    for split in ("train", "val", "test"):
        vals  = [r[f"{split}_loss"] for r in history]
        color = SPLIT_COLORS[split]
        ax.plot(steps, vals,
                label=SPLIT_LABELS[split],
                color=color,
                marker="o",
                **MARKER_STYLE,
                **LINE_STYLE,
                zorder=3)
        ax.fill_between(steps, vals, alpha=0.04, color=color)

    # Best val marker
    ax.scatter([best["step"]], [best["val_loss"]],
               color=C_BEST, s=130, zorder=5,
               edgecolors="white", linewidths=1.5,
               label="best val loss")

    ax.annotate(
        f"Epoch {best['epoch']}  step {best['step']}\nval loss {best['val_loss']:.4f}",
        xy=(best["step"], best["val_loss"]),
        xytext=(28, 28),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.5", fc="#F5F3FF", ec=C_BEST, alpha=0.95, lw=0.8),
        fontsize=10,
        color="#3B0764",
        fontweight="semibold",
    )

    status_txt = "⚠ Overfitting detected" if overfit else "✓ No overfit signal"
    status_col = "#B45309" if overfit else "#065F46"
    status_bg  = "#FFFBEB" if overfit else "#ECFDF5"
    status_ec  = "#FCD34D" if overfit else "#6EE7B7"

    ax.text(
        0.985, 0.04,
        (
            f"Recommended: epoch {best['epoch']} / step {best['step']}\n"
            f"Near-best range: step {first['step']} – {last['step']}\n"
            f"{status_txt}"
        ),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9.5,
        color=status_col,
        bbox=dict(boxstyle="round,pad=0.55", fc=status_bg,
                  ec=status_ec, alpha=0.97, lw=0.8),
    )

    ax.set_title("Best Checkpoint & Overfit Diagnostic")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(left=0)

    fig.tight_layout()
    _finish(fig, path, plt)


# ─────────────────────────────────────────────────────────────────────────────
# PNG table
# ─────────────────────────────────────────────────────────────────────────────

def is_png_safe_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x20 <= code <= 0x7E
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xAC00 <= code <= 0xD7A3
        or ch in "·…\u201c\u201d\u2018\u2019"
    )


def wrap_table_cell(value, width: int = 32, max_lines: int = 5) -> str:
    text = str(value).replace("\uFFFD", "?").replace("\n", " ")
    text = "".join(ch if ch.isprintable() and is_png_safe_char(ch) else "?" for ch in text)
    lines = textwrap.wrap(text, width=width, break_long_words=True) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "…"
    return "\n".join(lines)


def write_png_table(
    path: Path,
    rows: list[dict],
    title: str,
    fieldnames: list[str] | None = None,
    wrap_width: int = 32,
    max_lines: int = 5,
) -> None:
    plt = get_pyplot(path)

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.45, "No rows", ha="center", va="center", fontsize=12, color=COL_SUB)
        fig.tight_layout()
        _finish(fig, path, plt)
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    cell_text = [
        [wrap_table_cell(row.get(name, ""), width=wrap_width, max_lines=max_lines)
         for name in fieldnames]
        for row in rows
    ]
    row_heights = [
        max(str(cell).count("\n") + 1 for cell in row)
        for row in cell_text
    ]

    fig_w = min(max(9.0, len(fieldnames) * 2.0), 18.0)
    fig_h = min(max(2.8, 1.2 + sum(max(1.0, h * 0.44) for h in row_heights)), 28.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, pad=16)

    table = ax.table(
        cellText=cell_text,
        colLabels=fieldnames,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.5)

    # Alternate row colors
    row_bg_even = "#FFFFFF"
    row_bg_odd  = "#F8FAFC"
    hdr_bg      = "#EEF2FF"
    hdr_fg      = "#1E3A5F"

    for (ri, ci), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor(COL_EDGE)
        if ri == 0:
            cell.set_facecolor(hdr_bg)
            cell.set_text_props(weight="semibold", color=hdr_fg, fontsize=9.5)
            cell.set_height(0.085)
        else:
            cell.set_facecolor(row_bg_even if ri % 2 == 1 else row_bg_odd)
            cell.set_height(max(0.08, row_heights[ri - 1] * 0.052))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
    _finish(fig, path, plt)


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged utility / training functions
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible training runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's per-worker initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def read_text(path: Path, max_chars: int | None) -> str:
    if path.suffix == ".jsonl":
        texts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = (item.get("text") or "").strip()
                if text:
                    texts.append(text)
        text = "\n".join(texts)
    else:
        text = path.read_text(encoding="utf-8")
    if max_chars is not None:
        return text[:max_chars]
    return text


def build_tokenizer(args: argparse.Namespace, train_text: str) -> BPETokenizer:
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = args.output_dir / f"bpe_vocab_{args.vocab_size}.json"
    tokenizer_path = tokenizer_path.resolve()
    args.tokenizer_path = tokenizer_path

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    if (args.use_existing_tokenizer or tokenizer_path.exists()) and not args.retrain_tokenizer:
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        args.tokenizer_mode = "loaded-existing"
        print(f"Tokenizer mode: {args.tokenizer_mode}")
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer: {tokenizer_path}")
        return tokenizer

    args.tokenizer_mode = "train-from-scratch"
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Tokenizer mode: {args.tokenizer_mode}")
    print(f"Training tokenizer on {len(train_text):,} characters from {args.train_path}")
    tokenizer.train(train_text)
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer: {tokenizer_path}")
    return tokenizer


def parse_args() -> argparse.Namespace:
    defaults = DEFAULT_HYPERPARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=defaults["train_path"])
    parser.add_argument("--val-path", type=Path, default=defaults["val_path"])
    parser.add_argument("--test-path", type=Path, default=defaults["test_path"])
    parser.add_argument("--tokenizer-path", type=Path, default=defaults["tokenizer_path"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--results-dir", type=Path, default=defaults["results_dir"])
    parser.add_argument("--use-existing-tokenizer", action="store_true")
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=defaults["vocab_size"])
    parser.add_argument("--context-length", type=int, default=defaults["context_length"])
    parser.add_argument("--stride", type=int, default=defaults["stride"])
    parser.add_argument("--emb-dim", type=int, default=defaults["emb_dim"])
    parser.add_argument("--n-heads", type=int, default=defaults["n_heads"])
    parser.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    parser.add_argument("--drop-rate", type=float, default=defaults["drop_rate"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--eval-freq", type=int, default=defaults["eval_freq"])
    parser.add_argument("--eval-iter", type=int, default=defaults["eval_iter"])
    parser.add_argument("--ckpt-freq", type=int, default=defaults["ckpt_freq"])
    parser.add_argument("--resume", type=Path, default=defaults["resume"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--start-context", type=str, default=defaults["start_context"])
    parser.add_argument("--max-train-chars", type=int, default=defaults["max_train_chars"])
    parser.add_argument("--max-val-chars", type=int, default=defaults["max_val_chars"])
    parser.add_argument("--max-test-chars", type=int, default=defaults["max_test_chars"])
    parser.add_argument("--report-eval-iter", type=int, default=defaults["report_eval_iter"])
    parser.add_argument("--top-k", type=str, default=defaults["top_k"])
    parser.add_argument("--report-sample-tokens", type=int, default=defaults["report_sample_tokens"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def print_summary(args: argparse.Namespace, train_losses: list[float], val_losses: list[float]) -> None:
    print("\n=== Training summary ===")
    print(
        "Hyperparameters: "
        f"vocab={args.vocab_size}, context={args.context_length}, "
        f"emb={args.emb_dim}, layers={args.n_layers}, heads={args.n_heads}, "
        f"drop={args.drop_rate}, batch={args.batch_size}, epochs={args.epochs}, "
        f"lr={args.lr}, seed={args.seed}"
    )
    if not train_losses or not val_losses:
        print("No evaluation losses were recorded. Lower --eval-freq or train longer.")
        return

    last_train = train_losses[-1]
    last_val   = val_losses[-1]
    best_val   = min(val_losses)
    print(f"Eval points: {len(train_losses)}")
    print(f"Final train loss: {last_train:.3f}")
    print(f"Final val loss:   {last_val:.3f}")
    print(f"Best val loss:    {best_val:.3f}")
    print(f"Final val perplexity: {math.exp(last_val):.2f}")
    print(f"Best val perplexity:  {math.exp(best_val):.2f}")


def save_run_config(
    args: argparse.Namespace,
    train_text: str,
    val_text: str,
    test_text: str,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    train_batches: int,
    val_batches: int,
    test_batches: int,
    model_config: dict,
    num_params: int,
    device: torch.device,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_dir / "run_config.json"
    payload = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "test_path": str(args.test_path),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_mode": getattr(args, "tokenizer_mode", "unknown"),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "test_chars": len(test_text),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "test_tokens": len(test_ids),
        "train_batches": train_batches,
        "val_batches": val_batches,
        "test_batches": test_batches,
        "model_config": model_config,
        "num_params": num_params,
        "device": str(device),
        "hyperparameters": {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "stride": args.stride,
            "emb_dim": args.emb_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "drop_rate": args.drop_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "weight_decay": args.weight_decay,
            "eval_freq": args.eval_freq,
            "eval_iter": args.eval_iter,
            "report_eval_iter": args.report_eval_iter,
            "ckpt_freq": args.ckpt_freq,
        },
    }
    run_config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved run config: {run_config_path}")


def parse_top_k_values(value: str) -> list[int]:
    top_k_values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            top_k_values.append(int(item))
    if not top_k_values:
        raise ValueError("--top-k must contain at least one integer.")
    return sorted(set(top_k_values))


def safe_ppl(loss: float) -> float:
    if math.isnan(loss):
        return float("nan")
    return math.exp(min(loss, 50.0))


def make_results_run_dir(results_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = results_dir / f"pretrain_{timestamp}"
    run_dir = base
    counter = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{counter}")
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        values = [str(row.get(name, "")).replace("\n", " ") for name in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_topk_accuracy(
    model: GPTModel,
    data_loader,
    device: torch.device,
    top_k_values: list[int],
    num_batches: int | None,
) -> dict[int, float]:
    max_k   = max(top_k_values)
    correct = {k: 0 for k in top_k_values}
    total   = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            input_batch  = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits       = model(input_batch)
            _, top_idx   = torch.topk(logits, k=max_k, dim=-1)
            target       = target_batch.unsqueeze(-1)
            for k in top_k_values:
                correct[k] += top_idx[..., :k].eq(target).any(dim=-1).sum().item()
            total += target_batch.numel()

    model.train()
    if total == 0:
        return {k: float("nan") for k in top_k_values}
    return {k: correct[k] / total for k in top_k_values}


def generate_sample_text(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    max_new_tokens: int,
) -> str:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return ""
    idx = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=context_length,
            temperature=0.8,
            top_k=40,
        )
    model.train()
    try:
        return tokenizer.decode(output.squeeze(0).tolist(), errors="replace").replace("\n", " ")
    except TypeError:
        return tokenizer.decode(output.squeeze(0).tolist()).replace("\n", " ")


def make_best_epoch_rows(history: list[dict]) -> list[dict]:
    best      = min(history, key=lambda r: r["val_loss"])
    threshold = best["val_loss"] * 1.01
    near_best = [r for r in history if r["val_loss"] <= threshold]
    first, last = near_best[0], near_best[-1]
    overfit   = history[-1]["val_loss"] > best["val_loss"] * 1.02
    return [
        {
            "metric":   "best_val_loss",
            "epoch":    best["epoch"],
            "step":     best["step"],
            "train_loss": f"{best['train_loss']:.6f}",
            "val_loss":   f"{best['val_loss']:.6f}",
            "test_loss":  f"{best['test_loss']:.6f}",
            "val_ppl":    f"{best['val_ppl']:.6f}",
            "note":     "recommended checkpoint",
        },
        {
            "metric":   "near_best_1_percent_start",
            "epoch":    first["epoch"],
            "step":     first["step"],
            "train_loss": f"{first['train_loss']:.6f}",
            "val_loss":   f"{first['val_loss']:.6f}",
            "test_loss":  f"{first['test_loss']:.6f}",
            "val_ppl":    f"{first['val_ppl']:.6f}",
            "note":     "first eval within 1% of best val loss",
        },
        {
            "metric":   "near_best_1_percent_end",
            "epoch":    last["epoch"],
            "step":     last["step"],
            "train_loss": f"{last['train_loss']:.6f}",
            "val_loss":   f"{last['val_loss']:.6f}",
            "test_loss":  f"{last['test_loss']:.6f}",
            "val_ppl":    f"{last['val_ppl']:.6f}",
            "note":     f"overfit_flag={overfit}",
        },
    ]


def write_training_results(
    args: argparse.Namespace,
    model: GPTModel,
    tokenizer: BPETokenizer,
    device: torch.device,
    history: list[dict],
    train_loader,
    val_loader,
    test_loader,
    train_text: str,
    val_text: str,
    test_text: str,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    num_params: int,
) -> Path:
    if not history:
        raise ValueError("No metric history to write.")

    run_dir     = make_results_run_dir(args.results_dir)
    report_iter = args.report_eval_iter if args.report_eval_iter is not None else args.eval_iter
    top_k_values = parse_top_k_values(args.top_k)

    # ── metrics_history ───────────────────────────────────────────────────────
    history_rows = [
        {
            "epoch":      row["epoch"],
            "step":       row["step"],
            "train_loss": f"{row['train_loss']:.6f}",
            "val_loss":   f"{row['val_loss']:.6f}",
            "test_loss":  f"{row['test_loss']:.6f}",
            "train_ppl":  f"{row['train_ppl']:.6f}",
            "val_ppl":    f"{row['val_ppl']:.6f}",
            "test_ppl":   f"{row['test_ppl']:.6f}",
        }
        for row in history
    ]
    write_csv(run_dir / "metrics_history.csv", history_rows)
    write_png_table(run_dir / "metrics_history.png", history_rows, "Metrics History")

    # ── curves ────────────────────────────────────────────────────────────────
    plot_metric_curve(history, "loss", "Cross-entropy loss",
                      "Train / Val / Test — Loss", run_dir / "loss_curve.png")
    plot_metric_curve(history, "ppl", "Perplexity",
                      "Train / Val / Test — Perplexity", run_dir / "perplexity_curve.png")

    # ── top-N accuracy ────────────────────────────────────────────────────────
    top_rows = []
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        scores = compute_topk_accuracy(model, loader, device, top_k_values, report_iter)
        for k in top_k_values:
            top_rows.append({"dataset": split, "top_n": k, "accuracy": f"{scores[k]:.6f}"})
    write_csv(run_dir / "top_n_accuracy.csv", top_rows)
    write_markdown_table(run_dir / "top_n_accuracy.md", top_rows)
    plot_topn_accuracy(top_rows, run_dir / "top_n_accuracy.png")

    # ── best epoch ────────────────────────────────────────────────────────────
    best_rows = make_best_epoch_rows(history)
    write_csv(run_dir / "best_epoch_range.csv", best_rows)
    write_markdown_table(run_dir / "best_epoch_range.md", best_rows)
    plot_best_epoch_range(history, run_dir / "best_epoch_range.png")

    # ── sample outputs ────────────────────────────────────────────────────────
    prompt_rows = [
        {"source": "start_context", "prompt": args.start_context[:80]},
        {"source": "train_sample",  "prompt": train_text[:80].replace("\n", " ")},
        {"source": "val_sample",    "prompt": val_text[:80].replace("\n", " ")},
        {"source": "test_sample",   "prompt": test_text[:80].replace("\n", " ")},
    ]
    sample_rows = [
        {
            "source": row["source"],
            "input":  row["prompt"],
            "output": generate_sample_text(
                model=model, tokenizer=tokenizer,
                prompt=row["prompt"], device=device,
                context_length=args.context_length,
                max_new_tokens=args.report_sample_tokens,
            ),
        }
        for row in prompt_rows
    ]
    write_csv(run_dir / "sample_outputs.csv", sample_rows)
    write_markdown_table(run_dir / "sample_outputs.md", sample_rows)
    write_png_table(run_dir / "sample_outputs.png", sample_rows,
                    "Input / Output Samples", wrap_width=42, max_lines=7)

    # ── summary ───────────────────────────────────────────────────────────────
    final = history[-1]
    best  = min(history, key=lambda r: r["val_loss"])
    summary_rows = [
        {"item": "result_dir",        "value": str(run_dir)},
        {"item": "train_path",        "value": str(args.train_path)},
        {"item": "val_path",          "value": str(args.val_path)},
        {"item": "test_path",         "value": str(args.test_path)},
        {"item": "tokenizer_path",    "value": str(args.tokenizer_path)},
        {"item": "tokenizer_mode",    "value": getattr(args, "tokenizer_mode", "unknown")},
        {"item": "device",            "value": str(device)},
        {"item": "num_params",        "value": num_params},
        {"item": "train_chars",       "value": len(train_text)},
        {"item": "val_chars",         "value": len(val_text)},
        {"item": "test_chars",        "value": len(test_text)},
        {"item": "train_tokens",      "value": len(train_ids)},
        {"item": "val_tokens",        "value": len(val_ids)},
        {"item": "test_tokens",       "value": len(test_ids)},
        {"item": "vocab_size",        "value": len(tokenizer.id_to_token)},
        {"item": "context_length",    "value": args.context_length},
        {"item": "emb_dim",           "value": args.emb_dim},
        {"item": "n_heads",           "value": args.n_heads},
        {"item": "n_layers",          "value": args.n_layers},
        {"item": "drop_rate",         "value": args.drop_rate},
        {"item": "batch_size",        "value": args.batch_size},
        {"item": "epochs",            "value": args.epochs},
        {"item": "lr",                "value": args.lr},
        {"item": "seed",              "value": args.seed},
        {"item": "eval_freq",         "value": args.eval_freq},
        {"item": "eval_iter",         "value": args.eval_iter},
        {"item": "final_step",        "value": final["step"]},
        {"item": "final_train_loss",  "value": f"{final['train_loss']:.6f}"},
        {"item": "final_val_loss",    "value": f"{final['val_loss']:.6f}"},
        {"item": "final_test_loss",   "value": f"{final['test_loss']:.6f}"},
        {"item": "final_train_ppl",   "value": f"{final['train_ppl']:.6f}"},
        {"item": "final_val_ppl",     "value": f"{final['val_ppl']:.6f}"},
        {"item": "final_test_ppl",    "value": f"{final['test_ppl']:.6f}"},
        {"item": "best_epoch",        "value": best["epoch"]},
        {"item": "best_step",         "value": best["step"]},
        {"item": "best_val_loss",     "value": f"{best['val_loss']:.6f}"},
        {"item": "best_val_ppl",      "value": f"{best['val_ppl']:.6f}"},
    ]
    write_csv(run_dir / "summary.csv", summary_rows)
    write_markdown_table(run_dir / "summary.md", summary_rows)
    write_png_table(run_dir / "summary.png", summary_rows, "Training Summary",
                    wrap_width=48, max_lines=4)
    (run_dir / "summary.json").write_text(
        json.dumps({r["item"]: r["value"] for r in summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved result artifacts: {run_dir}")
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.smoke:
        args.vocab_size      = min(args.vocab_size, 300)
        args.tokenizer_path  = args.output_dir / f"bpe_vocab_{args.vocab_size}_smoke.json"
        args.max_train_chars = args.max_train_chars or 3_000
        args.max_val_chars   = args.max_val_chars   or 1_000
        args.max_test_chars  = args.max_test_chars  or 1_000
        args.context_length  = min(args.context_length, 32)
        args.emb_dim         = min(args.emb_dim, 32)
        args.n_layers        = 1
        args.epochs          = 1
        args.batch_size      = min(args.batch_size, 2)
        args.eval_freq       = min(args.eval_freq, 20)
        args.eval_iter       = min(args.eval_iter, 1)
        args.ckpt_freq       = 0

    train_text = read_text(args.train_path, args.max_train_chars)
    val_text   = read_text(args.val_path,   args.max_val_chars)
    if args.test_path.exists():
        test_text = read_text(args.test_path, args.max_test_chars)
    else:
        print(f"Test path not found: {args.test_path}. Using validation text as test text.")
        test_text = val_text

    print(f"Train path: {args.train_path}")
    print(f"Val path:   {args.val_path}")
    print(f"Test path:  {args.test_path}")
    tokenizer = build_tokenizer(args, train_text)

    train_ids = tokenizer.encode(train_text)
    val_ids   = tokenizer.encode(val_text)
    test_ids  = tokenizer.encode(test_text)
    if (
        len(train_ids) <= args.context_length
        or len(val_ids) <= args.context_length
        or len(test_ids) <= args.context_length
    ):
        raise ValueError("Not enough tokens for the chosen context length.")

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    worker_init_fn = seed_worker if args.num_workers > 0 else None
    loader_kwargs = dict(context_length=args.context_length, batch_size=args.batch_size,
                         stride=args.stride, num_workers=args.num_workers,
                         worker_init_fn=worker_init_fn)
    train_loader = create_dataloader(train_ids, drop_last=True,  shuffle=True,
                                     generator=train_generator, **loader_kwargs)
    val_loader   = create_dataloader(val_ids,   drop_last=False, shuffle=False, **loader_kwargs)
    test_loader  = create_dataloader(test_ids,  drop_last=False, shuffle=False, **loader_kwargs)

    config = {
        "vocab_size":     len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim":        args.emb_dim,
        "n_heads":        args.n_heads,
        "n_layers":       args.n_layers,
        "drop_rate":      args.drop_rate,
        "qkv_bias":       False,
    }
    model     = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    device    = pick_device(args.device)

    start_epoch  = 0
    global_step  = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from {args.resume}: epoch={start_epoch}, step={global_step}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device:        {device}")
    print(f"Seed:          {args.seed}")
    print(f"Tokenizer vocab: {config['vocab_size']}")
    print(f"Model params:  {num_params:,}")
    print(f"Batches: train={len(train_loader):,}, val={len(val_loader):,}, test={len(test_loader):,}")

    save_run_config(
        args=args,
        train_text=train_text, val_text=val_text, test_text=test_text,
        train_ids=train_ids,   val_ids=val_ids,   test_ids=test_ids,
        train_batches=len(train_loader), val_batches=len(val_loader),
        test_batches=len(test_loader),
        model_config=config, num_params=num_params, device=device,
    )

    metric_history: list[dict] = []

    def record_eval(epoch: int, step: int, train_loss: float, val_loss: float) -> None:
        model.eval()
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=args.eval_iter)
        model.train()
        metric_history.append({
            "epoch":      epoch,
            "step":       step,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "test_loss":  test_loss,
            "train_ppl":  safe_ppl(train_loss),
            "val_ppl":    safe_ppl(val_loss),
            "test_ppl":   safe_ppl(test_loss),
        })
        print(f"테스트 손실 {test_loss:.3f}")

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
        ckpt_freq=args.ckpt_freq if args.ckpt_freq > 0 else None,
        start_epoch=start_epoch,
        global_step=global_step,
        checkpoint_dir=args.output_dir,
        eval_callback=record_eval,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "final_pretrain.pt"
    final_step = global_step + len(train_loader) * args.epochs
    save_checkpoint(model, optimizer, start_epoch + args.epochs, final_step, str(final_path))
    print(f"Saved final checkpoint: {final_path}")

    if not metric_history:
        model.eval()
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=args.eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=args.eval_iter)
        test_loss  = calc_loss_loader(test_loader,  model, device, num_batches=args.eval_iter)
        model.train()
        metric_history.append({
            "epoch":      start_epoch + args.epochs,
            "step":       final_step,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "test_loss":  test_loss,
            "train_ppl":  safe_ppl(train_loss),
            "val_ppl":    safe_ppl(val_loss),
            "test_ppl":   safe_ppl(test_loss),
        })

    write_training_results(
        args=args, model=model, tokenizer=tokenizer, device=device,
        history=metric_history,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        train_text=train_text, val_text=val_text, test_text=test_text,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        num_params=num_params,
    )
    print_summary(args, train_losses, val_losses)


if __name__ == "__main__":
    main()

