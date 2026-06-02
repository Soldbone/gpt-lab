# -*- coding: utf-8 -*-
"""Step 2: train GPT from saved BPE/token-id artifacts."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from report_pipeline_common import (  # noqa: E402
    ROOT,
    artifact_path,
    build_epoch_summary,
    load_json,
    load_token_ids,
    read_csv_rows,
    save_json,
    set_seed,
    write_csv,
    write_epoch_markdown,
)
from dataset import create_dataloader  # noqa: E402
from model import GPTModel  # noqa: E402
import train as train_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="outputs/report_pipeline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--eval-iter", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--train-token-limit", type=int, default=None)
    parser.add_argument("--val-token-limit", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_run_name(args: argparse.Namespace) -> str:
    lr_text = f"{args.lr:.0e}".replace("-", "m")
    scheduler = args.scheduler
    dropout = str(args.drop_rate).replace(".", "p")
    return f"vocab_from_bpe_ctx{args.context_length}_l{args.n_layers}_lr{lr_text}_do{dropout}_{scheduler}"


def save_checkpoint(
    path: Path,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    config: dict,
    args: argparse.Namespace,
    vocab_path: Path,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "training_args": vars(args),
        "tokenizer_path": str(vocab_path),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resume(
    checkpoint_path: str | None,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int, float, int]:
    if checkpoint_path is None:
        return 0, 0, float("inf"), 0
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
        int(checkpoint.get("best_epoch", 0)),
    )


def plot_metrics(run_dir: Path, metrics: list[dict]) -> None:
    if not metrics:
        return
    epoch_rows = [row for row in metrics if row["phase"] == "epoch_end"]
    eval_rows = [row for row in metrics if row["phase"] == "eval_step"]

    plt.figure(figsize=(11, 6))
    if eval_rows:
        plt.plot([float(row["epoch"]) for row in eval_rows], [float(row["train_loss"]) for row in eval_rows], alpha=0.35, label="train eval-step")
        plt.plot([float(row["epoch"]) for row in eval_rows], [float(row["val_loss"]) for row in eval_rows], alpha=0.35, label="val eval-step")
    if epoch_rows:
        plt.plot([float(row["epoch"]) for row in epoch_rows], [float(row["train_loss"]) for row in epoch_rows], marker="o", label="train epoch")
        plt.plot([float(row["epoch"]) for row in epoch_rows], [float(row["val_loss"]) for row in epoch_rows], marker="o", label="val epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(run_dir / "loss_curves.png", dpi=160, bbox_inches="tight")
    plt.close()

    if epoch_rows:
        plt.figure(figsize=(11, 4))
        plt.plot([float(row["epoch"]) for row in epoch_rows], [float(row["lr"]) for row in epoch_rows], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.savefig(run_dir / "lr_by_epoch.png", dpi=160, bbox_inches="tight")
        plt.close()


def save_run_artifacts(run_dir: Path, metrics: list[dict], summary: dict | None = None) -> None:
    write_csv(run_dir / "metrics.csv", metrics)
    epoch_summary = build_epoch_summary(metrics)
    write_csv(run_dir / "epoch_summary.csv", epoch_summary)
    write_epoch_markdown(run_dir / "epoch_summary.md", epoch_summary)
    plot_metrics(run_dir, metrics)
    if summary is not None:
        save_json(run_dir / "summary.json", summary)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifact_dir = (ROOT / args.artifact_dir).resolve() if not Path(args.artifact_dir).is_absolute() else Path(args.artifact_dir)
    manifest = load_json(artifact_dir / "bpe_manifest.json")
    vocab_path = artifact_path(artifact_dir, manifest["vocab_path"])
    train_ids = load_token_ids(artifact_path(artifact_dir, manifest["train_ids_path"]))
    val_ids = load_token_ids(artifact_path(artifact_dir, manifest["val_ids_path"]))
    if args.train_token_limit is not None and args.train_token_limit > 0:
        train_ids = train_ids[: args.train_token_limit]
    if args.val_token_limit is not None and args.val_token_limit > 0:
        val_ids = val_ids[: args.val_token_limit]

    run_name = args.run_name or default_run_name(args)
    run_dir = artifact_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_vocab_size = int(manifest["actual_vocab_size"])
    config = {
        "vocab_size": actual_vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }
    save_json(run_dir / "config.json", {"model": config, "training": vars(args), "device": str(device)})

    stride = args.stride if args.stride is not None else args.context_length
    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=stride,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=stride,
        shuffle=False,
        drop_last=False,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Not enough tokens for the selected context_length/batch_size.")

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.num_epochs)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.min_lr,
        )

    start_epoch, global_step, best_val_loss, best_epoch = load_resume(
        args.resume_checkpoint,
        model,
        optimizer,
        scheduler,
        device,
    )

    print(f"device={device}", flush=True)
    print(f"artifact_dir={artifact_dir}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)
    print(f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    metrics: list[dict] = read_csv_rows(run_dir / "metrics.csv") if args.resume_checkpoint else []
    start_time = time.time()
    bad_epochs = 0
    best_checkpoint = run_dir / "best_checkpoint.pt"
    last_checkpoint = run_dir / "last_checkpoint.pt"

    for epoch_idx in range(start_epoch, args.num_epochs):
        model.train()
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = train_utils.calc_loss_batch(input_batch, target_batch, model, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch_idx + 1} step={global_step + 1}: {loss.item()}")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if args.eval_freq > 0 and global_step % args.eval_freq == 0:
                train_loss, val_loss = train_utils.evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.eval_iter,
                )
                metrics.append(
                    {
                        "phase": "eval_step",
                        "epoch": epoch_idx + (batch_idx / len(train_loader)),
                        "global_step": global_step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_sec": time.time() - start_time,
                    }
                )
                print(
                    f"eval epoch={epoch_idx + (batch_idx / len(train_loader)):.3f} "
                    f"step={global_step:06d} train={train_loss:.4f} val={val_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )

        train_loss, val_loss = train_utils.evaluate_model(
            model,
            train_loader,
            val_loader,
            device,
            args.eval_iter,
        )
        epoch = epoch_idx + 1
        row = {
            "phase": "epoch_end",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - start_time,
        }
        metrics.append(row)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                args,
                vocab_path,
                best_val_loss,
                best_epoch,
            )
        else:
            bad_epochs += 1

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            config,
            args,
            vocab_path,
            best_val_loss,
            best_epoch,
        )
        summary = {
            "run_name": run_name,
            "artifact_dir": str(artifact_dir),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "global_step": global_step,
            "completed_epochs": epoch,
            "early_stopped": False,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
        }
        save_run_artifacts(run_dir, metrics, summary)
        print(
            f"epoch={epoch:03d}/{args.num_epochs} step={global_step:06d} "
            f"train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}",
            flush=True,
        )

        if args.patience > 0 and bad_epochs >= args.patience:
            summary["early_stopped"] = True
            save_run_artifacts(run_dir, metrics, summary)
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break

    if metrics:
        final = metrics[-1]
        if math.isnan(float(final["val_loss"])):
            raise RuntimeError("final validation loss is NaN")
    print(f"saved={run_dir}", flush=True)


if __name__ == "__main__":
    main()
