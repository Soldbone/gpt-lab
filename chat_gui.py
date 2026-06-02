# -*- coding: utf-8 -*-
"""Tkinter chat GUI for the trained mini GPT model."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import torch

from pretrain import DEFAULT_HYPERPARAMS, ROOT, pick_device
from src.bpe import BPETokenizer
from src.model import GPTModel
from src.train import generate


def find_checkpoint(path: Path | None) -> Path:
    if path is not None:
        return path

    final_path = ROOT / "checkpoints" / "final_pretrain.pt"
    if final_path.exists():
        return final_path

    candidates = sorted((ROOT / "checkpoints").glob("ckpt_step_*.pt"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found under checkpoints/.")
    return candidates[-1]


class MiniGPTChatApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.model: GPTModel | None = None
        self.tokenizer: BPETokenizer | None = None
        self.device = pick_device(args.device)

        self.root.title("Mini GPT Chat")
        self.root.geometry("820x640")
        self.root.minsize(640, 480)

        self._build_widgets()
        self._set_busy(True, "모델 로딩 중...")
        self.root.after(100, self._load_model)

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.chat = ScrolledText(main, wrap="word", height=24)
        self.chat.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.chat.configure(state="disabled")

        ttk.Label(main, text="토큰").grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.max_tokens = tk.IntVar(value=self.args.max_new_tokens)
        ttk.Spinbox(main, from_=10, to=300, textvariable=self.max_tokens, width=8).grid(
            row=2, column=0, sticky="w"
        )

        ttk.Label(main, text="온도").grid(row=1, column=1, sticky="w", pady=(10, 2))
        self.temperature = tk.DoubleVar(value=self.args.temperature)
        ttk.Spinbox(main, from_=0.1, to=1.5, increment=0.1, textvariable=self.temperature, width=8).grid(
            row=2, column=1, sticky="w"
        )

        ttk.Label(main, text="Top-k").grid(row=1, column=2, sticky="w", pady=(10, 2))
        self.top_k = tk.IntVar(value=self.args.top_k)
        ttk.Spinbox(main, from_=1, to=200, textvariable=self.top_k, width=8).grid(
            row=2, column=2, sticky="w"
        )

        self.status = ttk.Label(main, text="")
        self.status.grid(row=2, column=3, sticky="e")

        self.prompt = tk.Text(main, height=4, wrap="word")
        self.prompt.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 6))
        self.prompt.bind("<Command-Return>", lambda _event: self.send())
        self.prompt.bind("<Control-Return>", lambda _event: self.send())

        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, columnspan=4, sticky="e")

        self.clear_button = ttk.Button(buttons, text="비우기", command=self.clear)
        self.clear_button.grid(row=0, column=0, padx=(0, 6))
        self.send_button = ttk.Button(buttons, text="보내기", command=self.send)
        self.send_button.grid(row=0, column=1)

    def _load_model(self) -> None:
        try:
            self._set_busy(True, "토크나이저 로딩 중...")
            tokenizer = BPETokenizer()
            tokenizer.load(self.args.tokenizer_path)

            self._set_busy(True, "모델 생성 중...")
            config = {
                "vocab_size": len(tokenizer.id_to_token),
                "context_length": self.args.context_length,
                "emb_dim": self.args.emb_dim,
                "n_heads": self.args.n_heads,
                "n_layers": self.args.n_layers,
                "drop_rate": self.args.drop_rate,
                "qkv_bias": False,
            }
            model = GPTModel(config).to(self.device)

            self._set_busy(True, "체크포인트 로딩 중...")
            checkpoint_path = find_checkpoint(self.args.checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            message = f"로드 완료: {checkpoint_path.name} / step {checkpoint.get('global_step')}"
            self._set_busy(False, message)
            self._append_chat("시스템", "모델 준비 완료. 이제 메시지를 입력해도 됩니다.")
        except Exception as exc:
            message = f"로드 실패: {exc}"
            self._set_busy(False, message)
            self.send_button.configure(state="disabled")
            self._append_chat("오류", message)

    def _set_busy(self, busy: bool, message: str) -> None:
        print(message, flush=True)
        self.status.configure(text=message)
        state = "disabled" if busy else "normal"
        self.send_button.configure(state=state)
        self.clear_button.configure(state=state)

    def _append_chat(self, speaker: str, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}: {text.strip()}\n\n")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def clear(self) -> None:
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")

    def send(self) -> None:
        text = self.prompt.get("1.0", "end").strip()
        if not text or self.model is None or self.tokenizer is None:
            return

        self.prompt.delete("1.0", "end")
        self._append_chat("나", text)
        self._set_busy(True, "생성 중...")
        max_new_tokens = self.max_tokens.get()
        temperature = self.temperature.get()
        top_k = self.top_k.get()
        threading.Thread(
            target=self._generate_reply,
            args=(text, max_new_tokens, temperature, top_k),
            daemon=True,
        ).start()

    def _generate_reply(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> None:
        assert self.model is not None
        assert self.tokenizer is not None

        try:
            prompt_ids = self.tokenizer.encode(text)
            idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            token_ids = generate(
                model=self.model,
                idx=idx,
                max_new_tokens=max_new_tokens,
                context_size=self.args.context_length,
                temperature=temperature,
                top_k=top_k,
            )
            decoded = self.tokenizer.decode(token_ids.squeeze(0).tolist(), errors="replace")
            reply = decoded[len(text):].strip() or decoded.strip()
            self.root.after(0, lambda: self._append_chat("MiniGPT", reply))
            self.root.after(0, lambda: self._set_busy(False, "준비됨"))
        except Exception as exc:
            message = str(exc)
            self.root.after(0, lambda: self._append_chat("오류", message))
            self.root.after(0, lambda: self._set_busy(False, "준비됨"))


def parse_args() -> argparse.Namespace:
    defaults = DEFAULT_HYPERPARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "final_pretrain.pt")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "checkpoints" / f"bpe_vocab_{defaults['vocab_size']}.json")
    parser.add_argument("--context-length", type=int, default=defaults["context_length"])
    parser.add_argument("--emb-dim", type=int, default=defaults["emb_dim"])
    parser.add_argument("--n-heads", type=int, default=defaults["n_heads"])
    parser.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    parser.add_argument("--drop-rate", type=float, default=defaults["drop_rate"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def run_check(args: argparse.Namespace) -> None:
    started = time.time()
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    print(f"Tokenizer loaded: {len(tokenizer.id_to_token)} tokens")

    config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
    }
    device = pick_device(args.device)
    model = GPTModel(config).to(device)
    checkpoint_path = find_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Checkpoint loaded: {checkpoint_path}")

    idx = torch.tensor(tokenizer.encode("영화"), dtype=torch.long, device=device).unsqueeze(0)
    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=10,
        context_size=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(token_ids.squeeze(0).tolist(), errors="replace"))
    print(f"Check finished in {time.time() - started:.2f}s")


def main() -> None:
    args = parse_args()
    if args.check:
        run_check(args)
        return

    root = tk.Tk()
    MiniGPTChatApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
