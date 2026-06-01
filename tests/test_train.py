# -*- coding: utf-8 -*-
"""
사전 훈련 유틸 단위 테스트 (calc_loss_batch, calc_loss_loader, save/load_checkpoint, generate).
실행: `pytest tests/test_train.py -v`
"""

import sys
import tempfile
from pathlib import Path

import torch
import pytest
import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GPT_CONFIG_SMALL = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


# =============================================================================
# calc_loss_batch
# =============================================================================


class TestCalcLossBatch:
    """calc_loss_batch 구현 후 실행."""

    def test_calc_loss_batch_returns_scalar(self):
        """calc_loss_batch()가 한 배치의 cross entropy loss를 scalar Tensor로 반환하는지 확인한다."""
        from model import GPTModel
        from train import calc_loss_batch

        try:
            model = GPTModel(GPT_CONFIG_SMALL)
            device = torch.device("cpu")
            batch_size, seq_len = 2, 8
            inp = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (batch_size, seq_len))
            tgt = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (batch_size, seq_len))
            loss = calc_loss_batch(inp, tgt, model, device)
        except NotImplementedError:
            pytest.fail("calc_loss_batch 미구현")
        assert loss.dim() == 0
        assert loss.item() >= 0


# =============================================================================
# calc_loss_loader
# =============================================================================


class TestCalcLossLoader:
    """calc_loss_loader 구현 후 실행."""

    def test_calc_loss_loader_returns_float(self):
        """calc_loss_loader()가 여러 배치의 평균 손실을 float 또는 scalar Tensor로 반환하는지 확인한다."""
        from model import GPTModel
        from train import calc_loss_loader
        from dataset import create_dataloader

        try:
            loader = create_dataloader(
                list(range(200)),
                context_length=16,
                batch_size=4,
                shuffle=False,
            )
            model = GPTModel(GPT_CONFIG_SMALL)
            device = torch.device("cpu")
            avg_loss = calc_loss_loader(loader, model, device, num_batches=2)
        except NotImplementedError:
            pytest.fail("calc_loss_loader 또는 create_dataloader 미구현")
        assert isinstance(avg_loss, (float, torch.Tensor))
        if isinstance(avg_loss, torch.Tensor):
            avg_loss = avg_loss.item()
        assert avg_loss >= 0


# =============================================================================
# save_checkpoint / load_checkpoint
# =============================================================================


class TestCheckpoint:
    """save_checkpoint, load_checkpoint 구현 후 실행."""

    def test_save_load_checkpoint_restores_epoch_and_step(self):
        """체크포인트 저장 후 다른 모델/옵티마이저에 epoch와 step까지 복원되는지 확인한다."""
        from model import GPTModel
        from train import save_checkpoint, load_checkpoint

        try:
            model = GPTModel(GPT_CONFIG_SMALL)
            params = list(model.parameters())
            if not params:
                pytest.fail("GPTModel.__init__ 미구현")
            optimizer = torch.optim.AdamW(params, lr=1e-4)
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                path = f.name
            save_checkpoint(model, optimizer, epoch=1, global_step=10, path=path)
            model2 = GPTModel(GPT_CONFIG_SMALL)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
            epoch, step = load_checkpoint(model2, opt2, path, torch.device("cpu"))
            Path(path).unlink(missing_ok=True)
        except NotImplementedError:
            pytest.fail("save_checkpoint/load_checkpoint 미구현")
        assert epoch == 1
        assert step == 10


# =============================================================================
# generate (temperature, top_k)
# =============================================================================


class TestGenerate:
    """generate 구현 후 실행."""

    def test_generate_shape(self):
        """generate()가 temperature/top_k 옵션을 사용해 max_new_tokens만큼 토큰을 추가하는지 확인한다."""
        from model import GPTModel
        from train import generate

        try:
            model = GPTModel(GPT_CONFIG_SMALL)
            idx = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (1, 4))
            out = generate(
                model, idx, max_new_tokens=5, context_size=GPT_CONFIG_SMALL["context_length"],
                temperature=1.0, top_k=10,
            )
        except NotImplementedError:
            pytest.fail("generate 미구현")
        assert out.shape == (1, 4 + 5)


# =============================================================================
# plot_losses (제공 함수)
# =============================================================================


class TestPlotLosses:
    """plot_losses는 제공됨. 호출만 확인."""

    def test_plot_losses_callable(self):
        """plot_losses()가 train/val loss 리스트를 받아 예외 없이 그래프를 그리는지 확인한다."""
        import matplotlib.pyplot as plt
        from train import plot_losses

        plt.close("all")
        plot_losses([0.5, 0.4, 0.3], [0.6, 0.5, 0.4])

        ax = plt.gca()
        assert len(ax.lines) == 2
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Loss"
        assert ax.get_title() == "Training / Validation Loss"
        assert [line.get_label() for line in ax.lines] == ["Train", "Val"]


# =============================================================================
# train_model
# =============================================================================


GPT_CONFIG_TINY = {
    "vocab_size": 64,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


class DummyTokenizer:
    """Minimal tokenizer for train_model sample generation."""

    def encode(self, text, allowed_special=None):
        return [ord(ch) % GPT_CONFIG_TINY["vocab_size"] for ch in text]

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class TestTrainModel:
    """train_model training-loop tests."""

    def test_train_model_returns_train_losses(self, monkeypatch):
        from dataset import create_dataloader
        from model import GPTModel
        import train

        monkeypatch.setattr(train, "generate_and_print_sample", lambda *args, **kwargs: None)

        train_tokens = [i % GPT_CONFIG_TINY["vocab_size"] for i in range(80)]
        val_tokens = [i % GPT_CONFIG_TINY["vocab_size"] for i in range(80, 120)]

        train_loader = create_dataloader(
            train_tokens,
            context_length=GPT_CONFIG_TINY["context_length"],
            batch_size=2,
            shuffle=False,
        )
        val_loader = create_dataloader(
            val_tokens,
            context_length=GPT_CONFIG_TINY["context_length"],
            batch_size=2,
            shuffle=False,
        )
        model = GPTModel(GPT_CONFIG_TINY)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        result = train.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            num_epochs=1,
            eval_freq=1,
            eval_iter=1,
            start_context="hi",
            tokenizer=DummyTokenizer(),
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        train_losses, val_losses = result
        assert isinstance(train_losses, list)
        assert isinstance(val_losses, list)
        assert len(train_losses) > 0
        assert len(val_losses) > 0
        assert len(train_losses) == len(val_losses)
        assert all(isinstance(loss, float) for loss in train_losses)
        assert all(isinstance(loss, float) for loss in val_losses)
        assert all(loss >= 0 for loss in train_losses)
        assert all(loss >= 0 for loss in val_losses)

        train.plot_losses(train_losses, val_losses)
