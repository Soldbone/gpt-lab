# -*- coding: utf-8 -*-
"""Reproducibility tests for shared seed handling."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_set_global_seed_repeats_random_numpy_torch_values():
    from src.reproducibility import set_global_seed

    set_global_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(3))

    set_global_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_create_dataloader_shuffle_is_reproducible_with_generator():
    from src.dataset import create_dataloader
    from src.reproducibility import make_torch_generator, seed_worker

    token_ids = list(range(128))
    loader1 = create_dataloader(
        token_ids,
        context_length=8,
        batch_size=4,
        shuffle=True,
        generator=make_torch_generator(42),
        worker_init_fn=seed_worker,
    )
    loader2 = create_dataloader(
        token_ids,
        context_length=8,
        batch_size=4,
        shuffle=True,
        generator=make_torch_generator(42),
        worker_init_fn=seed_worker,
    )

    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    torch.testing.assert_close(batch1[0], batch2[0])
    torch.testing.assert_close(batch1[1], batch2[1])
