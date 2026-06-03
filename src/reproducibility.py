# -*- coding: utf-8 -*-
"""Reproducibility helpers shared by training and experiment scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA if available."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Create a CPU generator for deterministic DataLoader shuffling."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
