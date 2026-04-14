"""Deterministic seeding of random, numpy, and torch (CPU + CUDA)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed every RNG we touch.

    Returns the seed so callers can log it. torch is imported lazily so this
    module is usable from non-torch contexts (e.g. pure-pandas preprocessing).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
