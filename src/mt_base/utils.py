"""Misc helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module, only_trainable: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if (not only_trainable) or p.requires_grad
    )


def human_format(n: int) -> str:
    units = ["", "K", "M", "B", "T"]
    i = 0
    x = float(n)
    while abs(x) >= 1000 and i < len(units) - 1:
        x /= 1000.0
        i += 1
    return f"{x:.2f}{units[i]}"


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_dtype(name: str) -> torch.dtype | None:
    name = name.lower()
    if name in {"fp32", "float32", "none"}:
        return None
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(name)
