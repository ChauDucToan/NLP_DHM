"""Helpers for reproducible evaluation subsets and runtime metadata."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any, Sequence, TypeVar

import torch

T = TypeVar("T")


def select_eval_subset(
    pairs: Sequence[T],
    *,
    num_samples: int | None,
    sample_seed: int | None,
) -> tuple[list[T], dict[str, Any]]:
    """Select a reproducible evaluation subset.

    If ``sample_seed`` is ``None``, keep backward-compatible head slicing.
    Otherwise sample ``num_samples`` indices uniformly without replacement and
    return them in original dataset order.
    """
    total = len(pairs)
    if num_samples is None:
        target = total
    else:
        target = max(0, min(int(num_samples), total))

    if target >= total:
        return list(pairs), {
            "subset_kind": "full",
            "sample_seed": sample_seed,
            "num_total_pairs": total,
            "num_selected_pairs": total,
            "selected_index_preview": list(range(min(total, 10))),
        }

    if sample_seed is None:
        idx = list(range(target))
        subset_kind = "head"
    else:
        rng = random.Random(int(sample_seed))
        idx = sorted(rng.sample(range(total), target))
        subset_kind = "random_seeded"

    subset = [pairs[i] for i in idx]
    return subset, {
        "subset_kind": subset_kind,
        "sample_seed": sample_seed,
        "num_total_pairs": total,
        "num_selected_pairs": len(subset),
        "selected_index_preview": idx[:10],
    }


def infer_eval_label(*, num_total_pairs: int, num_selected_pairs: int, beam_size: int) -> str:
    """Return a human-readable evaluation label for reporting."""
    if num_selected_pairs < num_total_pairs or int(beam_size) <= 1:
        return "DRAFT METRICS"
    return "FINAL METRICS"


def device_info(device: torch.device) -> dict[str, Any]:
    """Collect runtime device metadata."""
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_type": device.type,
        "device": torch.cuda.get_device_name(device) if is_cuda else str(device),
        "cuda_available": bool(is_cuda),
        "cuda_version": torch.version.cuda if is_cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if is_cuda else None,
        "gpu_capability": (
            ".".join(map(str, torch.cuda.get_device_capability(device)))
            if is_cuda
            else None
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk, creating parents when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
