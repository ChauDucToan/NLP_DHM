"""Inference helpers: translate one or many sentences."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from .model import BiMambaTranslator
from .tokenizer import PAD_ID, Tokenizer


def _pad_batch(
    batches: Sequence[List[int]], pad_id: int = PAD_ID
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(b) for b in batches)
    src = torch.full((len(batches), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batches):
        src[i, : len(b)] = torch.tensor(b, dtype=torch.long)
    return src, src.eq(pad_id)


@torch.no_grad()
def translate_batch(
    model: BiMambaTranslator,
    tokenizer: Tokenizer,
    sentences: Sequence[str],
    direction: str,
    max_len: int = 256,
    beam_size: int = 1,
    length_penalty: float = 1.0,
    device: torch.device | None = None,
) -> List[str]:
    """Translate a batch of sentences in a single direction.

    ``direction`` is one of "zh2vi" or "vi2zh".
    """
    device = device or next(model.parameters()).device
    model.eval()
    encoded = [tokenizer.encode_src(s, direction)[: model.cfg.max_src_len] for s in sentences]
    src, pad_mask = _pad_batch(encoded)
    src = src.to(device)
    pad_mask = pad_mask.to(device)
    out_ids = model.generate(
        src,
        src_pad_mask=pad_mask,
        max_len=max_len,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )
    return [tokenizer.decode(ids) for ids in out_ids]


@torch.no_grad()
def translate(
    model: BiMambaTranslator,
    tokenizer: Tokenizer,
    sentence: str,
    direction: str,
    **kwargs,
) -> str:
    return translate_batch(model, tokenizer, [sentence], direction, **kwargs)[0]
