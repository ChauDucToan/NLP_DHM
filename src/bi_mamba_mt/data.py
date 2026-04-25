"""Dataset, collator, and parallel-corpus utilities."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    Tokenizer,
    VI2ZH_ID,
    ZH2VI_ID,
)


# ---------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------
def basic_clean(text: str) -> str:
    return " ".join(text.split()).strip()


def length_ok(src: str, tgt: str, min_len: int, max_chars: int) -> bool:
    if len(src) < min_len or len(tgt) < min_len:
        return False
    if len(src) > max_chars or len(tgt) > max_chars:
        return False
    ratio = len(src) / max(1, len(tgt))
    return 0.25 <= ratio <= 4.0


# ---------------------------------------------------------------------
# Pair representation
# ---------------------------------------------------------------------
@dataclass
class Pair:
    zh: str
    vi: str


def write_jsonl(pairs: Iterable[Pair], path: str | os.PathLike) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"zh": p.zh, "vi": p.vi}, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike) -> List[Pair]:
    pairs: List[Pair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append(Pair(zh=d["zh"], vi=d["vi"]))
    return pairs


# ---------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------
class TranslationDataset(Dataset):
    """Bidirectional parallel-corpus dataset.

    Each sample is randomly chosen to be (zh→vi) or (vi→zh) when
    ``bidirectional=True``. Direction is encoded by prepending a tag
    token to the source.
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        tokenizer: Tokenizer,
        max_src_len: int,
        max_tgt_len: int,
        bidirectional: bool = True,
        seed: int = 0,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bidirectional = bidirectional
        self.rng = random.Random(seed)
        # Pre-build a deterministic direction list when bidirectional, so
        # both directions are seen equally and reproducibly.
        if bidirectional:
            self._dirs = [
                "zh2vi" if i % 2 == 0 else "vi2zh" for i in range(2 * len(self.pairs))
            ]
        else:
            self._dirs = ["zh2vi"] * len(self.pairs)

    def __len__(self) -> int:
        return len(self._dirs)

    def __getitem__(self, idx: int) -> dict:
        if self.bidirectional:
            pair = self.pairs[idx // 2]
            direction = self._dirs[idx]
        else:
            pair = self.pairs[idx]
            direction = "zh2vi"

        if direction == "zh2vi":
            src_text, tgt_text = pair.zh, pair.vi
        else:
            src_text, tgt_text = pair.vi, pair.zh

        src_ids = self.tokenizer.encode_src(src_text, direction)
        tgt_ids = self.tokenizer.encode_tgt(tgt_text)

        # Truncate
        src_ids = src_ids[: self.max_src_len]
        tgt_ids = tgt_ids[: self.max_tgt_len]

        # Decoder input is shifted right (drop last); target is shifted left (drop bos)
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_ids[:-1], dtype=torch.long),
            "tgt_out": torch.tensor(tgt_ids[1:], dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------
@dataclass
class Collator:
    pad_id: int = PAD_ID

    def __call__(self, batch: List[dict]) -> dict:
        src_lens = [len(b["src"]) for b in batch]
        tgt_lens = [len(b["tgt_in"]) for b in batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)
        B = len(batch)
        src = torch.full((B, max_src), self.pad_id, dtype=torch.long)
        tgt_in = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)
        tgt_out = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ls = len(b["src"])
            lt = len(b["tgt_in"])
            src[i, :ls] = b["src"]
            tgt_in[i, :lt] = b["tgt_in"]
            tgt_out[i, :lt] = b["tgt_out"]
        src_pad_mask = src.eq(self.pad_id)
        return {
            "src": src,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "src_pad_mask": src_pad_mask,
        }


# ---------------------------------------------------------------------
# Plain-text helpers used by the tokenizer trainer
# ---------------------------------------------------------------------
def write_plain_corpus(pairs: Iterable[Pair], out_path: str | os.PathLike) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            if p.zh:
                f.write(p.zh + "\n")
                n += 1
            if p.vi:
                f.write(p.vi + "\n")
                n += 1
    return n
