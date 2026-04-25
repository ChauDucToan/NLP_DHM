"""Dataset, collator, and parallel-corpus utilities."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

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


# CJK Unified Ideographs (Basic + Extension A). Matches Han characters used in
# both Chinese (Hanzi) and historical Vietnamese (Chữ Nôm), but modern
# Vietnamese is Latin-script so this is a strong discriminator zh vs vi.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# Latin letters (incl. Vietnamese tone-marked variants).
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for c in text if _CJK_RE.match(c))
    return n / len(text)


def _latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for c in text if _LATIN_RE.match(c))
    return n / len(text)


def looks_like_zh(text: str, *, min_cjk: float = 0.30) -> bool:
    """Heuristic: a Chinese sentence should be majority CJK characters."""
    return _cjk_ratio(text) >= min_cjk


def looks_like_vi(text: str, *, max_cjk: float = 0.05, min_latin: float = 0.30) -> bool:
    """Heuristic: a Vietnamese sentence should be Latin and have almost no CJK."""
    return _cjk_ratio(text) <= max_cjk and _latin_ratio(text) >= min_latin


def length_ok(
    src: str,
    tgt: str,
    min_len: int,
    max_chars: int,
    *,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0,
    script_check: bool = False,
) -> bool:
    """Length / ratio / (optional) script-id sanity check.

    When ``script_check`` is True and ``src_lang``/``tgt_lang`` are 'zh' or 'vi',
    we additionally require the source/target to look like the claimed language.
    This filters out the worst noise from WikiMatrix (mis-aligned pairs).
    """
    if len(src) < min_len or len(tgt) < min_len:
        return False
    if len(src) > max_chars or len(tgt) > max_chars:
        return False
    ratio = len(src) / max(1, len(tgt))
    if not (min_ratio <= ratio <= max_ratio):
        return False
    if script_check:
        for text, lang in ((src, src_lang), (tgt, tgt_lang)):
            if lang == "zh" and not looks_like_zh(text):
                return False
            if lang == "vi" and not looks_like_vi(text):
                return False
    return True


def pair_ok(
    zh: str,
    vi: str,
    *,
    min_len: int = 1,
    max_chars: int = 1000,
    min_zh_vi_ratio: float = 0.10,
    max_zh_vi_ratio: float = 1.20,
    script_check: bool = True,
) -> bool:
    """Filter for a (zh, vi) pair.

    The default ratio bounds reflect that Vietnamese sentences are typically
    longer in characters than Chinese (vi has whitespace + diacritics; zh has
    one character per morpheme). Empirically, len(zh)/len(vi) for clean OPUS
    pairs sits around 0.20–0.60 — the bounds [0.10, 1.20] keep almost all real
    pairs while dropping severely mis-aligned WikiMatrix entries.
    """
    return length_ok(
        zh,
        vi,
        min_len=min_len,
        max_chars=max_chars,
        src_lang="zh",
        tgt_lang="vi",
        min_ratio=min_zh_vi_ratio,
        max_ratio=max_zh_vi_ratio,
        script_check=script_check,
    )


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
        bpe_dropout: float = 0.0,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bidirectional = bidirectional
        self.rng = random.Random(seed)
        # BPE-dropout / subword regularization (α). 0.0 = deterministic
        # tokenization (default for valid/test). Use ~0.1 on train data.
        self.bpe_dropout = float(bpe_dropout)
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

    def bucket_lengths(self) -> List[int]:
        """Return an approximate token length per item, used for length-bucketed batching.

        We use a cheap character-level proxy that's monotonic with the actual
        SentencePiece token count: each Chinese character is ~1 token, each
        Vietnamese word (~5 chars on average) is ~1\u20132 tokens. The proxy
        ``max(len_zh, len_vi // 3)`` correlates strongly with the post-tokenization
        max(src_len, tgt_len) which drives padding cost.
        """
        out: List[int] = []
        for i in range(len(self._dirs)):
            if self.bidirectional:
                p = self.pairs[i // 2]
            else:
                p = self.pairs[i]
            zh_len = len(p.zh)
            vi_len = len(p.vi) // 3 + 1
            out.append(max(zh_len, vi_len))
        return out

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

        sampling = self.bpe_dropout > 0.0
        src_ids = self.tokenizer.encode_src(
            src_text, direction, sampling=sampling, alpha=self.bpe_dropout
        )
        tgt_ids = self.tokenizer.encode_tgt(
            tgt_text, sampling=sampling, alpha=self.bpe_dropout
        )

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


# ---------------------------------------------------------------------
# Length-bucketed sampler (sort-pool batching)
# ---------------------------------------------------------------------
class SortPoolBatchSampler(Sampler[List[int]]):
    """Length-bucketed batch sampler.

    Within each "pool" of ``pool_factor * batch_size`` consecutive shuffled
    items, indices are sorted by length and chopped into batches; pools are
    then re-shuffled so the global iteration order is still random but each
    batch contains items of similar length. Cuts padding waste by 30–50%
    versus naive shuffling, which directly increases effective tokens/sec.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        pool_factor: int = 100,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.lengths = np.asarray(list(lengths), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.pool_size = max(self.batch_size, int(pool_factor) * self.batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n = len(self.lengths)
        if self.shuffle:
            order = rng.permutation(n)
        else:
            order = np.arange(n)
        batches: List[List[int]] = []
        for start in range(0, n, self.pool_size):
            pool = order[start : start + self.pool_size]
            sort_idx = np.argsort(self.lengths[pool], kind="stable")
            pool = pool[sort_idx]
            for j in range(0, len(pool), self.batch_size):
                b = pool[j : j + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        for b in batches:
            yield b

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
