"""SentencePiece BPE tokenizer wrapper, shared across Chinese + Vietnamese.

Reserved special tokens (ids 0..5):
    0  <pad>
    1  <bos>
    2  <eos>
    3  <unk>
    4  <2vi>    direction tag: translate to Vietnamese
    5  <2zh>    direction tag: translate to Chinese
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
ZH2VI_ID = 4
VI2ZH_ID = 5

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<2vi>", "<2zh>"]


@dataclass
class TokenizerConfig:
    vocab_size: int = 16000
    model_type: str = "bpe"
    character_coverage: float = 0.9995


def train_sentencepiece(
    text_files: Iterable[str | os.PathLike],
    model_prefix: str | os.PathLike,
    cfg: TokenizerConfig,
) -> None:
    """Train a shared SentencePiece BPE model.

    Reserves ids 0..5 for the special tokens above.
    """
    text_files = [str(p) for p in text_files]
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=",".join(text_files),
        model_prefix=str(model_prefix),
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        character_coverage=cfg.character_coverage,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=["<2vi>", "<2zh>"],
        normalization_rule_name="nmt_nfkc",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
    )


class Tokenizer:
    """Thin wrapper around ``sentencepiece.SentencePieceProcessor``."""

    def __init__(self, model_path: str | os.PathLike) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        # Sanity-check special-token alignment with our constants.
        assert self.sp.PieceToId("<pad>") == PAD_ID, "Vocab not aligned: <pad>"
        assert self.sp.PieceToId("<bos>") == BOS_ID, "Vocab not aligned: <bos>"
        assert self.sp.PieceToId("<eos>") == EOS_ID, "Vocab not aligned: <eos>"
        assert self.sp.PieceToId("<unk>") == UNK_ID, "Vocab not aligned: <unk>"
        assert self.sp.PieceToId("<2vi>") == ZH2VI_ID, "Vocab not aligned: <2vi>"
        assert self.sp.PieceToId("<2zh>") == VI2ZH_ID, "Vocab not aligned: <2zh>"

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode(self, text: str) -> List[int]:
        return self.sp.EncodeAsIds(text)

    def encode_src(self, text: str, direction: str) -> List[int]:
        """Encode source with the right direction tag.

        ``direction`` is one of "zh2vi" or "vi2zh".
        Returns ``[<dir>, ...src_ids..., <eos>]``.
        """
        if direction == "zh2vi":
            tag = ZH2VI_ID
        elif direction == "vi2zh":
            tag = VI2ZH_ID
        else:
            raise ValueError(f"Unknown direction: {direction}")
        return [tag] + self.encode(text) + [EOS_ID]

    def encode_tgt(self, text: str) -> List[int]:
        """Encode target as ``[<bos>, ...tgt_ids..., <eos>]``."""
        return [BOS_ID] + self.encode(text) + [EOS_ID]

    def decode(self, ids: List[int]) -> str:
        # Strip special tokens
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID, ZH2VI_ID, VI2ZH_ID)]
        return self.sp.DecodeIds(ids)
