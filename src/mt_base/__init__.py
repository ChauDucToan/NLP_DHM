"""Shared utilities for sequence-to-sequence translation models.

Holds tokenizer, data pipeline, beam-search inference, evaluator, and the
generic training loop. Architecture-specific code lives in ``bi_mamba_mt``
and ``transformer_mt`` (Transformer baseline).
"""

from .tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    Tokenizer,
    TokenizerConfig,
    train_sentencepiece,
)

__all__ = [
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "UNK_ID",
    "Tokenizer",
    "TokenizerConfig",
    "train_sentencepiece",
]
