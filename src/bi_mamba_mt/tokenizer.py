"""Backward-compat shim: re-export :mod:`mt_base.tokenizer`."""

from mt_base.tokenizer import *  # noqa: F401,F403
from mt_base.tokenizer import (  # noqa: F401
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    Tokenizer,
    TokenizerConfig,
    train_sentencepiece,
)
