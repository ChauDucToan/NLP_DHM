"""Vanilla Transformer encoder-decoder baseline for zh↔vi translation.

Provides :class:`TransformerTranslator` with the same public API as
:class:`bi_mamba_mt.model.BiMambaTranslator`, so all shared infrastructure
in :mod:`mt_base` (training loop, beam-search, evaluator) works unchanged.
"""

from .model import ModelConfig, TransformerTranslator

__all__ = ["TransformerTranslator", "ModelConfig"]

__version__ = "0.1.0"
