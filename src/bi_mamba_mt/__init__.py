"""Bi-Mamba Chinese↔Vietnamese translator."""

from .model import BiMambaTranslator, ModelConfig
from .tokenizer import Tokenizer

__all__ = ["BiMambaTranslator", "ModelConfig", "Tokenizer"]

__version__ = "0.1.0"
