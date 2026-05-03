"""Smoke tests for the Transformer baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformer_mt.model import ModelConfig, TransformerTranslator


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_src_len=32,
        max_tgt_len=32,
    )


def test_transformer_forward_shape() -> None:
    cfg = _tiny_cfg()
    model = TransformerTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (2, 5))
    tgt = torch.randint(0, cfg.vocab_size, (2, 4))
    logits = model(src, tgt)
    assert logits.shape == (2, 4, cfg.vocab_size)


def test_transformer_greedy_decode_returns_token_lists() -> None:
    cfg = _tiny_cfg()
    model = TransformerTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (3, 6))
    out = model.generate(src, max_len=4, beam_size=1)
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(seq, list) and len(seq) <= 4 for seq in out)


def test_transformer_beam_decode_returns_token_lists() -> None:
    cfg = _tiny_cfg()
    model = TransformerTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model.generate(src, max_len=4, beam_size=3, length_penalty=1.0)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(seq, list) and len(seq) <= 4 for seq in out)


def test_transformer_param_count_target() -> None:
    """Default config (d_model=384, 5+5, d_ff=2048) targets ~31M params.

    Keeps Transformer baseline directly comparable to Bi-Mamba 32.4M.
    """
    cfg = ModelConfig()
    model = TransformerTranslator(cfg)
    n = model.num_parameters()
    assert 28_000_000 <= n <= 35_000_000, (
        f"expected ~31M params, got {n:,}"
    )
