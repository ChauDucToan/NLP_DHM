"""Smoke tests for the Hybrid Mamba-Attention model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.modules.mamba_block import MambaBlock
from hybrid_mt.model import HybridMambaAttentionTranslator, ModelConfig


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_decoder_heads=4,
        encoder_d_ff=64,
        decoder_d_ff=64,
        dropout=0.0,
        max_src_len=32,
        max_tgt_len=32,
    )


def test_hybrid_forward_shape() -> None:
    cfg = _tiny_cfg()
    model = HybridMambaAttentionTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (2, 5))
    tgt = torch.randint(0, cfg.vocab_size, (2, 4))
    logits = model(src, tgt)
    assert logits.shape == (2, 4, cfg.vocab_size)


def test_hybrid_encoder_shape() -> None:
    cfg = _tiny_cfg()
    model = HybridMambaAttentionTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (2, 7))
    enc = model.encode(src)
    assert enc.shape == (2, 7, cfg.d_model)


def test_hybrid_greedy_decode_returns_token_lists() -> None:
    cfg = _tiny_cfg()
    model = HybridMambaAttentionTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (3, 6))
    out = model.generate(src, max_len=4, beam_size=1)
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(seq, list) and len(seq) <= 4 for seq in out)


def test_hybrid_beam_decode_returns_token_lists() -> None:
    cfg = _tiny_cfg()
    model = HybridMambaAttentionTranslator(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model.generate(src, max_len=4, beam_size=3, length_penalty=1.0)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(seq, list) and len(seq) <= 4 for seq in out)


def test_hybrid_param_count_target() -> None:
    """Default config (d_model=384, 5+5, encoder_d_ff=960, decoder_d_ff=1536)
    targets ~32-36M params — comparable to Bi-Mamba 32.4M and Transformer 30.8M.
    """
    cfg = ModelConfig()
    model = HybridMambaAttentionTranslator(cfg)
    n = model.num_parameters()
    assert 30_000_000 <= n <= 38_000_000, f"expected ~32-36M params, got {n:,}"


def test_hybrid_init_does_not_touch_mamba_internals() -> None:
    """Critical: the global ``_init_non_mamba_weights`` must skip MambaBlock
    subtrees so the SSM-specific init of ``A_log``, ``D`` and the
    softplus-inverse ``dt_proj.bias`` is preserved.
    """
    cfg = _tiny_cfg()
    model = HybridMambaAttentionTranslator(cfg)
    mb: MambaBlock = model.encoder_layers[0].bi_mamba.fwd
    # 1. A_log should equal log(arange(1, d_state+1)) row-wise (Mamba init).
    expected_A = -torch.exp(
        torch.log(torch.arange(1, cfg.d_state + 1, dtype=torch.float))
    )
    A = -torch.exp(mb.A_log.float())
    for i in range(mb.A_log.size(0)):
        assert torch.allclose(A[i], expected_A, atol=1e-3), (
            f"row {i} of A diverged from Mamba init"
        )
    # 2. D should be all ones.
    assert torch.allclose(mb.D, torch.ones_like(mb.D))
    # 3. dt_proj.bias is marked _no_reinit (must survive any future init pass).
    assert getattr(mb.dt_proj.bias, "_no_reinit", False) is True
    # 4. dt_proj.bias should be a softplus-inverse of values in (dt_min, dt_max),
    #    so always negative (since dt_max < 1 → softplus(b) < 1 → b < 0.54).
    assert mb.dt_proj.bias.max().item() < 0.0
    # 5. _no_weight_decay flags preserved on A_log and D.
    assert getattr(mb.A_log, "_no_weight_decay", False) is True
    assert getattr(mb.D, "_no_weight_decay", False) is True


if __name__ == "__main__":
    test_hybrid_forward_shape()
    test_hybrid_encoder_shape()
    test_hybrid_greedy_decode_returns_token_lists()
    test_hybrid_beam_decode_returns_token_lists()
    test_hybrid_param_count_target()
    test_hybrid_init_does_not_touch_mamba_internals()
    print("All hybrid tests passed.")
