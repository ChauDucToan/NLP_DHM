"""Smoke tests: model forward/backward, autoregressive step consistency."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.model import BiMambaTranslator, ModelConfig
from bi_mamba_mt.modules.mamba_block import MambaBlock


def test_mamba_step_matches_full_scan():
    """Single-step decoding should match a full-sequence forward."""
    torch.manual_seed(0)
    block = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2, use_fast_path=False)
    block.eval()
    B, L = 2, 10
    x = torch.randn(B, L, 32)

    with torch.no_grad():
        y_full = block(x)
        state = block.init_state(B, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            y_t, state = block.step(x[:, t], state)
            ys.append(y_t)
        y_step = torch.stack(ys, dim=1)

    assert y_full.shape == y_step.shape == (B, L, 32)
    # Outputs may differ slightly because the forward path multiplies by SiLU(z) inside
    # the fused branch but applies it inside selective_scan_ref. Both branches in this
    # test go through the same code, so check tight equality.
    assert torch.allclose(y_full, y_step, atol=1e-5, rtol=1e-4), (
        f"max diff {torch.max(torch.abs(y_full - y_step))}"
    )


def test_translator_forward_shapes():
    cfg = ModelConfig(
        vocab_size=200,
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        n_cross_attn_heads=4,
        dropout=0.0,
        max_src_len=32,
        max_tgt_len=32,
    )
    model = BiMambaTranslator(cfg)
    src = torch.randint(0, cfg.vocab_size, (3, 12))
    tgt_in = torch.randint(0, cfg.vocab_size, (3, 9))
    src_pad = src.eq(cfg.pad_id)
    out = model(src, tgt_in, src_pad_mask=src_pad)
    assert out.shape == (3, 9, cfg.vocab_size)


def test_translator_backward_runs():
    cfg = ModelConfig(
        vocab_size=200,
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        n_cross_attn_heads=4,
        dropout=0.0,
        max_src_len=32,
        max_tgt_len=32,
    )
    model = BiMambaTranslator(cfg)
    src = torch.randint(0, cfg.vocab_size, (2, 6))
    tgt_in = torch.randint(0, cfg.vocab_size, (2, 5))
    tgt_out = torch.randint(0, cfg.vocab_size, (2, 5))
    src_pad = src.eq(cfg.pad_id)
    logits = model(src, tgt_in, src_pad_mask=src_pad)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), tgt_out.reshape(-1)
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_translator_greedy_generate_runs():
    cfg = ModelConfig(
        vocab_size=200,
        d_model=64,
        d_state=8,
        d_conv=4,
        expand=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        n_cross_attn_heads=4,
        dropout=0.0,
        max_src_len=32,
        max_tgt_len=32,
    )
    model = BiMambaTranslator(cfg).eval()
    src = torch.randint(6, cfg.vocab_size, (2, 7))
    src_pad = src.eq(cfg.pad_id)
    out = model.generate(src, src_pad_mask=src_pad, max_len=10, beam_size=1)
    assert len(out) == 2
    assert all(isinstance(seq, list) for seq in out)


if __name__ == "__main__":
    test_mamba_step_matches_full_scan()
    test_translator_forward_shapes()
    test_translator_backward_runs()
    test_translator_greedy_generate_runs()
    print("All tests passed.")
