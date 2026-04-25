"""Decoder block: causal Mamba + cross-attention + FFN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cross_attention import CrossAttention
from .mamba_block import MambaBlock, MambaState


@dataclass
class DecoderState:
    mamba_state: MambaState


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block:

    x' = x + Mamba(LN(x))
    x'' = x' + CrossAttn(LN(x'), enc, enc_pad)
    out = x'' + FFN(LN(x''))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_cross_attn_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_cross_attn_heads, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.mamba(self.norm_self(x)))
        x = x + self.dropout(
            self.cross_attn(
                self.norm_cross(x),
                encoder,
                key_padding_mask=encoder_padding_mask,
            )
        )
        x = x + self.ffn(self.norm_ffn(x))
        return x

    def init_state(self, batch_size: int, device, dtype) -> DecoderState:
        return DecoderState(mamba_state=self.mamba.init_state(batch_size, device, dtype))

    def step(
        self,
        x_t: torch.Tensor,  # (B, D)
        state: DecoderState,
        encoder: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DecoderState]:
        # Self (causal Mamba step)
        h = self.norm_self(x_t)
        m_out, new_mamba = self.mamba.step(h, state.mamba_state)
        x_t = x_t + m_out
        # Cross-attn (single query)
        c_out = self.cross_attn(
            self.norm_cross(x_t).unsqueeze(1),
            encoder,
            key_padding_mask=encoder_padding_mask,
        ).squeeze(1)
        x_t = x_t + c_out
        # FFN
        x_t = x_t + self.ffn(self.norm_ffn(x_t))
        return x_t, DecoderState(mamba_state=new_mamba)
