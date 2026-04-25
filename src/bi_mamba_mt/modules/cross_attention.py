"""Multi-head cross-attention used by the decoder."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, Lq, D)
        kv: torch.Tensor,  # (B, Lk, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Lk) True = pad
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = kv.shape[1]
        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, Lk), True where padded -> -inf
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)

        # Use SDPA when available
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=~attn_mask if attn_mask is not None else None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o_proj(out)
