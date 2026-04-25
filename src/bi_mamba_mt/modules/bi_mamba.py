"""Bidirectional Mamba block used by the encoder.

A standard "Bi-Mamba" layer runs the SSM both forward and reversed over
the input and combines the two outputs. We follow the simplest variant —
sum of forward + flipped-backward outputs — which roughly matches
bidirectional RNN-style encoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_block import MambaBlock


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba: forward + reversed mamba, summed.

    Both directions share **separate** weights; this is the standard
    bi-directional setup. The reverse path operates on the time-reversed
    input and its output is flipped back before combining.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x`` of shape (B, L, D). ``key_padding_mask`` (B, L) is True for pad tokens."""
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
            x = x * mask
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(x.flip(dims=[1])).flip(dims=[1])
        if key_padding_mask is not None:
            y_fwd = y_fwd * mask
            y_bwd = y_bwd * mask
        out = self.combine(torch.cat([y_fwd, y_bwd], dim=-1))
        return out
