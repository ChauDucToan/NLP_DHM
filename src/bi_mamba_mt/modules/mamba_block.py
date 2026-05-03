"""Mamba (selective SSM) block.

A clean PyTorch implementation of the Mamba block from Gu & Dao (2023),
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces".

Two execution paths:

1. **Fast path (GPU)**: when ``mamba_ssm`` and ``causal_conv1d`` are
   installed, we delegate to ``selective_scan_fn`` and ``causal_conv1d_fn``
   for fused CUDA kernels.
2. **Reference path (anywhere)**: a pure-PyTorch sequential scan that runs
   on CPU or GPU. It is functionally equivalent and used for tests, CPU
   inference, and as a fallback when the CUDA kernels are unavailable.

The block also supports single-step inference (``step``) used by the
autoregressive decoder during generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:  # pragma: no cover - optional fast kernels
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _HAS_SELECTIVE_SCAN = True
except Exception:  # pragma: no cover
    selective_scan_fn = None
    _HAS_SELECTIVE_SCAN = False

try:  # pragma: no cover - optional fast kernels
    from causal_conv1d import causal_conv1d_fn

    _HAS_CAUSAL_CONV1D = True
except Exception:  # pragma: no cover
    causal_conv1d_fn = None
    _HAS_CAUSAL_CONV1D = False


@dataclass
class MambaState:
    """Per-layer state for autoregressive single-step decoding."""

    conv_state: torch.Tensor  # [B, d_inner, d_conv]
    ssm_state: torch.Tensor  # [B, d_inner, d_state]


class MambaBlock(nn.Module):
    """A single causal Mamba block.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int, default 16
        SSM hidden state size N.
    d_conv : int, default 4
        Width of the local causal conv applied before the SSM.
    expand : int, default 2
        Inner expansion factor; d_inner = expand * d_model.
    dt_rank : int or "auto", default "auto"
        Rank of the input-dependent delta projection. "auto" -> ceil(d_model / 16).
    dt_min, dt_max : float
        Initialisation range for delta.
    dt_init : str, "random" or "constant"
    dt_scale : float
        Scaling for delta init.
    bias : bool
        Whether to use bias in the inner Linear projections.
    conv_bias : bool
        Whether to use bias in the conv1d.
    use_fast_path : bool
        If True and CUDA kernels are available, use them.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        use_fast_path: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = int(dt_rank)
        self.use_fast_path = bool(use_fast_path)

        # Input projection: x -> [x, z]
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Local causal conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # Activation
        self.act = nn.SiLU()

        # x_proj: x -> [delta_proj, B, C]
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt_proj: delta_proj -> delta (per-channel)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj init
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(dt_init)

        # Initialise dt bias so that softplus(bias) ~ U[dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # mark to skip in default init

        # A: [d_inner, d_state], real, negative — parametrised as -exp(A_log) for stability.
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: per-channel skip
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Mamba over a full sequence.

        Parameters
        ----------
        x : (B, L, D)

        Returns
        -------
        y : (B, L, D)
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv1d expects (B, d_inner, L)
        x_conv = rearrange(x_in, "b l d -> b d l")
        if (
            self.use_fast_path
            and _HAS_CAUSAL_CONV1D
            and x.is_cuda
            and self.conv1d.bias is not None
        ):
            x_conv = causal_conv1d_fn(
                x_conv,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None,
            )
        else:
            x_conv = self.conv1d(x_conv)[..., :L]
        x_conv = self.act(x_conv[..., :L])
        x_conv = rearrange(x_conv, "b d l -> b l d")

        # Project to delta, B, C
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        delta_proj, B_t, C_t = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj(delta_proj)  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if self.use_fast_path and _HAS_SELECTIVE_SCAN and x.is_cuda:
            # selective_scan_fn expects channel-first
            y = selective_scan_fn(
                rearrange(x_conv, "b l d -> b d l"),
                rearrange(delta, "b l d -> b d l"),
                A,
                rearrange(B_t, "b l n -> b n l"),
                rearrange(C_t, "b l n -> b n l"),
                self.D.float(),
                z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
            y = rearrange(y, "b d l -> b l d")
        else:
            y = self._selective_scan_ref(x_conv, delta, A, B_t, C_t, self.D)
        y = y * self.act(z)

        return self.out_proj(y)

    # ------------------------------------------------------------------
    @staticmethod
    def _selective_scan_ref(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B_t: torch.Tensor,
        C_t: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Reference (sequential) selective scan in pure PyTorch.

        Shapes
        ------
        u, delta : (B, L, d_inner)
        A : (d_inner, d_state)
        B_t, C_t : (B, L, d_state)
        D : (d_inner,)
        Returns y : (B, L, d_inner)
        """
        B_, L, d_inner = u.shape
        N = A.shape[1]
        delta = F.softplus(delta)  # ensure positive
        # Discretise: deltaA = exp(delta * A); deltaB_u = delta * B * u
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, N)
        deltaB_u = delta.unsqueeze(-1) * B_t.unsqueeze(2) * u.unsqueeze(-1)
        # Sequential scan
        h = u.new_zeros(B_, d_inner, N)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * D
        return y

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device, dtype) -> MambaState:
        return MambaState(
            conv_state=torch.zeros(
                batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype
            ),
            ssm_state=torch.zeros(
                batch_size, self.d_inner, self.d_state, device=device, dtype=dtype
            ),
        )

    # ------------------------------------------------------------------
    def step(
        self, x: torch.Tensor, state: MambaState
    ) -> Tuple[torch.Tensor, MambaState]:
        """Process one token. ``x`` has shape (B, D); returns (B, D)."""
        B = x.shape[0]
        xz = self.in_proj(x)  # (B, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)

        # Update conv state (shift left, append new)
        conv_state = torch.roll(state.conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_in
        # Conv weight: (d_inner, 1, d_conv) -> (d_inner, d_conv)
        conv_w = self.conv1d.weight.squeeze(1)
        x_conv = (conv_state * conv_w).sum(-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.act(x_conv)  # (B, d_inner)

        # Project
        x_dbl = self.x_proj(x_conv)
        delta_proj, B_t, C_t = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_proj))  # (B, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B_t.unsqueeze(1) * x_conv.unsqueeze(-1)
        ssm_state = deltaA * state.ssm_state + deltaB_u
        y = (ssm_state * C_t.unsqueeze(1)).sum(-1)  # (B, d_inner)
        y = y + x_conv * self.D
        y = y * self.act(z)
        out = self.out_proj(y)

        return out, MambaState(conv_state=conv_state, ssm_state=ssm_state)


def _set_no_weight_decay(module: nn.Module) -> None:
    """Mark Mamba's A_log and D parameters to skip weight decay."""
    for name, p in module.named_parameters():
        if name.endswith("A_log") or name.endswith(".D"):
            p._no_weight_decay = True
