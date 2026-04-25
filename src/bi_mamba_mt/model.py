"""Bi-Mamba sequence-to-sequence translator.

Architecture
------------
* Shared SentencePiece vocabulary for Chinese + Vietnamese.
* Bidirectional Mamba encoder (``n_encoder_layers`` BiMamba blocks + FFN).
* Causal Mamba decoder with cross-attention (``n_decoder_layers`` blocks).
* Tied input embedding / lm_head.

A direction tag (``<2vi>`` or ``<2zh>``) is prepended to the source by
the data pipeline so a single model handles both translation directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.bi_mamba import BiMambaBlock
from .modules.decoder_block import DecoderBlock, DecoderState
from .modules.mamba_block import MambaBlock


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    n_cross_attn_heads: int = 8
    dropout: float = 0.1
    tie_embeddings: bool = True
    max_src_len: int = 256
    max_tgt_len: int = 256
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    zh2vi_id: int = 4
    vi2zh_id: int = 5


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.bi_mamba = BiMambaBlock(
            cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn_w1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.ffn_w2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.bi_mamba(self.norm1(x), key_padding_mask=padding_mask))
        x = x + self.dropout(self.ffn_w2(F.gelu(self.ffn_w1(self.norm2(x)))))
        return x


class BiMambaTranslator(nn.Module):
    """Bi-Mamba seq2seq translator (~55M parameters with default config)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.dropout = nn.Dropout(cfg.dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    n_cross_attn_heads=cfg.n_cross_attn_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (not only_trainable) or p.requires_grad
        )

    # ------------------------------------------------------------------
    def encode(
        self, src: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.dropout(self.embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, padding_mask=src_pad_mask)
        return self.encoder_norm(x)

    # ------------------------------------------------------------------
    def decode(
        self,
        tgt: torch.Tensor,
        encoder: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.dropout(self.embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(x, encoder, encoder_padding_mask=src_pad_mask)
        x = self.decoder_norm(x)
        return self.lm_head(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder = self.encode(src, src_pad_mask=src_pad_mask)
        return self.decode(tgt_in, encoder, src_pad_mask=src_pad_mask)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        beam_size: int = 1,
        length_penalty: float = 1.0,
    ) -> List[List[int]]:
        """Greedy / beam-search decoding. Returns list[B] of token id lists
        (without BOS, may include EOS)."""
        if beam_size <= 1:
            return self._greedy_decode(src, src_pad_mask=src_pad_mask, max_len=max_len)
        return self._beam_decode(
            src,
            src_pad_mask=src_pad_mask,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )

    # ------------------------------------------------------------------
    def _greedy_decode(
        self,
        src: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor],
        max_len: int,
    ) -> List[List[int]]:
        device = src.device
        B = src.shape[0]
        encoder = self.encode(src, src_pad_mask=src_pad_mask)
        states = [
            layer.init_state(B, device=device, dtype=encoder.dtype)
            for layer in self.decoder_layers
        ]
        tokens = torch.full(
            (B,), self.cfg.bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs: List[List[int]] = [[] for _ in range(B)]
        for _ in range(max_len):
            x_t = self.embedding(tokens)
            for i, layer in enumerate(self.decoder_layers):
                x_t, states[i] = layer.step(
                    x_t, states[i], encoder, encoder_padding_mask=src_pad_mask
                )
            x_t = self.decoder_norm(x_t)
            logits = self.lm_head(x_t)
            tokens = logits.argmax(dim=-1)
            for b in range(B):
                if not finished[b]:
                    tok = int(tokens[b].item())
                    outputs[b].append(tok)
                    if tok == self.cfg.eos_id:
                        finished[b] = True
            if finished.all():
                break
        return outputs

    # ------------------------------------------------------------------
    def _beam_decode(
        self,
        src: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor],
        max_len: int,
        beam_size: int,
        length_penalty: float,
    ) -> List[List[int]]:
        """Simple per-example beam search.

        For clarity we run the loop one example at a time; this is
        adequate for evaluation with ``beam_size`` ≈ 4.
        """
        device = src.device
        results: List[List[int]] = []
        for b in range(src.shape[0]):
            src_b = src[b : b + 1]
            mask_b = src_pad_mask[b : b + 1] if src_pad_mask is not None else None
            encoder = self.encode(src_b, src_pad_mask=mask_b)  # (1, L, D)
            # Expand to beam size
            encoder = encoder.expand(beam_size, -1, -1).contiguous()
            mask_beam = (
                mask_b.expand(beam_size, -1).contiguous() if mask_b is not None else None
            )
            states = [
                layer.init_state(beam_size, device=device, dtype=encoder.dtype)
                for layer in self.decoder_layers
            ]
            tokens = torch.full(
                (beam_size,), self.cfg.bos_id, dtype=torch.long, device=device
            )
            seq_logp = torch.zeros(beam_size, device=device)
            seq_logp[1:] = float("-inf")  # only one live beam at start
            seqs: List[List[int]] = [[] for _ in range(beam_size)]
            finished_seqs: List[tuple[float, list[int]]] = []
            for step in range(max_len):
                x_t = self.embedding(tokens)
                for i, layer in enumerate(self.decoder_layers):
                    x_t, states[i] = layer.step(
                        x_t, states[i], encoder, encoder_padding_mask=mask_beam
                    )
                x_t = self.decoder_norm(x_t)
                logits = self.lm_head(x_t)  # (beam, V)
                logp = F.log_softmax(logits.float(), dim=-1)
                # Score = old_logp + new logp
                cand = seq_logp.unsqueeze(-1) + logp  # (beam, V)
                flat = cand.view(-1)
                topk = torch.topk(flat, k=beam_size, dim=0)
                new_token = topk.indices % self.cfg.vocab_size
                beam_idx = topk.indices // self.cfg.vocab_size
                new_seq_logp = topk.values

                # Reorder per-layer states + sequences
                states = [
                    DecoderState(
                        mamba_state=type(s.mamba_state)(
                            conv_state=s.mamba_state.conv_state[beam_idx],
                            ssm_state=s.mamba_state.ssm_state[beam_idx],
                        )
                    )
                    for s in states
                ]
                seqs = [list(seqs[int(i)]) for i in beam_idx.tolist()]
                tokens = new_token
                live_seq_logp = []
                live_tokens = []
                live_idx = []
                for k in range(beam_size):
                    seqs[k].append(int(tokens[k].item()))
                    if int(tokens[k].item()) == self.cfg.eos_id:
                        # length-normalised score
                        lp = (
                            new_seq_logp[k].item()
                            / max(len(seqs[k]), 1) ** length_penalty
                        )
                        finished_seqs.append((lp, seqs[k]))
                        live_seq_logp.append(float("-inf"))
                    else:
                        live_seq_logp.append(new_seq_logp[k].item())
                        live_tokens.append(k)
                        live_idx.append(k)
                seq_logp = torch.tensor(live_seq_logp, device=device)
                if all(s == float("-inf") for s in live_seq_logp):
                    break
            # Pick best
            if not finished_seqs:
                # Fall back: best partial
                best_k = int(torch.argmax(seq_logp).item())
                results.append(seqs[best_k])
            else:
                finished_seqs.sort(key=lambda x: x[0], reverse=True)
                results.append(finished_seqs[0][1])
        return results
