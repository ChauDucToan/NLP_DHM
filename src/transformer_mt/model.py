"""Vanilla Transformer encoder-decoder translator (baseline).

Designed as a drop-in baseline for :class:`bi_mamba_mt.model.BiMambaTranslator`:
identical input/output API, identical tokenizer, identical training loop,
identical beam-search interface. Use this to ablate whether translation
quality is bottlenecked by the Mamba architecture or by data/tokenizer.

Architecture
------------
* Shared SentencePiece vocabulary, direction tag prepended to source.
* Pre-norm Transformer encoder (``n_encoder_layers`` layers).
* Pre-norm Transformer decoder with self- and cross-attention
  (``n_decoder_layers`` layers).
* Sinusoidal positional encodings, tied input/output embeddings.

The default hyper-parameters (d_model=384, 5+5 layers, d_ff=2048) yield
~31M parameters at vocab 16k — comparable to Bi-Mamba 32.4M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    d_model: int = 384
    n_heads: int = 8
    n_encoder_layers: int = 5
    n_decoder_layers: int = 5
    d_ff: int = 2048
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


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class TransformerTranslator(nn.Module):
    """Vanilla Transformer encoder-decoder. Same interface as ``BiMambaTranslator``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, max_len=max(cfg.max_src_len, cfg.max_tgt_len) + 8
        )
        self.dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_encoder_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.n_decoder_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.xavier_uniform_(module.weight)
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
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids) * self.embed_scale
        x = self.pos_enc(x)
        return self.dropout(x)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # nn.Transformer expects True at positions to mask out.
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    # ------------------------------------------------------------------
    def encode(
        self, src: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self._embed(src)
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._embed(tgt)
        causal = self._causal_mask(tgt.size(1), tgt.device)
        x = self.decoder(
            x,
            encoder,
            tgt_mask=causal,
            memory_key_padding_mask=src_pad_mask,
        )
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
        """Greedy / beam-search decoding.

        Returns ``list[B]`` of token id lists (without BOS, may include EOS).
        """
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
        B = src.size(0)
        encoder = self.encode(src, src_pad_mask=src_pad_mask)
        tgt = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs: List[List[int]] = [[] for _ in range(B)]
        for _ in range(max_len):
            logits = self.decode(tgt, encoder, src_pad_mask=src_pad_mask)
            next_tok = logits[:, -1].argmax(dim=-1)  # (B,)
            for b in range(B):
                if not finished[b]:
                    t = int(next_tok[b].item())
                    outputs[b].append(t)
                    if t == self.cfg.eos_id:
                        finished[b] = True
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
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
        """Per-example beam search (clarity over speed)."""
        device = src.device
        results: List[List[int]] = []
        for b in range(src.size(0)):
            src_b = src[b : b + 1]
            mask_b = src_pad_mask[b : b + 1] if src_pad_mask is not None else None
            encoder = self.encode(src_b, src_pad_mask=mask_b)  # (1, L, D)
            encoder = encoder.expand(beam_size, -1, -1).contiguous()
            mask_beam = (
                mask_b.expand(beam_size, -1).contiguous() if mask_b is not None else None
            )
            tgt = torch.full(
                (beam_size, 1), self.cfg.bos_id, dtype=torch.long, device=device
            )
            seq_logp = torch.zeros(beam_size, device=device)
            seq_logp[1:] = float("-inf")
            finished_seqs: List[tuple[float, list[int]]] = []
            for _step in range(max_len):
                logits = self.decode(tgt, encoder, src_pad_mask=mask_beam)
                logp = F.log_softmax(logits[:, -1].float(), dim=-1)  # (beam, V)
                cand = seq_logp.unsqueeze(-1) + logp
                flat = cand.view(-1)
                topk = torch.topk(flat, k=beam_size, dim=0)
                new_token = topk.indices % self.cfg.vocab_size
                beam_idx = topk.indices // self.cfg.vocab_size
                new_seq_logp = topk.values

                tgt = torch.cat([tgt[beam_idx], new_token.unsqueeze(1)], dim=1)
                live_seq_logp = []
                for k in range(beam_size):
                    tok = int(new_token[k].item())
                    if tok == self.cfg.eos_id:
                        seq_ids = tgt[k, 1:].tolist()  # skip BOS
                        lp = (
                            new_seq_logp[k].item()
                            / max(len(seq_ids), 1) ** length_penalty
                        )
                        finished_seqs.append((lp, seq_ids))
                        live_seq_logp.append(float("-inf"))
                    else:
                        live_seq_logp.append(new_seq_logp[k].item())
                seq_logp = torch.tensor(live_seq_logp, device=device)
                if all(s == float("-inf") for s in live_seq_logp):
                    break
            if not finished_seqs:
                best_k = int(torch.argmax(seq_logp).item())
                results.append(tgt[best_k, 1:].tolist())
            else:
                finished_seqs.sort(key=lambda x: x[0], reverse=True)
                results.append(finished_seqs[0][1])
        return results
