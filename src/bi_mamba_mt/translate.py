"""Inference helpers: translate one or many sentences."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

from .model import BiMambaTranslator
from .tokenizer import PAD_ID, Tokenizer


def _pad_batch(
    batches: Sequence[List[int]], pad_id: int = PAD_ID
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(b) for b in batches)
    src = torch.full((len(batches), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batches):
        src[i, : len(b)] = torch.tensor(b, dtype=torch.long)
    return src, src.eq(pad_id)


@torch.no_grad()
def translate_batch(
    model: BiMambaTranslator,
    tokenizer: Tokenizer,
    sentences: Sequence[str],
    direction: str,
    max_len: int = 256,
    beam_size: int = 1,
    length_penalty: float = 1.0,
    device: torch.device | None = None,
) -> List[str]:
    """Translate a batch of sentences in a single direction.

    ``direction`` is one of "zh2vi" or "vi2zh".
    """
    device = device or next(model.parameters()).device
    model.eval()
    encoded = [tokenizer.encode_src(s, direction)[: model.cfg.max_src_len] for s in sentences]
    src, pad_mask = _pad_batch(encoded)
    src = src.to(device)
    pad_mask = pad_mask.to(device)
    out_ids = model.generate(
        src,
        src_pad_mask=pad_mask,
        max_len=max_len,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )
    return [tokenizer.decode(ids) for ids in out_ids]


@torch.no_grad()
def translate(
    model: BiMambaTranslator,
    tokenizer: Tokenizer,
    sentence: str,
    direction: str,
    **kwargs,
) -> str:
    return translate_batch(model, tokenizer, [sentence], direction, **kwargs)[0]


# ----------------------------------------------------------------------
# Multi-checkpoint ensemble decoding
# ----------------------------------------------------------------------
@torch.no_grad()
def _ensemble_beam_decode_one(
    models: Sequence[BiMambaTranslator],
    src: torch.Tensor,           # (1, L)
    src_pad_mask: torch.Tensor | None,
    max_len: int,
    beam_size: int,
    length_penalty: float,
) -> List[int]:
    """Beam search over the *averaged log-probabilities* of N models on a
    single example. Each model maintains its own decoder state cache; at
    every step we average their log-softmax outputs before the topk.
    """
    device = src.device
    M = len(models)
    # Per-model encoder cache + decoder state
    enc_caches: list[torch.Tensor] = []
    enc_masks: list[torch.Tensor | None] = []
    dec_states: list[list] = []
    for m in models:
        enc = m.encode(src, src_pad_mask=src_pad_mask)  # (1, L, D)
        enc = enc.expand(beam_size, -1, -1).contiguous()
        enc_caches.append(enc)
        if src_pad_mask is not None:
            enc_masks.append(src_pad_mask.expand(beam_size, -1).contiguous())
        else:
            enc_masks.append(None)
        dec_states.append(
            [layer.init_state(beam_size, device=device, dtype=enc.dtype) for layer in m.decoder_layers]
        )
    # Beam state is shared across models (we score on the averaged log-prob).
    bos_id = models[0].cfg.bos_id
    eos_id = models[0].cfg.eos_id
    vocab_size = models[0].cfg.vocab_size
    tokens = torch.full((beam_size,), bos_id, dtype=torch.long, device=device)
    seq_logp = torch.zeros(beam_size, device=device)
    seq_logp[1:] = float("-inf")  # only one live beam at start
    seqs: list[list[int]] = [[] for _ in range(beam_size)]
    finished_seqs: list[tuple[float, list[int]]] = []
    from .modules.decoder_block import DecoderState
    for _ in range(max_len):
        avg_logp = None
        for m_idx, m in enumerate(models):
            x_t = m.embedding(tokens)
            for li, layer in enumerate(m.decoder_layers):
                x_t, dec_states[m_idx][li] = layer.step(
                    x_t, dec_states[m_idx][li],
                    enc_caches[m_idx], encoder_padding_mask=enc_masks[m_idx],
                )
            x_t = m.decoder_norm(x_t)
            logits = m.lm_head(x_t)            # (beam, V)
            logp = F.log_softmax(logits.float(), dim=-1)
            avg_logp = logp if avg_logp is None else avg_logp + logp
        avg_logp = avg_logp / float(M)         # (beam, V)
        cand = seq_logp.unsqueeze(-1) + avg_logp
        flat = cand.view(-1)
        topk = torch.topk(flat, k=beam_size, dim=0)
        new_token = topk.indices % vocab_size
        beam_idx = topk.indices // vocab_size
        new_seq_logp = topk.values
        # Reorder per-model decoder states by chosen beams
        for m_idx in range(M):
            new_states = []
            for s in dec_states[m_idx]:
                new_states.append(
                    DecoderState(
                        mamba_state=type(s.mamba_state)(
                            conv_state=s.mamba_state.conv_state[beam_idx],
                            ssm_state=s.mamba_state.ssm_state[beam_idx],
                        )
                    )
                )
            dec_states[m_idx] = new_states
        seqs = [list(seqs[int(i)]) for i in beam_idx.tolist()]
        tokens = new_token
        live_seq_logp = []
        for k in range(beam_size):
            seqs[k].append(int(tokens[k].item()))
            if int(tokens[k].item()) == eos_id:
                lp = (
                    new_seq_logp[k].item()
                    / max(len(seqs[k]), 1) ** length_penalty
                )
                finished_seqs.append((lp, seqs[k]))
                live_seq_logp.append(float("-inf"))
            else:
                live_seq_logp.append(new_seq_logp[k].item())
        seq_logp = torch.tensor(live_seq_logp, device=device)
        if all(s == float("-inf") for s in live_seq_logp):
            break
    if finished_seqs:
        finished_seqs.sort(key=lambda x: x[0], reverse=True)
        return finished_seqs[0][1]
    best_k = int(torch.argmax(seq_logp).item())
    return seqs[best_k]


@torch.no_grad()
def ensemble_translate_batch(
    models: Sequence[BiMambaTranslator],
    tokenizer: Tokenizer,
    sentences: Sequence[str],
    direction: str,
    max_len: int = 256,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    device: torch.device | None = None,
) -> List[str]:
    """Beam-search translation that averages N models' next-token
    log-probabilities at every decoder step. All models must share the same
    tokenizer/vocab and special-token ids.

    Typically combine 3 checkpoints, e.g. ``best_ema``, ``avg_last5_ema``,
    and ``best`` \u2014 you'll usually get +0.5\u20131.5 BLEU vs the strongest single
    checkpoint at the cost of N\u00d7 inference time.
    """
    assert len(models) >= 1
    if len(models) == 1:
        return translate_batch(
            models[0], tokenizer, sentences, direction,
            max_len=max_len, beam_size=beam_size,
            length_penalty=length_penalty, device=device,
        )
    device = device or next(models[0].parameters()).device
    for m in models:
        m.eval()
    cfg0 = models[0].cfg
    encoded = [tokenizer.encode_src(s, direction)[: cfg0.max_src_len] for s in sentences]
    out_strings: List[str] = []
    for enc_ids in encoded:
        src = torch.tensor([enc_ids], dtype=torch.long, device=device)
        pad_mask = src.eq(PAD_ID)
        ids = _ensemble_beam_decode_one(
            models, src, pad_mask, max_len=max_len,
            beam_size=max(2, beam_size), length_penalty=length_penalty,
        )
        out_strings.append(tokenizer.decode(ids))
    return out_strings
