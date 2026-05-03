"""Evaluate translation quality with SacreBLEU + chrF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import sacrebleu
import torch
from tqdm.auto import tqdm

import torch.nn as nn

from .data import Pair
from .tokenizer import Tokenizer
from .translate import translate_batch

# Default source-length buckets (by character count of the source sentence).
# Tuned for mixed zh/vi: short chat, medium sentences, longer paragraphs.
DEFAULT_LENGTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("short", 0, 20),
    ("medium", 20, 50),
    ("long", 50, 10_000),
)


@dataclass
class EvalResult:
    bleu: float
    chrf: float
    direction: str
    n: int
    buckets: Dict[str, "EvalResult"] = field(default_factory=dict)


def _corpus_scores(hyps: Sequence[str], refs: Sequence[str], bleu_lang: str) -> Tuple[float, float]:
    bleu = sacrebleu.corpus_bleu(
        list(hyps), [list(refs)], tokenize="zh" if bleu_lang == "zh" else "13a"
    )
    chrf = sacrebleu.corpus_chrf(list(hyps), [list(refs)])
    return float(bleu.score), float(chrf.score)


def evaluate(
    model: nn.Module,
    tokenizer: Tokenizer,
    pairs: Sequence[Pair],
    direction: str,  # "zh2vi" or "vi2zh"
    *,
    batch_size: int = 16,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    max_len: int = 256,
    device: torch.device | None = None,
    length_buckets: Sequence[Tuple[str, int, int]] | None = None,
) -> EvalResult:
    """Translate ``pairs`` in ``direction`` and return corpus BLEU/chrF.

    If ``length_buckets`` is given (list of ``(name, lo, hi)`` with source-char
    bounds ``[lo, hi)``), per-bucket BLEU/chrF are also attached under
    ``result.buckets``.
    """
    if direction == "zh2vi":
        sources = [p.zh for p in pairs]
        refs = [p.vi for p in pairs]
        bleu_lang = "vi"
    elif direction == "vi2zh":
        sources = [p.vi for p in pairs]
        refs = [p.zh for p in pairs]
        bleu_lang = "zh"
    else:
        raise ValueError(direction)

    hyps: List[str] = []
    for i in tqdm(range(0, len(sources), batch_size), desc=f"eval {direction}"):
        batch = sources[i : i + batch_size]
        out = translate_batch(
            model,
            tokenizer,
            batch,
            direction,
            beam_size=beam_size,
            length_penalty=length_penalty,
            max_len=max_len,
            device=device,
        )
        hyps.extend(out)

    bleu, chrf = _corpus_scores(hyps, refs, bleu_lang)
    result = EvalResult(bleu=bleu, chrf=chrf, direction=direction, n=len(hyps))

    if length_buckets:
        src_lens = [len(s) for s in sources]
        for name, lo, hi in length_buckets:
            idx = [i for i, ln in enumerate(src_lens) if lo <= ln < hi]
            if not idx:
                continue
            b_hyps = [hyps[i] for i in idx]
            b_refs = [refs[i] for i in idx]
            b_bleu, b_chrf = _corpus_scores(b_hyps, b_refs, bleu_lang)
            result.buckets[name] = EvalResult(
                bleu=b_bleu, chrf=b_chrf, direction=direction, n=len(idx)
            )

    return result

