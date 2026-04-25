"""Evaluate translation quality with SacreBLEU + chrF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import sacrebleu
import torch
from tqdm.auto import tqdm

from .data import Pair
from .model import BiMambaTranslator
from .tokenizer import Tokenizer
from .translate import ensemble_translate_batch, translate_batch


@dataclass
class EvalResult:
    bleu: float
    chrf: float
    direction: str
    n: int


def evaluate(
    model: BiMambaTranslator | Sequence[BiMambaTranslator],
    tokenizer: Tokenizer,
    pairs: Sequence[Pair],
    direction: str,  # "zh2vi" or "vi2zh"
    *,
    batch_size: int = 16,
    beam_size: int = 4,
    length_penalty: float = 1.0,
    max_len: int = 256,
    device: torch.device | None = None,
) -> EvalResult:
    """Run BLEU + chrF on a single direction.

    ``model`` may also be a list / tuple of models; in that case decoding is
    done with ensemble (averaged log-probabilities at every step).
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

    is_ensemble = isinstance(model, (list, tuple))
    if is_ensemble and len(model) == 1:
        model = model[0]
        is_ensemble = False

    hyps: List[str] = []
    desc = f"eval {direction}{' (ensemble x' + str(len(model)) + ')' if is_ensemble else ''}"
    for i in tqdm(range(0, len(sources), batch_size), desc=desc):
        batch = sources[i : i + batch_size]
        if is_ensemble:
            out = ensemble_translate_batch(
                model,
                tokenizer,
                batch,
                direction,
                beam_size=beam_size,
                length_penalty=length_penalty,
                max_len=max_len,
                device=device,
            )
        else:
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
    # SacreBLEU expects refs as list-of-references; here we have a single ref.
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh" if bleu_lang == "zh" else "13a")
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return EvalResult(
        bleu=float(bleu.score),
        chrf=float(chrf.score),
        direction=direction,
        n=len(hyps),
    )
