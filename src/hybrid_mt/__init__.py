"""Hybrid Mamba-Attention encoder-decoder translator.

Bi-Mamba encoder (re-uses :class:`bi_mamba_mt.modules.bi_mamba.BiMambaBlock`)
combined with a vanilla Transformer decoder (masked self-attention +
cross-attention + FFN, ``nn.TransformerDecoder``).

Designed as an ablation between :class:`bi_mamba_mt.model.BiMambaTranslator`
(Bi-Mamba enc + Mamba dec) and
:class:`transformer_mt.model.TransformerTranslator` (Transformer enc + dec):

* If Hybrid >> Bi-Mamba and ≈ Transformer ⇒ the Mamba decoder was the
  bottleneck.
* If Hybrid still far below Transformer ⇒ the Bi-Mamba encoder is
  insufficient for MT source representation.
* If Hybrid ≈ Transformer with fewer params/compute ⇒ keep this as the
  main model.

Public API matches the existing translators (``forward``, ``encode``,
``decode``, ``generate``, ``cfg``) so the shared :mod:`mt_base` training
loop, evaluator, and beam-search work without modification.
"""

from .model import HybridMambaAttentionTranslator, ModelConfig

__all__ = ["HybridMambaAttentionTranslator", "ModelConfig"]

__version__ = "0.1.0"
