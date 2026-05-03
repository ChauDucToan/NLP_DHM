"""Evaluate a Hybrid Mamba-Attention checkpoint with SacreBLEU + chrF.

Same CLI as :mod:`scripts.evaluate` and :mod:`scripts.evaluate_transformer`
but loads :class:`hybrid_mt.model.HybridMambaAttentionTranslator`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_mt.model import HybridMambaAttentionTranslator, ModelConfig
from mt_base.data import read_jsonl
from mt_base.evaluator import DEFAULT_LENGTH_BUCKETS, evaluate
from mt_base.tokenizer import Tokenizer
from mt_base.utils import get_device, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/hybrid_mamba_attention.yaml")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Default: <output_dir>/latest.pt from the config. "
             "Pass an averaged or EMA checkpoint to get the best BLEU.",
    )
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Override length penalty (single value applied to all directions).",
    )
    p.add_argument("--directions", nargs="+", default=["zh2vi", "vi2zh"])
    p.add_argument(
        "--length-penalty-zh2vi",
        type=float,
        default=None,
        help="Override length penalty for zh→vi direction only.",
    )
    p.add_argument(
        "--length-penalty-vi2zh",
        type=float,
        default=None,
        help="Override length penalty for vi→zh direction only.",
    )
    p.add_argument(
        "--length-buckets",
        action="store_true",
        help="Also print per-length-bucket BLEU/chrF breakdown "
             "(short: <20 src chars, medium: 20–50, long: ≥50).",
    )
    p.add_argument(
        "--allow-non-strict",
        action="store_true",
        help="Permit checkpoints with missing/unexpected keys "
             "(e.g. averaged checkpoints). Default is strict.",
    )
    return p.parse_args()


def _resolve_length_penalty(
    eval_cfg: Dict[str, Any],
    direction: str,
    global_override: float | None,
    per_direction_override: Dict[str, float | None] | None = None,
) -> float:
    """Resolve effective length penalty.

    Priority: per-direction CLI override > global CLI override > config dict.
    """
    if per_direction_override and per_direction_override.get(direction) is not None:
        return float(per_direction_override[direction])
    if global_override is not None:
        return float(global_override)
    lp = eval_cfg.get("length_penalty", 1.0)
    if isinstance(lp, dict):
        return float(lp.get(direction, 1.0))
    return float(lp)


def _load_strict(model: torch.nn.Module, state: dict, allow_non_strict: bool) -> None:
    """Load with strict=True; on failure, print missing/unexpected keys
    and either raise (default) or fall back to ``strict=False``.
    """
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError as e:
        # PyTorch's RuntimeError lists missing & unexpected keys
        print(f"Strict checkpoint load failed:\n{e}")
        if not allow_non_strict:
            raise SystemExit(
                "Refusing to load checkpoint non-strictly. "
                "Pass --allow-non-strict to override."
            )
        print("Falling back to strict=False (--allow-non-strict).")
        model.load_state_dict(state, strict=False)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()

    ckpt_path = Path(
        args.checkpoint or Path(cfg["train"]["output_dir"]) / "latest.pt"
    )
    print(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**ckpt.get("model_cfg", cfg["model"]))
    model = HybridMambaAttentionTranslator(model_cfg).to(device)
    _load_strict(model, ckpt["model"], allow_non_strict=args.allow_non_strict)
    model.eval()
    if ckpt.get("is_ema"):
        print("  (using EMA weights)")
    if ckpt.get("averaged_n"):
        print(f"  (averaged from {ckpt['averaged_n']} checkpoints)")

    tokenizer = Tokenizer(Path(cfg["data"]["tokenizer_dir"]) / "spm.model")
    test_pairs = read_jsonl(Path(cfg["data"]["processed_dir"]) / "test.jsonl")
    n = args.num_samples or int(cfg["eval"].get("num_samples", 1000))
    test_pairs = test_pairs[:n]

    beam = args.beam_size or int(cfg["eval"].get("beam_size", 4))
    max_len = int(cfg["eval"].get("max_decode_len", 256))

    per_dir = {
        "zh2vi": args.length_penalty_zh2vi,
        "vi2zh": args.length_penalty_vi2zh,
    }
    buckets = list(DEFAULT_LENGTH_BUCKETS) if args.length_buckets else None

    for d in args.directions:
        lp = _resolve_length_penalty(cfg["eval"], d, args.length_penalty, per_dir)
        res = evaluate(
            model,
            tokenizer,
            test_pairs,
            d,
            beam_size=beam,
            length_penalty=lp,
            max_len=max_len,
            device=device,
            length_buckets=buckets,
        )
        print(
            f"[{d}] n={res.n} BLEU={res.bleu:.2f} chrF={res.chrf:.2f} "
            f"(beam={beam}, lp={lp:.2f})"
        )
        for name, b in res.buckets.items():
            print(
                f"    bucket={name:<6} n={b.n:<5} "
                f"BLEU={b.bleu:.2f} chrF={b.chrf:.2f}"
            )


if __name__ == "__main__":
    main()
