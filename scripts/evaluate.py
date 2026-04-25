"""Evaluate a trained checkpoint with SacreBLEU + chrF (both directions)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import read_jsonl
from bi_mamba_mt.evaluator import evaluate
from bi_mamba_mt.model import BiMambaTranslator, ModelConfig
from bi_mamba_mt.tokenizer import Tokenizer
from bi_mamba_mt.utils import get_device, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
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
    return p.parse_args()


def _resolve_length_penalty(eval_cfg: Dict[str, Any], direction: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    lp = eval_cfg.get("length_penalty", 1.0)
    if isinstance(lp, dict):
        # Per-direction: {"zh2vi": 1.2, "vi2zh": 0.8}
        return float(lp.get(direction, 1.0))
    return float(lp)


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
    model = BiMambaTranslator(model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
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

    for d in args.directions:
        lp = _resolve_length_penalty(cfg["eval"], d, args.length_penalty)
        res = evaluate(
            model,
            tokenizer,
            test_pairs,
            d,
            beam_size=beam,
            length_penalty=lp,
            max_len=max_len,
            device=device,
        )
        print(
            f"[{d}] n={res.n} BLEU={res.bleu:.2f} chrF={res.chrf:.2f} "
            f"(beam={beam}, lp={lp:.2f})"
        )


if __name__ == "__main__":
    main()
