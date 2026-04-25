"""Evaluate a trained checkpoint with SacreBLEU + chrF (both directions)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    p.add_argument("--checkpoint", default=None,
                   help="Default: <output_dir>/latest.pt from the config")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--directions", nargs="+", default=["zh2vi", "vi2zh"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()

    ckpt_path = Path(
        args.checkpoint or Path(cfg["train"]["output_dir"]) / "latest.pt"
    )
    print(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt.get("model_cfg", cfg["model"]))
    model = BiMambaTranslator(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = Tokenizer(Path(cfg["data"]["tokenizer_dir"]) / "spm.model")
    test_pairs = read_jsonl(Path(cfg["data"]["processed_dir"]) / "test.jsonl")
    n = args.num_samples or int(cfg["eval"].get("num_samples", 1000))
    test_pairs = test_pairs[:n]

    beam = args.beam_size or int(cfg["eval"].get("beam_size", 4))
    max_len = int(cfg["eval"].get("max_decode_len", 256))

    for d in args.directions:
        res = evaluate(
            model,
            tokenizer,
            test_pairs,
            d,
            beam_size=beam,
            max_len=max_len,
            device=device,
        )
        print(
            f"[{d}] n={res.n} BLEU={res.bleu:.2f} chrF={res.chrf:.2f}"
        )


if __name__ == "__main__":
    main()
