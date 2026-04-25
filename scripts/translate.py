"""Interactive / batch translation CLI.

Examples
--------
Single sentence (Chinese → Vietnamese):
    python scripts/translate.py --direction zh2vi --text "你好，世界！"

Read from a file (one sentence per line) and write to another:
    python scripts/translate.py --direction vi2zh --input src.txt --output out.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.model import BiMambaTranslator, ModelConfig
from bi_mamba_mt.tokenizer import Tokenizer
from bi_mamba_mt.translate import translate_batch
from bi_mamba_mt.utils import get_device, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--direction", choices=["zh2vi", "vi2zh"], required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--input", default=None, help="File with one sentence per line.")
    p.add_argument("--output", default=None)
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()

    ckpt_path = Path(
        args.checkpoint or Path(cfg["train"]["output_dir"]) / "latest.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ModelConfig(**ckpt.get("model_cfg", cfg["model"]))
    model = BiMambaTranslator(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = Tokenizer(Path(cfg["data"]["tokenizer_dir"]) / "spm.model")
    beam = args.beam_size or int(cfg["eval"].get("beam_size", 4))

    if args.text:
        out = translate_batch(model, tokenizer, [args.text], args.direction, beam_size=beam, device=device)
        print(out[0])
        return
    if args.input:
        sents = [s.strip() for s in Path(args.input).read_text(encoding="utf-8").splitlines() if s.strip()]
        results = []
        for i in range(0, len(sents), args.batch_size):
            batch = sents[i : i + args.batch_size]
            results.extend(
                translate_batch(model, tokenizer, batch, args.direction, beam_size=beam, device=device)
            )
        if args.output:
            Path(args.output).write_text("\n".join(results) + "\n", encoding="utf-8")
            print(f"Wrote {len(results)} lines to {args.output}")
        else:
            for s in results:
                print(s)
        return
    print("Provide --text or --input.")
    sys.exit(1)


if __name__ == "__main__":
    main()
