"""Train the shared SentencePiece BPE tokenizer.

Reads ``data/processed/train.jsonl`` (created by ``prepare_data.py``),
dumps a flat plain-text corpus with both Chinese and Vietnamese sentences,
and trains a shared BPE model.

Output: ``data/tokenizer/spm.model`` and ``data/tokenizer/spm.vocab``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import read_jsonl, write_plain_corpus
from bi_mamba_mt.tokenizer import TokenizerConfig, train_sentencepiece
from bi_mamba_mt.utils import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
    p.add_argument("--input-jsonl", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    tok_cfg = cfg["tokenizer"]

    in_path = Path(
        args.input_jsonl or Path(data_cfg["processed_dir"]) / "train.jsonl"
    )
    out_dir = Path(args.out_dir or data_cfg["tokenizer_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading parallel corpus from {in_path} …")
    pairs = read_jsonl(in_path)
    print(f"  {len(pairs)} pairs")

    plain = out_dir / "plain.txt"
    n = write_plain_corpus(pairs, plain)
    print(f"Wrote {n} lines to {plain}")

    print("Training SentencePiece BPE …")
    train_sentencepiece(
        text_files=[plain],
        model_prefix=out_dir / "spm",
        cfg=TokenizerConfig(
            vocab_size=int(tok_cfg["vocab_size"]),
            model_type=str(tok_cfg["model_type"]),
            character_coverage=float(tok_cfg["character_coverage"]),
        ),
    )
    print(f"Saved to {out_dir/'spm.model'} (vocab {tok_cfg['vocab_size']})")


if __name__ == "__main__":
    main()
