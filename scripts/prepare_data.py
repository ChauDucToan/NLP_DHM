"""Download + clean + split the parallel corpus.

By default this pulls Helsinki-NLP/opus-100 (config "vi-zh", which is the
zh-vi pair on HuggingFace). Output is JSONL files under
``data/processed/``: ``train.jsonl``, ``valid.jsonl``, ``test.jsonl``.

Usage
-----
    python scripts/prepare_data.py --config configs/bi_mamba_55m.yaml

Optional overrides:
    --max-train-pairs 100000
    --dataset Helsinki-NLP/opus-100 --dataset-config vi-zh
    --custom-jsonl path/to/your.jsonl   (uses your own corpus instead of HF)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Make ``src/`` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import Pair, basic_clean, length_ok, write_jsonl
from bi_mamba_mt.utils import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
    p.add_argument("--dataset", default=None, help="HF dataset id (default: from config)")
    p.add_argument("--dataset-config", default=None, help="HF dataset config name")
    p.add_argument("--max-train-pairs", type=int, default=None)
    p.add_argument("--max-valid-pairs", type=int, default=None)
    p.add_argument("--max-test-pairs", type=int, default=None)
    p.add_argument(
        "--custom-jsonl",
        default=None,
        help="Path to custom JSONL with {'zh': ..., 'vi': ...}. Skips HF download.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override config.data.processed_dir.",
    )
    return p.parse_args()


def load_from_huggingface(dataset: str, config: str):
    from datasets import load_dataset

    ds = load_dataset(dataset, config)
    return ds


def iter_pairs_from_hf(ds, src_key: str, tgt_key: str):
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            tr = ex.get("translation", ex)
            zh = basic_clean(tr.get(src_key, ""))
            vi = basic_clean(tr.get(tgt_key, ""))
            yield split, Pair(zh=zh, vi=vi)


def iter_pairs_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield "train", Pair(
                zh=basic_clean(d["zh"]),
                vi=basic_clean(d["vi"]),
            )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    out_dir = Path(args.out_dir or data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(data_cfg.get("seed", 42))
    rng = random.Random(seed)

    min_len = int(data_cfg.get("min_len", 1))
    max_chars = int(data_cfg.get("max_len", 250)) * 4  # rough char limit

    splits = {"train": [], "validation": [], "test": []}

    if args.custom_jsonl:
        # Load all pairs from a custom JSONL, then create our own splits.
        all_pairs = []
        for _, p in iter_pairs_from_jsonl(args.custom_jsonl):
            if length_ok(p.zh, p.vi, min_len, max_chars):
                all_pairs.append(p)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
        n_test = int(min(args.max_test_pairs or data_cfg.get("max_test_pairs", 2000), n // 50))
        n_valid = int(
            min(args.max_valid_pairs or data_cfg.get("max_valid_pairs", 2000), n // 50)
        )
        splits["test"] = all_pairs[:n_test]
        splits["validation"] = all_pairs[n_test : n_test + n_valid]
        splits["train"] = all_pairs[n_test + n_valid :]
    else:
        dataset = args.dataset or data_cfg["dataset"]
        config = args.dataset_config or data_cfg["config"]
        src_key = data_cfg["src_lang_key"]
        tgt_key = data_cfg["tgt_lang_key"]
        print(f"Loading {dataset}/{config} from Hugging Face …", flush=True)
        ds = load_from_huggingface(dataset, config)
        for split, p in iter_pairs_from_hf(ds, src_key, tgt_key):
            if length_ok(p.zh, p.vi, min_len, max_chars):
                splits[split].append(p)
        rng.shuffle(splits["train"])

    # Apply caps
    caps = {
        "train": args.max_train_pairs or data_cfg.get("max_train_pairs"),
        "validation": args.max_valid_pairs or data_cfg.get("max_valid_pairs"),
        "test": args.max_test_pairs or data_cfg.get("max_test_pairs"),
    }
    for split, cap in caps.items():
        if cap is not None and len(splits[split]) > cap:
            splits[split] = splits[split][: int(cap)]

    counts = {}
    counts["train"] = write_jsonl(splits["train"], out_dir / "train.jsonl")
    counts["valid"] = write_jsonl(splits["validation"], out_dir / "valid.jsonl")
    counts["test"] = write_jsonl(splits["test"], out_dir / "test.jsonl")

    print("Wrote:")
    for k, v in counts.items():
        print(f"  {out_dir/(k + '.jsonl')}: {v} pairs")


if __name__ == "__main__":
    main()
