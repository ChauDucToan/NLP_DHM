"""Grid-sweep beam size × length penalty and write BLEU/chrF to CSV.

The sweep is run **per direction**: for ``zh2vi`` we iterate ``beam × lp_zh2vi``
values, for ``vi2zh`` we iterate ``beam × lp_vi2zh``. Directions are independent
so we avoid the full cartesian product (which would waste compute).

The checkpoint is loaded once; only decoding parameters change per cell.

Example::

    python scripts/sweep_decode.py \\
        --config configs/bi_mamba_55m.yaml \\
        --checkpoint runs/bi_mamba_55m/best_ema.pt \\
        --num-samples 2000 \\
        --beams 1 2 4 6 \\
        --lp-zh2vi 0.8 0.9 1.0 1.1 1.2 \\
        --lp-vi2zh 0.6 0.8 0.9 1.0 \\
        --out runs/bi_mamba_55m/sweep.csv

Optionally add ``--length-buckets`` to also emit one CSV row per
(direction, bucket, beam, lp).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import read_jsonl
from bi_mamba_mt.evaluator import DEFAULT_LENGTH_BUCKETS, evaluate
from bi_mamba_mt.model import BiMambaTranslator, ModelConfig
from bi_mamba_mt.tokenizer import Tokenizer
from bi_mamba_mt.utils import get_device, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint to sweep (e.g. runs/bi_mamba_55m/best_ema.pt).",
    )
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument(
        "--beams",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6],
        help="Beam sizes to sweep.",
    )
    p.add_argument(
        "--lp-zh2vi",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.1, 1.2],
        help="Length penalties for zh→vi (Vietnamese targets tend to be longer).",
    )
    p.add_argument(
        "--lp-vi2zh",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 0.9, 1.0],
        help="Length penalties for vi→zh (Chinese targets tend to be shorter).",
    )
    p.add_argument(
        "--directions",
        nargs="+",
        default=["zh2vi", "vi2zh"],
        choices=["zh2vi", "vi2zh"],
    )
    p.add_argument(
        "--length-buckets",
        action="store_true",
        help="Also emit per-length-bucket rows (short/medium/long).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    return p.parse_args()


def _load(
    checkpoint: Path, cfg: dict, device: torch.device
) -> tuple[BiMambaTranslator, Tokenizer, list]:
    print(f"Loading checkpoint {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
    return model, tokenizer, test_pairs


def _sweep_direction(
    direction: str,
    lps: Sequence[float],
    beams: Sequence[int],
    *,
    model,
    tokenizer,
    test_pairs,
    device,
    max_len: int,
    length_buckets,
) -> List[dict]:
    rows: List[dict] = []
    for beam in beams:
        for lp in lps:
            res = evaluate(
                model,
                tokenizer,
                test_pairs,
                direction,
                beam_size=beam,
                length_penalty=lp,
                max_len=max_len,
                device=device,
                length_buckets=length_buckets,
            )
            print(
                f"[{direction}] beam={beam} lp={lp:.2f} "
                f"BLEU={res.bleu:.2f} chrF={res.chrf:.2f} n={res.n}"
            )
            rows.append(
                {
                    "direction": direction,
                    "beam": beam,
                    "length_penalty": lp,
                    "bucket": "all",
                    "n": res.n,
                    "bleu": round(res.bleu, 3),
                    "chrf": round(res.chrf, 3),
                }
            )
            for name, b in res.buckets.items():
                rows.append(
                    {
                        "direction": direction,
                        "beam": beam,
                        "length_penalty": lp,
                        "bucket": name,
                        "n": b.n,
                        "bleu": round(b.bleu, 3),
                        "chrf": round(b.chrf, 3),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()

    model, tokenizer, test_pairs = _load(Path(args.checkpoint), cfg, device)
    n = args.num_samples or int(cfg["eval"].get("num_samples", 1000))
    test_pairs = test_pairs[:n]
    max_len = int(cfg["eval"].get("max_decode_len", 256))
    length_buckets = list(DEFAULT_LENGTH_BUCKETS) if args.length_buckets else None

    rows: List[dict] = []
    if "zh2vi" in args.directions:
        rows.extend(
            _sweep_direction(
                "zh2vi",
                args.lp_zh2vi,
                args.beams,
                model=model,
                tokenizer=tokenizer,
                test_pairs=test_pairs,
                device=device,
                max_len=max_len,
                length_buckets=length_buckets,
            )
        )
    if "vi2zh" in args.directions:
        rows.extend(
            _sweep_direction(
                "vi2zh",
                args.lp_vi2zh,
                args.beams,
                model=model,
                tokenizer=tokenizer,
                test_pairs=test_pairs,
                device=device,
                max_len=max_len,
                length_buckets=length_buckets,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["direction", "beam", "length_penalty", "bucket", "n", "bleu", "chrf"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Print best per direction for convenience.
    for d in args.directions:
        dir_rows = [r for r in rows if r["direction"] == d and r["bucket"] == "all"]
        if not dir_rows:
            continue
        best = max(dir_rows, key=lambda r: r["bleu"])
        print(
            f"  best {d}: beam={best['beam']} lp={best['length_penalty']} "
            f"BLEU={best['bleu']:.2f} chrF={best['chrf']:.2f}"
        )


if __name__ == "__main__":
    main()
