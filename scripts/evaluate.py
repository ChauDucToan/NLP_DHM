"""Evaluate a trained checkpoint with SacreBLEU + chrF (both directions)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import read_jsonl
from bi_mamba_mt.evaluator import DEFAULT_LENGTH_BUCKETS, evaluate
from bi_mamba_mt.model import BiMambaTranslator, ModelConfig
from bi_mamba_mt.tokenizer import Tokenizer
from mt_base.eval_utils import (
    device_info,
    infer_eval_label,
    select_eval_subset,
    write_json,
)
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
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="If set, evaluate a reproducible random subset instead of the first N pairs.",
    )
    p.add_argument("--beam-size", type=int, default=None)
    p.add_argument(
        "--max-decode-len",
        type=int,
        default=None,
        help="Override max decode length for reporting and controlled ablations.",
    )
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
        "--runtime-out",
        type=Path,
        default=None,
        help="Optional JSON file for runtime metadata, device info, and decoded scores.",
    )
    return p.parse_args()


def _resolve_length_penalty(
    eval_cfg: Dict[str, Any],
    direction: str,
    global_override: float | None,
    per_direction_override: Dict[str, float | None] | None = None,
) -> float:
    """Resolve effective length penalty for a direction.

    Priority: per-direction CLI override > global CLI override > config dict.
    """
    if per_direction_override and per_direction_override.get(direction) is not None:
        return float(per_direction_override[direction])
    if global_override is not None:
        return float(global_override)
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
    test_pairs_all = read_jsonl(Path(cfg["data"]["processed_dir"]) / "test.jsonl")
    n = args.num_samples or int(cfg["eval"].get("num_samples", 1000))
    test_pairs, subset_meta = select_eval_subset(
        test_pairs_all,
        num_samples=n,
        sample_seed=args.sample_seed,
    )

    beam = args.beam_size or int(cfg["eval"].get("beam_size", 4))
    max_len = args.max_decode_len or int(cfg["eval"].get("max_decode_len", 256))
    eval_label = infer_eval_label(
        num_total_pairs=subset_meta["num_total_pairs"],
        num_selected_pairs=subset_meta["num_selected_pairs"],
        beam_size=beam,
    )
    print(
        f"Eval mode: {eval_label} | subset={subset_meta['subset_kind']} "
        f"({subset_meta['num_selected_pairs']}/{subset_meta['num_total_pairs']}, "
        f"seed={subset_meta['sample_seed']}) | beam={beam} | max_len={max_len}"
    )

    per_dir = {
        "zh2vi": args.length_penalty_zh2vi,
        "vi2zh": args.length_penalty_vi2zh,
    }
    buckets = list(DEFAULT_LENGTH_BUCKETS) if args.length_buckets else None
    runtime_payload: dict[str, Any] = {
        "model_kind": "mamba",
        "checkpoint": str(ckpt_path),
        "eval_label": eval_label,
        "config_path": args.config,
        "beam_size": beam,
        "max_decode_len": max_len,
        "length_buckets": bool(args.length_buckets),
        "directions": list(args.directions),
        "subset": subset_meta,
        "device": device_info(device),
        "timings_sec": {},
        "results": {},
    }

    t0 = time.perf_counter()
    for d in args.directions:
        lp = _resolve_length_penalty(cfg["eval"], d, args.length_penalty, per_dir)
        t_dir = time.perf_counter()
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
        runtime_payload["timings_sec"][d] = round(time.perf_counter() - t_dir, 3)
        runtime_payload["results"][d] = {
            "n": res.n,
            "bleu": round(res.bleu, 4),
            "chrf": round(res.chrf, 4),
            "length_penalty": lp,
            "buckets": {
                name: {
                    "n": b.n,
                    "bleu": round(b.bleu, 4),
                    "chrf": round(b.chrf, 4),
                }
                for name, b in res.buckets.items()
            },
        }
        print(
            f"[{d}] n={res.n} BLEU={res.bleu:.2f} chrF={res.chrf:.2f} "
            f"(beam={beam}, lp={lp:.2f})"
        )
        for name, b in res.buckets.items():
            print(
                f"    bucket={name:<6} n={b.n:<5} "
                f"BLEU={b.bleu:.2f} chrF={b.chrf:.2f}"
            )
    runtime_payload["timings_sec"]["total"] = round(time.perf_counter() - t0, 3)
    if args.runtime_out is not None:
        write_json(args.runtime_out, runtime_payload)
        print(f"Saved runtime metadata to {args.runtime_out}")


if __name__ == "__main__":
    main()
