#!/usr/bin/env python
"""Average the weights of the last N checkpoints.

This is the standard "checkpoint averaging" trick from Marian / Fairseq:
averaging the parameters of the last few model checkpoints typically gives
+0.3–0.8 BLEU on top of any single checkpoint, because it smooths out the
high-frequency oscillation in the final phase of training.

Usage:
    python scripts/avg_ckpts.py --ckpts-dir runs/bi_mamba_55m --n 5 \
        --output runs/bi_mamba_55m/avg_last5.pt

By default we average ``checkpoint_step*.pt`` files (the periodic snapshots).
Pass ``--ema`` to average their EMA shadow weights instead, which often gives
yet another small bump on top.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import torch


_STEP_RE = re.compile(r"checkpoint_step(\d+)\.pt$")


def _list_checkpoints(d: Path) -> List[Path]:
    cps = []
    for p in d.glob("checkpoint_step*.pt"):
        m = _STEP_RE.search(p.name)
        if m:
            cps.append((int(m.group(1)), p))
    cps.sort()
    return [p for _, p in cps]


def average(ckpts: List[Path], output: Path, *, key: str = "model") -> None:
    if not ckpts:
        raise SystemExit("No checkpoints to average.")
    print(f"Averaging {len(ckpts)} checkpoints (key='{key}')")
    for c in ckpts:
        print(f"  - {c.name}")

    sd = None
    n = len(ckpts)
    base_payload = None
    for c in ckpts:
        d = torch.load(c, map_location="cpu", weights_only=False)
        base_payload = d
        if key not in d:
            raise SystemExit(
                f"Checkpoint {c} has no '{key}' entry. "
                f"Available keys: {sorted(d.keys())}"
            )
        cur = d[key]
        if sd is None:
            sd = {k: v.detach().to(torch.float64).clone() for k, v in cur.items()}
        else:
            for k in sd:
                sd[k].add_(cur[k].detach().to(torch.float64))

    for k in sd:
        sd[k].div_(n)

    out_payload = {**(base_payload or {})}
    # Start from the *full* state_dict of the last checkpoint (so buffers /
    # non-trainable tensors are preserved) and override the trainable params
    # with the averaged values.
    full_sd = {k: v.clone() for k, v in base_payload["model"].items()}
    n_overridden = 0
    for k, v in sd.items():
        if k in full_sd:
            full_sd[k] = v.to(full_sd[k].dtype)
            n_overridden += 1
    print(f"Overrode {n_overridden}/{len(sd)} keys into the base state_dict")
    out_payload["model"] = full_sd
    out_payload["averaged_from"] = [c.name for c in ckpts]
    out_payload["averaged_n"] = n
    out_payload.pop("ema", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output)
    print(f"Saved averaged checkpoint: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpts-dir",
        type=Path,
        default=Path("runs/bi_mamba_55m"),
        help="Directory containing checkpoint_step*.pt files",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of most-recent checkpoints to average",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <ckpts-dir>/avg_last<N>.pt",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Average the EMA shadow weights instead of the raw model weights",
    )
    args = parser.parse_args()

    ckpts = _list_checkpoints(args.ckpts_dir)
    if len(ckpts) == 0:
        raise SystemExit(f"No checkpoint_step*.pt found in {args.ckpts_dir}")
    if args.n > 0:
        ckpts = ckpts[-args.n :]

    output = args.output or args.ckpts_dir / (
        f"avg_last{args.n}{'_ema' if args.ema else ''}.pt"
    )
    average(ckpts, output, key="ema" if args.ema else "model")


if __name__ == "__main__":
    main()
