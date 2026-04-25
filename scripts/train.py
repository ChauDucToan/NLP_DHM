"""End-to-end training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import Collator, SortPoolBatchSampler, TranslationDataset, read_jsonl
from bi_mamba_mt.model import BiMambaTranslator, ModelConfig
from bi_mamba_mt.tokenizer import Tokenizer
from bi_mamba_mt.trainer import TrainConfig, train
from bi_mamba_mt.utils import (
    count_parameters,
    get_device,
    human_format,
    load_yaml,
    save_yaml,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg["train"].get("seed", 42)))
    device = get_device()

    # Tokenizer
    tok_dir = Path(cfg["data"]["tokenizer_dir"])
    tokenizer = Tokenizer(tok_dir / "spm.model")

    # Data
    proc_dir = Path(cfg["data"]["processed_dir"])
    train_pairs = read_jsonl(proc_dir / "train.jsonl")
    valid_pairs = read_jsonl(proc_dir / "valid.jsonl")
    print(f"Train pairs: {len(train_pairs)} | Valid pairs: {len(valid_pairs)}")

    bpe_dropout = float(cfg["train"].get("bpe_dropout", 0.0))
    if bpe_dropout > 0.0:
        print(f"BPE-dropout enabled at \u03b1={bpe_dropout}")
    train_ds = TranslationDataset(
        pairs=train_pairs,
        tokenizer=tokenizer,
        max_src_len=int(cfg["model"]["max_src_len"]),
        max_tgt_len=int(cfg["model"]["max_tgt_len"]),
        bidirectional=bool(cfg["data"].get("bidirectional", True)),
        seed=int(cfg["train"].get("seed", 42)),
        bpe_dropout=bpe_dropout,
    )
    valid_ds = TranslationDataset(
        pairs=valid_pairs,
        tokenizer=tokenizer,
        max_src_len=int(cfg["model"]["max_src_len"]),
        max_tgt_len=int(cfg["model"]["max_tgt_len"]),
        bidirectional=bool(cfg["data"].get("bidirectional", True)),
        seed=int(cfg["train"].get("seed", 42)) + 1,
        # NEVER apply BPE-dropout to validation/test \u2014 we want a stable val_loss.
        bpe_dropout=0.0,
    )
    collate = Collator(pad_id=int(cfg["model"]["pad_id"]))

    bucket = bool(cfg["train"].get("length_bucket", True))
    pool_factor = int(cfg["train"].get("bucket_pool_factor", 100))
    if bucket:
        print(
            f"Using length-bucketed sampler "
            f"(pool={pool_factor}*batch={int(cfg['train']['batch_size'])})"
        )
        train_sampler = SortPoolBatchSampler(
            lengths=train_ds.bucket_lengths(),
            batch_size=int(cfg["train"]["batch_size"]),
            pool_factor=pool_factor,
            shuffle=True,
            drop_last=True,
            seed=int(cfg["train"].get("seed", 42)),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=int(cfg["train"].get("num_workers", 2)),
            pin_memory=bool(cfg["train"].get("pin_memory", True)) and device.type == "cuda",
            collate_fn=collate,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"].get("num_workers", 2)),
            pin_memory=bool(cfg["train"].get("pin_memory", True)) and device.type == "cuda",
            collate_fn=collate,
            drop_last=True,
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=bool(cfg["train"].get("pin_memory", True)) and device.type == "cuda",
        collate_fn=collate,
    )

    # Model
    model_cfg = ModelConfig(**cfg["model"])
    model = BiMambaTranslator(model_cfg)
    n_params = count_parameters(model)
    print(f"Model parameters: {human_format(n_params)} ({n_params:,})")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume} (step {ckpt.get('step', '?')})")

    # Filter to only keys TrainConfig knows about (others are loader/seed knobs).
    import dataclasses

    tc_keys = {f.name for f in dataclasses.fields(TrainConfig)}
    train_cfg = TrainConfig(**{k: v for k, v in cfg["train"].items() if k in tc_keys})
    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config.yaml")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        cfg=train_cfg,
        device=device,
    )

    final = out_dir / "final.pt"
    torch.save({"model": model.state_dict(), "model_cfg": vars(model_cfg)}, final)
    print(f"Saved final model: {final}")


if __name__ == "__main__":
    main()
