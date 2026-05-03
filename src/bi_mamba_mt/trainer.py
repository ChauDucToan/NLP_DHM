"""Training loop for BiMambaTranslator."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import BiMambaTranslator
from .tokenizer import PAD_ID
from .utils import amp_dtype


def cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.dim() < 2
            or name.endswith(".bias")
            or "LayerNorm" in name
            or "embedding" in name.lower()
            or getattr(p, "_no_weight_decay", False)
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
        eps=eps,
    )


@dataclass
class TrainConfig:
    output_dir: str = "runs/bi_mamba_55m"
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_steps: int = 60_000
    warmup_steps: int = 3_000
    lr: float = 3.0e-4
    min_lr: float = 1.0e-5
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1.0e-9
    grad_clip: float = 1.0
    label_smoothing: float = 0.08
    log_every: int = 50
    eval_every: int = 1_000
    save_every: int = 1_000
    amp_dtype: str = "bf16"
    # Exponential moving average of model weights for inference. Decay of
    # 0.9990 ≈ effective window of 1000 steps; 0.9999 ≈ 10000 steps.
    ema: bool = True
    ema_decay: float = 0.9990
    # Early stopping. Monitors ``ema_val_loss`` when EMA is enabled (smoother),
    # otherwise ``val_loss``. ``patience`` counts evaluations (each eval step
    # = one tick) without improvement before stopping. Set to 0 to disable.
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0


class EMA:
    """Exponential moving average of model parameters.

    Maintains a CPU/GPU copy of the model parameters that lags behind the
    training weights, and can be temporarily swapped in for evaluation /
    final checkpointing. EMA weights consistently give +0.3–1.0 BLEU on top
    of the raw last-step weights for NMT.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply(self, model: nn.Module) -> None:
        """Swap EMA weights into the model (remember to call ``restore`` after)."""
        self._backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name].data)
        self._backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].data.copy_(v.to(self.shadow[k].device))


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    smoothing: float,
    pad_id: int,
) -> torch.Tensor:
    """Standard label-smoothed CE that ignores pad_id."""
    n = logits.size(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth_loss
    mask = target.ne(pad_id).float()
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def train(
    model: BiMambaTranslator,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: TrainConfig,
    device: torch.device,
    log_callback=None,
    resume_checkpoint: str | Path | dict | None = None,
):
    """Run training. Saves checkpoints to ``cfg.output_dir``.

    ``log_callback`` (optional) takes a dict and is called every
    ``log_every`` steps. Useful for plotting in a notebook.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    resume_state = None
    if resume_checkpoint is not None:
        if isinstance(resume_checkpoint, (str, Path)):
            resume_state = torch.load(resume_checkpoint, map_location="cpu")
        else:
            resume_state = resume_checkpoint
        model.load_state_dict(resume_state["model"])

    optimizer = build_optimizer(
        model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps,
    )
    dtype = amp_dtype(cfg.amp_dtype)
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if (dtype is not None and device.type == "cuda")
        else nullcontext()
    )

    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema else None

    best_val_loss: float = float("inf")
    best_ema_val_loss: float = float("inf")
    # Early stopping bookkeeping. Counts evaluations since the monitored
    # metric last improved by at least ``min_delta``.
    es_metric_name = "ema_val_loss" if cfg.ema else "val_loss"
    es_best: float = float("inf")
    es_stale: int = 0
    early_stop = False

    step = 0
    if resume_state is not None:
        if resume_state.get("is_ema"):
            tqdm.write(
                "Warning: resuming from an EMA-weight checkpoint; raw optimizer "
                "state may not match those weights."
            )
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if scaler is not None and resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        if ema is not None and resume_state.get("ema") is not None:
            ema.load_state_dict(resume_state["ema"])
        step = int(resume_state.get("step", 0))
        best_val_loss = float(resume_state.get("best_val_loss", best_val_loss))
        best_ema_val_loss = float(
            resume_state.get("best_ema_val_loss", best_ema_val_loss)
        )
        es_best = float(resume_state.get("es_best", es_best))
        es_stale = int(resume_state.get("patience_counter", es_stale))
        tqdm.write(f"Resumed training state from step {step}")

    model.train()
    pbar = tqdm(total=cfg.max_steps, initial=step, desc="train", dynamic_ncols=True)
    t0 = time.time()
    accum = 0
    optimizer.zero_grad(set_to_none=True)

    train_iter = _infinite(train_loader)
    while step < cfg.max_steps and not early_stop:
        batch = next(train_iter)
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        src_pad_mask = batch["src_pad_mask"].to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(src, tgt_in, src_pad_mask=src_pad_mask)
            loss = label_smoothed_cross_entropy(
                logits, tgt_out, cfg.label_smoothing, PAD_ID
            )
            loss = loss / cfg.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum += 1
        if accum < cfg.grad_accum_steps:
            continue
        accum = 0

        # LR
        lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)

        step += 1
        pbar.update(1)

        if step % cfg.log_every == 0:
            tok_per_sec = (
                (step * cfg.batch_size * tgt_in.size(1)) / max(time.time() - t0, 1.0)
            )
            log = {
                "step": step,
                "loss": float(loss.item() * cfg.grad_accum_steps),
                "lr": lr,
                "tok/s": tok_per_sec,
            }
            pbar.set_postfix(loss=f"{log['loss']:.3f}", lr=f"{lr:.1e}")
            if log_callback is not None:
                log_callback(log)

        if val_loader is not None and step % cfg.eval_every == 0:
            val_loss = evaluate_loss(model, val_loader, device, dtype=dtype)
            log = {"step": step, "val_loss": val_loss}

            # Save best raw-weights checkpoint by val_loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_payload = {
                    "model": model.state_dict(),
                    "model_cfg": vars(model.cfg),
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_ema_val_loss": best_ema_val_loss,
                    "es_best": es_best,
                    "patience_counter": es_stale,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "train_cfg": asdict(cfg),
                    "is_best": True,
                }
                if ema is not None:
                    best_payload["ema"] = ema.state_dict()
                torch.save(best_payload, out_dir / "best.pt")
                tqdm.write(
                    f"  -> new best val_loss {val_loss:.4f} @ step {step} "
                    f"(saved best.pt)"
                )

            if ema is not None:
                ema.apply(model)
                ema_val_loss = evaluate_loss(model, val_loader, device, dtype=dtype)
                # Save best EMA-weights checkpoint by ema_val_loss while EMA is
                # still swapped into the model.
                if ema_val_loss < best_ema_val_loss:
                    best_ema_val_loss = ema_val_loss
                    best_ema_payload = {
                        "model": model.state_dict(),
                        "model_cfg": vars(model.cfg),
                        "step": step,
                        "ema_val_loss": ema_val_loss,
                        "best_val_loss": best_val_loss,
                        "best_ema_val_loss": best_ema_val_loss,
                        "es_best": es_best,
                        "patience_counter": es_stale,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "train_cfg": asdict(cfg),
                        "is_ema": True,
                        "is_best": True,
                    }
                    torch.save(best_ema_payload, out_dir / "best_ema.pt")
                    tqdm.write(
                        f"  -> new best ema_val_loss {ema_val_loss:.4f} @ step {step} "
                        f"(saved best_ema.pt)"
                    )
                ema.restore(model)
                log["ema_val_loss"] = ema_val_loss
                tqdm.write(
                    f"step {step} | val_loss {val_loss:.3f} (best {best_val_loss:.3f}) | "
                    f"ema_val_loss {ema_val_loss:.3f} (best {best_ema_val_loss:.3f})"
                )
            else:
                tqdm.write(
                    f"step {step} | val_loss {val_loss:.3f} (best {best_val_loss:.3f})"
                )
            if log_callback is not None:
                log_callback(log)

            # Early stopping check.
            if cfg.early_stopping_patience and cfg.early_stopping_patience > 0:
                metric_val = log.get(es_metric_name, val_loss)
                if metric_val < es_best - cfg.early_stopping_min_delta:
                    es_best = metric_val
                    es_stale = 0
                else:
                    es_stale += 1
                    tqdm.write(
                        f"  early-stop: no improvement on {es_metric_name} for "
                        f"{es_stale}/{cfg.early_stopping_patience} eval(s) "
                        f"(best {es_best:.4f})"
                    )
                    if es_stale >= cfg.early_stopping_patience:
                        tqdm.write(
                            f"  early-stop: triggered at step {step} "
                            f"(no {es_metric_name} improvement for "
                            f"{cfg.early_stopping_patience} evals). Stopping."
                        )
                        early_stop = True
            model.train()

        if step % cfg.save_every == 0 or step == cfg.max_steps or early_stop:
            ckpt_path = out_dir / f"checkpoint_step{step}.pt"
            payload = {
                "model": model.state_dict(),
                "model_cfg": vars(model.cfg),
                "step": step,
                "best_val_loss": best_val_loss,
                "best_ema_val_loss": best_ema_val_loss,
                "es_best": es_best,
                "patience_counter": es_stale,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "train_cfg": asdict(cfg),
            }
            if ema is not None:
                payload["ema"] = ema.state_dict()
            torch.save(payload, ckpt_path)
            # Also keep a 'latest' alias
            torch.save(payload, out_dir / "latest.pt")
            # And keep a separate 'latest_ema' alias holding the EMA weights
            # already swapped into the model state_dict for easy loading.
            if ema is not None:
                ema.apply(model)
                ema_payload = {
                    "model": model.state_dict(),
                    "model_cfg": vars(model.cfg),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "best_ema_val_loss": best_ema_val_loss,
                    "es_best": es_best,
                    "patience_counter": es_stale,
                    "train_cfg": asdict(cfg),
                    "is_ema": True,
                }
                torch.save(ema_payload, out_dir / "latest_ema.pt")
                ema.restore(model)
    pbar.close()


def _infinite(loader: DataLoader) -> Iterable[dict]:
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def evaluate_loss(
    model: BiMambaTranslator,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if (dtype is not None and device.type == "cuda")
        else nullcontext()
    )
    total = 0.0
    n_batches = 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        src_pad_mask = batch["src_pad_mask"].to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(src, tgt_in, src_pad_mask=src_pad_mask)
            loss = label_smoothed_cross_entropy(logits, tgt_out, 0.0, PAD_ID)
        total += float(loss.item())
        n_batches += 1
    return total / max(1, n_batches)
