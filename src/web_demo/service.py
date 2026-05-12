"""Shared model-loading and translation services for the web demo."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from bi_mamba_mt import BiMambaTranslator
from bi_mamba_mt.model import ModelConfig as BiMambaConfig
from mt_base.tokenizer import Tokenizer
from mt_base.translate import translate_batch
from mt_base.utils import get_device, load_yaml
from hybrid_mt import HybridMambaAttentionTranslator
from hybrid_mt.model import ModelConfig as HybridConfig
from transformer_mt import TransformerTranslator
from transformer_mt.model import ModelConfig as TransformerConfig


CHECKPOINT_CANDIDATES = (
    "avg_last5_ema.pt",
    "best_ema.pt",
    "avg_last5.pt",
    "best.pt",
    "latest_ema.pt",
    "latest.pt",
    "final.pt",
)

WARMUP_SENTENCES = {
    "zh2vi": "你好，世界！",
    "vi2zh": "xin chao the gioi",
}

Direction = str
ModelName = str
BuildModelFn = Callable[[dict[str, Any]], nn.Module]
TranslateBatchFn = Callable[..., list[str]]
TokenizerFactory = Callable[[str | Path], Tokenizer]
YamlLoader = Callable[[str | Path], dict[str, Any]]
CheckpointLoader = Callable[..., dict[str, Any]]


def _build_mamba(model_cfg: dict[str, Any]) -> nn.Module:
    return BiMambaTranslator(BiMambaConfig(**model_cfg))


def _build_hybrid(model_cfg: dict[str, Any]) -> nn.Module:
    return HybridMambaAttentionTranslator(HybridConfig(**model_cfg))


def _build_transformer(model_cfg: dict[str, Any]) -> nn.Module:
    return TransformerTranslator(TransformerConfig(**model_cfg))


def _lang_pair(direction: Direction) -> tuple[str, str]:
    if direction == "zh2vi":
        return "zh", "vi"
    if direction == "vi2zh":
        return "vi", "zh"
    raise ValueError(f"Unsupported direction: {direction}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    config_path: Path
    run_dir: Path
    build_model: BuildModelFn
    checkpoint_candidates: tuple[str, ...] = CHECKPOINT_CANDIDATES


@dataclass(frozen=True)
class CacheKey:
    model: str
    direction: str
    text: str
    beam_size: int
    length_penalty: float
    max_decode_len: int


@dataclass(frozen=True)
class CacheValue:
    translated_text: str
    source_lang: str
    target_lang: str


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: nn.Module
    checkpoint_path: Path
    beam_size: int
    max_decode_len: int
    length_penalty: dict[str, float]
    loaded_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class TranslationResultCache:
    """Thread-safe in-memory TTL + LRU cache for translation results."""

    def __init__(
        self,
        max_entries: int = 512,
        ttl_seconds: int = 3600,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.time_fn = time_fn
        self._entries: OrderedDict[CacheKey, tuple[float, CacheValue]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> CacheValue | None:
        now = self.time_fn()
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            created_at, value = item
            if now - created_at > self.ttl_seconds:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return value

    def set(self, key: CacheKey, value: CacheValue) -> None:
        with self._lock:
            self._entries[key] = (self.time_fn(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


def default_registry(repo_root: Path | None = None) -> dict[ModelName, ModelSpec]:
    root = repo_root or Path(__file__).resolve().parents[2]
    return {
        "mamba": ModelSpec(
            name="mamba",
            kind="mamba",
            config_path=root / "configs/bi_mamba_55m.yaml",
            run_dir=root / "runs/bi_mamba_55m",
            build_model=_build_mamba,
        ),
        "hybrid": ModelSpec(
            name="hybrid",
            kind="hybrid",
            config_path=root / "configs/hybrid_mamba_attention.yaml",
            run_dir=root / "runs/hybrid_mamba_attention",
            build_model=_build_hybrid,
        ),
        "transformer": ModelSpec(
            name="transformer",
            kind="transformer",
            config_path=root / "configs/transformer_30m.yaml",
            run_dir=root / "runs/transformer_30m",
            build_model=_build_transformer,
        ),
    }


class ModelManager:
    """Singleton-style manager that preloads and reuses all three models."""

    def __init__(
        self,
        registry: dict[ModelName, ModelSpec] | None = None,
        *,
        tokenizer_path: Path | None = None,
        device: torch.device | None = None,
        cache: TranslationResultCache | None = None,
        tokenizer_factory: TokenizerFactory = Tokenizer,
        yaml_loader: YamlLoader = load_yaml,
        checkpoint_loader: CheckpointLoader = torch.load,
        translate_batch_fn: TranslateBatchFn = translate_batch,
        warmup_sentences: dict[str, str] | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        if self.registry:
            any_spec = next(iter(self.registry.values()))
            root = any_spec.config_path.parents[1]
            default_tokenizer = root / "data/tokenizer/spm.model"
        else:
            default_tokenizer = None
        if tokenizer_path is None and default_tokenizer is None:
            raise ValueError("tokenizer_path is required when registry is empty.")
        self.tokenizer_path = tokenizer_path or default_tokenizer
        self.device = device or get_device()
        self.cache = cache or TranslationResultCache()
        self.tokenizer_factory = tokenizer_factory
        self.yaml_loader = yaml_loader
        self.checkpoint_loader = checkpoint_loader
        self.translate_batch_fn = translate_batch_fn
        self.warmup_sentences = warmup_sentences or WARMUP_SENTENCES
        self._tokenizer: Tokenizer | None = None
        self._tokenizer_lock = threading.Lock()
        self._models: dict[str, LoadedModel] = {}
        self._model_errors: dict[str, str] = {}
        self._preload_lock = threading.Lock()
        self._warmup_complete = False

    def get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._tokenizer_lock:
            if self._tokenizer is None:
                self._tokenizer = self.tokenizer_factory(self.tokenizer_path)
        return self._tokenizer

    def resolve_checkpoint(self, model_name: ModelName) -> Path:
        spec = self.registry[model_name]
        attempted: list[Path] = []
        for filename in spec.checkpoint_candidates:
            candidate = spec.run_dir / filename
            attempted.append(candidate)
            if candidate.exists():
                return candidate
        attempted_text = ", ".join(str(path) for path in attempted)
        raise FileNotFoundError(
            f"No checkpoint found for '{model_name}'. Tried: {attempted_text}"
        )

    def _config_for_spec(self, spec: ModelSpec) -> dict[str, Any]:
        run_config = spec.run_dir / "config.yaml"
        config_path = run_config if run_config.exists() else spec.config_path
        return self.yaml_loader(config_path)

    def _load_model(self, model_name: ModelName) -> LoadedModel:
        if model_name in self._models:
            return self._models[model_name]
        spec = self.registry[model_name]
        config = self._config_for_spec(spec)
        checkpoint_path = self.resolve_checkpoint(model_name)
        checkpoint = self.checkpoint_loader(checkpoint_path, map_location="cpu")
        model_cfg = checkpoint.get("model_cfg", config["model"])
        model = spec.build_model(model_cfg).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        eval_cfg = config.get("eval", {})
        penalties = eval_cfg.get("length_penalty", {})
        if isinstance(penalties, (int, float)):
            penalties = {"zh2vi": float(penalties), "vi2zh": float(penalties)}
        loaded = LoadedModel(
            spec=spec,
            model=model,
            checkpoint_path=checkpoint_path,
            beam_size=int(eval_cfg.get("beam_size", 4)),
            max_decode_len=int(eval_cfg.get("max_decode_len", 256)),
            length_penalty={
                "zh2vi": float(penalties.get("zh2vi", 1.0)),
                "vi2zh": float(penalties.get("vi2zh", 1.0)),
            },
        )
        self._models[model_name] = loaded
        self._model_errors.pop(model_name, None)
        return loaded

    def preload_all(self, warmup: bool = True) -> dict[str, str]:
        with self._preload_lock:
            self.get_tokenizer()
            for model_name in self.registry:
                try:
                    self._load_model(model_name)
                except Exception as exc:  # pragma: no cover - exercised in app health path
                    self._model_errors[model_name] = str(exc)
            if warmup:
                self._run_warmup()
            return dict(self._model_errors)

    def _run_warmup(self) -> None:
        for model_name, loaded in list(self._models.items()):
            for direction, sentence in self.warmup_sentences.items():
                try:
                    self.translate(
                        model=model_name,
                        direction=direction,
                        text=sentence,
                        beam_size=loaded.beam_size,
                        length_penalty=loaded.length_penalty[direction],
                        max_decode_len=min(loaded.max_decode_len, 64),
                    )
                except Exception as exc:  # pragma: no cover - best effort only
                    self._model_errors[model_name] = str(exc)
        self._warmup_complete = True

    def translate(
        self,
        *,
        model: ModelName,
        direction: Direction,
        text: str,
        beam_size: int | None = None,
        length_penalty: float | None = None,
        max_decode_len: int | None = None,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")
        loaded = self._load_model(model)
        actual_beam = beam_size or loaded.beam_size
        actual_length_penalty = (
            float(length_penalty)
            if length_penalty is not None
            else loaded.length_penalty[direction]
        )
        actual_max_decode_len = max_decode_len or loaded.max_decode_len
        source_lang, target_lang = _lang_pair(direction)
        cache_key = CacheKey(
            model=model,
            direction=direction,
            text=text,
            beam_size=actual_beam,
            length_penalty=actual_length_penalty,
            max_decode_len=actual_max_decode_len,
        )
        started = time.perf_counter()
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            return {
                "translated_text": cached_value.translated_text,
                "model": model,
                "direction": direction,
                "source_lang": cached_value.source_lang,
                "target_lang": cached_value.target_lang,
                "cached": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "model_loaded": True,
                "checkpoint": str(loaded.checkpoint_path),
            }

        tokenizer = self.get_tokenizer()
        with loaded.lock:
            translated_text = self.translate_batch_fn(
                loaded.model,
                tokenizer,
                [text],
                direction,
                max_len=actual_max_decode_len,
                beam_size=actual_beam,
                length_penalty=actual_length_penalty,
                device=self.device,
            )[0]
        cache_value = CacheValue(
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self.cache.set(cache_key, cache_value)
        return {
            "translated_text": translated_text,
            "model": model,
            "direction": direction,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "cached": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model_loaded": True,
            "checkpoint": str(loaded.checkpoint_path),
        }

    def swap_form(
        self,
        *,
        input_text: str,
        output_text: str,
        direction: Direction,
    ) -> dict[str, str]:
        next_direction = "vi2zh" if direction == "zh2vi" else "zh2vi"
        source_lang, target_lang = _lang_pair(next_direction)
        next_input = output_text if output_text.strip() else input_text
        return {
            "input_text": next_input,
            "direction": next_direction,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

    def health_snapshot(self) -> dict[str, Any]:
        statuses: dict[str, Any] = {}
        for name, spec in self.registry.items():
            loaded = self._models.get(name)
            statuses[name] = {
                "kind": spec.kind,
                "loaded": loaded is not None,
                "checkpoint": str(loaded.checkpoint_path) if loaded else None,
                "error": self._model_errors.get(name),
            }
        ready = len(self._models) == len(self.registry) and not self._model_errors
        return {
            "ready": ready,
            "device": str(self.device),
            "tokenizer_loaded": self._tokenizer is not None,
            "loaded_models": len(self._models),
            "total_models": len(self.registry),
            "warmup_complete": self._warmup_complete,
            "models": statuses,
        }
