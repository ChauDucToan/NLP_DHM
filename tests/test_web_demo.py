"""Tests for the FastAPI multi-model translation demo."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_demo.service import ModelManager, ModelSpec, TranslationResultCache


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = type("Cfg", (), {"max_src_len": 64})()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.loaded_state: dict[str, Any] | None = None

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):  # type: ignore[override]
        self.loaded_state = state_dict
        return None


class DummyTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _config(output_dir: str) -> dict[str, Any]:
    return {
        "model": {"vocab_size": 16},
        "train": {"output_dir": output_dir},
        "eval": {
            "beam_size": 4,
            "length_penalty": {"zh2vi": 1.0, "vi2zh": 0.9},
            "max_decode_len": 128,
        },
    }


def _build_spec(root: Path, name: str) -> ModelSpec:
    run_dir = root / "runs" / name
    config_path = root / "configs" / f"{name}.yaml"
    return ModelSpec(
        name=name,
        kind=name,
        config_path=config_path,
        run_dir=run_dir,
        build_model=lambda _cfg: DummyModel(),
    )


def test_model_manager_resolves_checkpoint_by_priority() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = _build_spec(root, "mamba")
        _write_yaml(spec.config_path, _config(str(spec.run_dir)))
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        (spec.run_dir / "best.pt").write_bytes(b"")
        (spec.run_dir / "latest.pt").write_bytes(b"")

        manager = ModelManager(
            registry={"mamba": spec},
            tokenizer_path=root / "data/tokenizer/spm.model",
            device=torch.device("cpu"),
            tokenizer_factory=DummyTokenizer,
        )

        resolved = manager.resolve_checkpoint("mamba")
        assert resolved.name == "best.pt"


def test_model_manager_loads_tokenizer_once() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        calls: list[str] = []

        def tokenizer_factory(model_path: str | Path) -> DummyTokenizer:
            calls.append(str(model_path))
            return DummyTokenizer(model_path)

        manager = ModelManager(
            registry={"mamba": _build_spec(root, "mamba")},
            tokenizer_path=root / "data/tokenizer/spm.model",
            device=torch.device("cpu"),
            tokenizer_factory=tokenizer_factory,
        )

        first = manager.get_tokenizer()
        second = manager.get_tokenizer()

        assert first is second
        assert calls == [str(root / "data/tokenizer/spm.model")]


def test_model_manager_preloads_all_models() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        registry = {}
        for name, checkpoint_name in {
            "mamba": "latest.pt",
            "hybrid": "best_ema.pt",
            "transformer": "final.pt",
        }.items():
            spec = _build_spec(root, name)
            _write_yaml(spec.config_path, _config(str(spec.run_dir)))
            spec.run_dir.mkdir(parents=True, exist_ok=True)
            (spec.run_dir / checkpoint_name).write_bytes(b"")
            registry[name] = spec

        manager = ModelManager(
            registry=registry,
            tokenizer_path=root / "data/tokenizer/spm.model",
            device=torch.device("cpu"),
            tokenizer_factory=DummyTokenizer,
            checkpoint_loader=lambda *_args, **_kwargs: {
                "model": {"ok": True},
                "model_cfg": {"vocab_size": 16},
            },
            translate_batch_fn=lambda *_args, **_kwargs: ["warmup"],
        )

        errors = manager.preload_all(warmup=False)
        health = manager.health_snapshot()

        assert errors == {}
        assert health["ready"] is True
        assert health["loaded_models"] == 3


def test_translation_cache_hits_same_request_only() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mamba = _build_spec(root, "mamba")
        hybrid = _build_spec(root, "hybrid")
        for spec in (mamba, hybrid):
            _write_yaml(spec.config_path, _config(str(spec.run_dir)))
            spec.run_dir.mkdir(parents=True, exist_ok=True)
            (spec.run_dir / "latest.pt").write_bytes(b"")

        translate_calls: list[tuple[str, str, str]] = []

        def fake_translate(
            model: DummyModel,
            _tokenizer: DummyTokenizer,
            sentences: list[str],
            direction: str,
            **_kwargs,
        ) -> list[str]:
            translated = f"{direction}:{sentences[0]}:{len(translate_calls)}"
            translate_calls.append((str(model.loaded_state), direction, sentences[0]))
            return [translated]

        manager = ModelManager(
            registry={"mamba": mamba, "hybrid": hybrid},
            tokenizer_path=root / "data/tokenizer/spm.model",
            device=torch.device("cpu"),
            tokenizer_factory=DummyTokenizer,
            checkpoint_loader=lambda *_args, **_kwargs: {
                "model": {"ok": True},
                "model_cfg": {"vocab_size": 16},
            },
            translate_batch_fn=fake_translate,
            cache=TranslationResultCache(max_entries=16, ttl_seconds=60),
        )

        first = manager.translate(model="mamba", direction="vi2zh", text="xin chao")
        second = manager.translate(model="mamba", direction="vi2zh", text="xin chao")
        third = manager.translate(model="mamba", direction="zh2vi", text="xin chao")
        fourth = manager.translate(model="hybrid", direction="vi2zh", text="xin chao")

        assert first["cached"] is False
        assert second["cached"] is True
        assert third["cached"] is False
        assert fourth["cached"] is False
        assert len(translate_calls) == 3


def test_swap_prefers_existing_output_text() -> None:
    manager = ModelManager(
        registry={},
        tokenizer_path=Path("unused"),
        device=torch.device("cpu"),
        tokenizer_factory=DummyTokenizer,
    )

    swapped = manager.swap_form(
        input_text="xin chao",
        output_text="你好",
        direction="vi2zh",
    )

    assert swapped["input_text"] == "你好"
    assert swapped["direction"] == "zh2vi"


def test_api_translate_and_swap_with_stub_manager() -> None:
    try:
        from fastapi.testclient import TestClient
    except ModuleNotFoundError:
        return

    from web_demo.server import create_app

    class StubManager:
        def __init__(self) -> None:
            self.preload_called = False

        def preload_all(self, warmup: bool = True) -> dict[str, str]:
            self.preload_called = warmup
            return {}

        def health_snapshot(self) -> dict[str, Any]:
            return {
                "ready": True,
                "loaded_models": 3,
                "total_models": 3,
                "device": "cpu",
            }

        def translate(self, *, model: str, direction: str, text: str) -> dict[str, Any]:
            if not text.strip():
                raise ValueError("Input text must not be empty.")
            return {
                "translated_text": f"{model}:{direction}:{text}",
                "model": model,
                "direction": direction,
                "source_lang": "vi" if direction == "vi2zh" else "zh",
                "target_lang": "zh" if direction == "vi2zh" else "vi",
                "cached": False,
                "latency_ms": 1.25,
                "model_loaded": True,
                "checkpoint": "/tmp/fake.pt",
            }

        def swap_form(self, *, input_text: str, output_text: str, direction: str) -> dict[str, str]:
            return {
                "input_text": output_text or input_text,
                "direction": "zh2vi" if direction == "vi2zh" else "vi2zh",
                "source_lang": "zh",
                "target_lang": "vi",
            }

    with TestClient(create_app(StubManager())) as client:
        health = client.get("/api/health")
        translated = client.post(
            "/api/translate",
            json={"text": "xin chào thế giới", "model": "mamba", "direction": "vi2zh"},
        )
        swapped = client.post(
            "/api/swap",
            json={"input_text": "xin chào", "output_text": "你好", "direction": "vi2zh"},
        )
        bad = client.post(
            "/api/translate",
            json={"text": "   ", "model": "mamba", "direction": "vi2zh"},
        )

        assert health.status_code == 200
        assert translated.status_code == 200
        assert translated.json()["translated_text"]
        assert swapped.status_code == 200
        assert swapped.json()["input_text"] == "你好"
        assert swapped.json()["direction"] == "zh2vi"
        assert bad.status_code == 400
