"""ASGI entrypoint for the translation web demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from web_demo.server import app

__all__ = ["app"]
