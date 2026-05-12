"""FastAPI server for the multi-model zh↔vi translation demo."""

from __future__ import annotations

from contextlib import asynccontextmanager
from html import escape
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .service import ModelManager


HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Demo Dịch 3 Mô Hình</title>
  <style>
    :root {
      --bg: #f3efe6;
      --card: rgba(255, 252, 246, 0.92);
      --ink: #18211a;
      --muted: #5e695b;
      --line: rgba(24, 33, 26, 0.12);
      --accent: #0c7c59;
      --accent-soft: #dff5ea;
      --warn: #9d3d16;
      --shadow: 0 24px 70px rgba(48, 56, 40, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(12, 124, 89, 0.18), transparent 32%),
        radial-gradient(circle at right, rgba(219, 141, 45, 0.18), transparent 28%),
        linear-gradient(135deg, #f7f4ea, #efe6d5 48%, #ece5dc);
      min-height: 100vh;
      padding: 32px 18px;
    }
    .shell {
      max-width: 980px;
      margin: 0 auto;
      background: var(--card);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.7);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .hero {
      padding: 28px 28px 14px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(120deg, rgba(12, 124, 89, 0.10), rgba(255, 255, 255, 0)),
        linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0));
    }
    h1 {
      margin: 0 0 6px;
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      padding: 20px 28px 28px;
    }
    .panel {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .toolbar {
      grid-column: 1 / -1;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 16px 28px 0;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }
    label {
      display: block;
      font-size: 0.92rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    select, textarea, button {
      font: inherit;
    }
    select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
    }
    textarea {
      min-height: 220px;
      resize: vertical;
      line-height: 1.55;
    }
    .direction {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px;
      border-radius: 999px;
      background: rgba(12, 124, 89, 0.08);
    }
    .direction button {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      background: transparent;
      color: var(--muted);
    }
    .direction button.active {
      background: var(--ink);
      color: #fffaf2;
    }
    .actions {
      display: flex;
      gap: 12px;
      justify-content: flex-end;
      padding: 0 28px 28px;
    }
    .primary, .secondary {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease;
    }
    .primary:hover, .secondary:hover { transform: translateY(-1px); }
    .primary {
      background: var(--accent);
      color: white;
      min-width: 110px;
    }
    .secondary {
      background: var(--accent-soft);
      color: var(--accent);
    }
    .status {
      padding: 0 28px 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .status span {
      padding: 7px 11px;
      border-radius: 999px;
      background: rgba(24, 33, 26, 0.06);
    }
    .status .error {
      background: rgba(157, 61, 22, 0.12);
      color: var(--warn);
    }
    @media (max-width: 800px) {
      .grid { grid-template-columns: 1fr; padding: 18px; }
      .toolbar, .actions, .status, .hero { padding-left: 18px; padding-right: 18px; }
      .actions { justify-content: stretch; }
      .primary, .secondary { flex: 1; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Demo Dịch 3 Mô Hình</h1>
      <p class="subtitle">FastAPI demo cho Bi-Mamba, Hybrid và Transformer. Model được preload và giữ nóng trong cùng tiến trình.</p>
    </section>

    <section class="toolbar">
      <div class="controls">
        <div>
          <label for="model">Mô hình</label>
          <select id="model">
            <option value="mamba">mamba</option>
            <option value="hybrid">hybrid</option>
            <option value="transformer">transformer</option>
          </select>
        </div>
      </div>
      <div class="direction" aria-label="Hướng dịch">
        <button type="button" id="dir-zh2vi" class="active">Trung → Việt</button>
        <button type="button" id="dir-vi2zh">Việt → Trung</button>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <label for="inputText">Văn bản nguồn</label>
        <textarea id="inputText" placeholder="Nhập câu tiếng Việt hoặc tiếng Trung..."></textarea>
      </div>
      <div class="panel">
        <label for="outputText">Bản dịch</label>
        <textarea id="outputText" readonly placeholder="Kết quả sẽ xuất hiện ở đây"></textarea>
      </div>
    </section>

    <div class="actions">
      <button type="button" class="secondary" id="swapBtn">Hoán đổi</button>
      <button type="button" class="primary" id="translateBtn">Dịch</button>
    </div>

    <div class="status" id="statusBar">
      <span id="healthStatus">Đang kiểm tra preload...</span>
      <span id="cacheStatus">cache: -</span>
      <span id="modelStatus">model loaded: -</span>
      <span id="latencyStatus">latency: -</span>
    </div>
  </main>

  <script>
    const inputText = document.getElementById("inputText");
    const outputText = document.getElementById("outputText");
    const modelSelect = document.getElementById("model");
    const healthStatus = document.getElementById("healthStatus");
    const cacheStatus = document.getElementById("cacheStatus");
    const modelStatus = document.getElementById("modelStatus");
    const latencyStatus = document.getElementById("latencyStatus");
    const dirZh2Vi = document.getElementById("dir-zh2vi");
    const dirVi2Zh = document.getElementById("dir-vi2zh");

    let direction = "zh2vi";

    function setDirection(nextDirection) {
      direction = nextDirection;
      dirZh2Vi.classList.toggle("active", nextDirection === "zh2vi");
      dirVi2Zh.classList.toggle("active", nextDirection === "vi2zh");
    }

    async function refreshHealth() {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.ready) {
        healthStatus.textContent = `preload: sẵn sàng (${data.loaded_models}/${data.total_models}) trên ${data.device}`;
        healthStatus.classList.remove("error");
      } else {
        healthStatus.textContent = `preload: chưa sẵn sàng (${data.loaded_models}/${data.total_models})`;
        healthStatus.classList.add("error");
      }
    }

    async function doTranslate() {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputText.value,
          model: modelSelect.value,
          direction: direction,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Dịch thất bại.");
      }
      outputText.value = data.translated_text;
      cacheStatus.textContent = `cache: ${data.cached ? "hit" : "miss"}`;
      modelStatus.textContent = `model loaded: ${data.model_loaded ? "yes" : "no"}`;
      latencyStatus.textContent = `latency: ${data.latency_ms} ms`;
    }

    async function doSwap() {
      const hadOutput = outputText.value.trim().length > 0;
      const res = await fetch("/api/swap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_text: inputText.value,
          output_text: outputText.value,
          direction: direction,
        }),
      });
      const data = await res.json();
      inputText.value = data.input_text;
      outputText.value = "";
      setDirection(data.direction);
      if (hadOutput && data.input_text.trim()) {
        await doTranslate();
      }
    }

    dirZh2Vi.addEventListener("click", () => setDirection("zh2vi"));
    dirVi2Zh.addEventListener("click", () => setDirection("vi2zh"));

    document.getElementById("translateBtn").addEventListener("click", async () => {
      try {
        await doTranslate();
      } catch (error) {
        latencyStatus.textContent = "latency: -";
        cacheStatus.textContent = "cache: -";
        modelStatus.textContent = "model loaded: -";
        healthStatus.textContent = error.message;
        healthStatus.classList.add("error");
      }
    });

    document.getElementById("swapBtn").addEventListener("click", async () => {
      try {
        await doSwap();
      } catch (error) {
        healthStatus.textContent = error.message;
        healthStatus.classList.add("error");
      }
    });

    refreshHealth().catch((error) => {
      healthStatus.textContent = error.message;
      healthStatus.classList.add("error");
    });
  </script>
</body>
</html>
"""


class TranslateRequest(BaseModel):
    text: str
    model: Literal["mamba", "hybrid", "transformer"]
    direction: Literal["zh2vi", "vi2zh"]


class TranslateResponse(BaseModel):
    translated_text: str
    model: str
    direction: str
    source_lang: Literal["vi", "zh"]
    target_lang: Literal["vi", "zh"]
    cached: bool
    latency_ms: float
    model_loaded: bool
    checkpoint: str


class SwapRequest(BaseModel):
    input_text: str
    output_text: str
    direction: Literal["zh2vi", "vi2zh"]


class SwapResponse(BaseModel):
    input_text: str
    direction: Literal["zh2vi", "vi2zh"]
    source_lang: Literal["vi", "zh"]
    target_lang: Literal["vi", "zh"]


def create_app(manager: ModelManager | None = None) -> FastAPI:
    runtime_manager = manager or ModelManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = runtime_manager
        runtime_manager.preload_all(warmup=True)
        yield

    app = FastAPI(title="3-model zh-vi translation demo", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return runtime_manager.health_snapshot()

    @app.post("/api/translate", response_model=TranslateResponse)
    async def translate(request: TranslateRequest) -> dict[str, object]:
        try:
            return runtime_manager.translate(
                model=request.model,
                direction=request.direction,
                text=request.text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - surfaced for manual demo
            message = escape(str(exc)) or "Unexpected translation error."
            raise HTTPException(status_code=500, detail=message) from exc

    @app.post("/api/swap", response_model=SwapResponse)
    async def swap(request: SwapRequest) -> dict[str, str]:
        return runtime_manager.swap_form(
            input_text=request.input_text,
            output_text=request.output_text,
            direction=request.direction,
        )

    return app


app = create_app()
