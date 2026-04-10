"""FastAPI web dashboard — model switching, GPU monitoring, context status."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from codator.core.chat_engine import ChatEngine

_engine: ChatEngine | None = None


def create_app(engine: ChatEngine) -> FastAPI:
    global _engine
    _engine = engine

    app = FastAPI(title="codator Dashboard", version="0.1.0")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _register_routes(app)
    return app


def _register_routes(app: FastAPI):

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return (Path(__file__).parent / "static" / "index.html").read_text()

    @app.get("/api/status")
    async def status():
        from codator.infrastructure.hardware import check_hardware
        hw, recs = check_hardware()
        return {
            "model": _engine.active_model if _engine else "none",
            "context": _engine.context_status if _engine else {},
            "hardware": {
                "gpu": hw.gpu_name,
                "vram_total_mb": hw.vram_total_mb,
                "vram_used_mb": hw.vram_used_mb,
                "vram_free_mb": hw.vram_free_mb,
                "ram_total_mb": hw.ram_total_mb,
                "ram_free_mb": hw.ram_free_mb,
                "rocm_version": hw.rocm_version,
            },
            "recommendations": [
                {
                    "name": r.name, "params": r.params, "quant": r.quant,
                    "mode": r.mode.value, "estimated_vram_mb": r.estimated_vram_mb,
                }
                for r in recs
            ],
        }

    @app.post("/api/model")
    async def switch_model(request: Request):
        data = await request.json()
        path = data.get("model_path", "")
        if not path:
            return JSONResponse({"error": "model_path required"}, status_code=400)
        result = await _engine.switch_model(path)
        return {"status": result, "model": _engine.active_model}

    @app.post("/api/provider")
    async def switch_provider(request: Request):
        data = await request.json()
        provider = data.get("provider", "")
        result = await _engine.switch_to_api(provider)
        return {"status": result, "model": _engine.active_model}

    @app.post("/api/chat")
    async def chat(request: Request):
        data = await request.json()
        message = data.get("message", "")
        if not message:
            return JSONResponse({"error": "message required"}, status_code=400)
        result = await _engine.chat(message)
        return {
            "response": result.text,
            "tokens_generated": result.tokens_generated,
            "time_seconds": round(result.time_seconds, 2),
            "model": result.model_name,
            "context": _engine.context_status,
        }

    @app.post("/api/index")
    async def reindex():
        result = await _engine.refresh_project_context()
        return {"status": result}

    @app.get("/api/gpu-temp")
    async def gpu_temperature():
        """Read GPU temperature via rocm-smi."""
        import subprocess
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return data
        except Exception:
            pass
        return {"temperature": "unavailable"}
