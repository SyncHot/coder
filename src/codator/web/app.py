"""FastAPI web dashboard — model switching, GPU monitoring, context status."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse

if TYPE_CHECKING:
    from codator.core.chat_engine import ChatEngine

_engine: ChatEngine | None = None


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Simple bearer-token auth middleware. Skipped when token is empty."""

    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._token:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {self._token}":
            return await call_next(request)
        # Allow dashboard HTML without auth for browser convenience
        if request.url.path in ("/", "/static/index.html"):
            return await call_next(request)
        return JSONResponse({"error": "Unauthorized"}, status_code=401)


def create_app(engine: ChatEngine) -> FastAPI:
    global _engine
    _engine = engine

    app = FastAPI(title="codator Dashboard", version="0.1.0")

    # Auth middleware (active only when auth_token is set in config)
    auth_token = engine._settings.web.auth_token
    if auth_token:
        app.add_middleware(TokenAuthMiddleware, token=auth_token)

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
                    "ollama_tag": r.ollama_tag,
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

    @app.post("/api/chat/stream")
    async def chat_stream(request: Request):
        """SSE streaming chat endpoint."""
        import json as json_mod

        data = await request.json()
        message = data.get("message", "")
        if not message:
            return JSONResponse(
                {"error": "message required"}, status_code=400,
            )

        async def event_generator():
            try:
                async for token in _engine.chat_stream(message):
                    payload = json_mod.dumps({"token": token})
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                err = json_mod.dumps({"error": str(exc)})
                yield f"data: {err}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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

    @app.get("/api/ollama/models")
    async def ollama_models():
        """List available Ollama models."""
        from codator.infrastructure.ollama_backend import OllamaBackend
        backend = OllamaBackend(_engine._settings)
        models = await backend.list_models()
        await backend.close()
        return {"models": models}

    @app.post("/api/ollama/switch")
    async def ollama_switch(request: Request):
        """Switch to a specific Ollama model."""
        data = await request.json()
        model = data.get("model", "")
        if not model:
            return JSONResponse({"error": "model required"}, status_code=400)
        result = await _engine.switch_ollama_model(model)
        return {"status": result, "model": _engine.active_model}

    @app.post("/api/ollama/pull")
    async def ollama_pull(request: Request):
        """Pull (download) an Ollama model. Streams progress via SSE."""
        import json as json_mod

        import httpx

        data = await request.json()
        model = data.get("model", "")
        if not model:
            return JSONResponse(
                {"error": "model required"}, status_code=400,
            )

        ollama_url = _engine._settings.ollama.base_url.rstrip("/")

        async def pull_stream():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{ollama_url}/api/pull",
                        json={"name": model, "stream": True},
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                obj = json_mod.loads(line)
                                status = obj.get("status", "")
                                total = obj.get("total", 0)
                                completed = obj.get("completed", 0)
                                pct = (
                                    round(completed / total * 100)
                                    if total > 0 else 0
                                )
                                payload = json_mod.dumps({
                                    "status": status,
                                    "total": total,
                                    "completed": completed,
                                    "percent": pct,
                                })
                                yield f"data: {payload}\n\n"
                            except json_mod.JSONDecodeError:
                                pass
                yield "data: [DONE]\n\n"
            except Exception as exc:
                err = json_mod.dumps({"error": str(exc)})
                yield f"data: {err}\n\n"

        return StreamingResponse(
            pull_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/api/ollama/model")
    async def ollama_delete(request: Request):
        """Delete an Ollama model."""
        import httpx

        data = await request.json()
        model = data.get("model", "")
        if not model:
            return JSONResponse(
                {"error": "model required"}, status_code=400,
            )

        ollama_url = _engine._settings.ollama.base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.delete(
                    f"{ollama_url}/api/delete",
                    json={"name": model},
                )
                if resp.status_code == 200:
                    return {"status": f"Deleted {model}"}
                return JSONResponse(
                    {"error": resp.text}, status_code=resp.status_code,
                )
        except Exception as exc:
            return JSONResponse(
                {"error": str(exc)}, status_code=500,
            )

    @app.get("/api/gpu/status")
    async def gpu_status():
        """Live GPU + Ollama stats via ROCm monitoring."""
        from codator.infrastructure.gpu_monitor import GPUMonitor
        monitor = GPUMonitor()
        return await monitor.get_full_status()

    @app.get("/api/model-selector/info")
    async def model_selector_info():
        """Current model selector state."""
        if _engine and _engine.model_selector:
            return {
                "available": True,
                "tiers": {
                    "fast": _engine.model_selector.FAST_MODELS,
                    "medium": _engine.model_selector.MEDIUM_MODELS,
                    "complex": _engine.model_selector.COMPLEX_MODELS,
                },
                "active_model": _engine.active_model,
            }
        return {"available": False}

    @app.post("/api/agent/run")
    async def agent_run(request: Request):
        """Run Plan-Act-Verify agent cycle."""
        from codator.core.agent_loop import PlanActVerifyAgent

        data = await request.json()
        task = data.get("task", "")
        if not task:
            return JSONResponse({"error": "task required"}, status_code=400)

        ollama_cfg = _engine._settings.ollama
        agent = PlanActVerifyAgent(
            ollama_base_url=ollama_cfg.base_url,
            model=_engine.active_model or ollama_cfg.model,
            project_root=_engine._project_root,
        )

        result = await agent.run(task)
        return {
            "task": result.plan.task,
            "reasoning": result.plan.reasoning,
            "steps": [
                {"action": s.step.action, "target": s.step.target,
                 "success": s.success, "output": s.output[:500], "error": s.error}
                for s in result.actions
            ],
            "verification": {
                "success": result.verification.success,
                "errors": result.verification.errors[:10],
                "warnings": result.verification.warnings[:10],
            },
            "heal_iterations": result.heal_iterations,
            "final_success": result.final_success,
        }

    @app.get("/api/context-index/stats")
    async def context_index_stats():
        """Contextual index statistics."""
        if _engine and _engine.contextual_index:
            idx = _engine.contextual_index
            return {
                "available": True,
                "chunk_count": len(idx._chunks),
            }
        return {"available": False}

    @app.post("/api/context-index/search")
    async def context_index_search(request: Request):
        """Search the contextual index."""
        data = await request.json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        if not query:
            return JSONResponse({"error": "query required"}, status_code=400)
        if _engine and _engine.contextual_index:
            chunks = _engine.contextual_index.search(query, top_k=top_k)
            return {
                "results": [
                    {
                        "file_path": c.file_path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "parent_context": c.parent_context,
                        "chunk_type": c.chunk_type,
                        "content": c.content[:500],
                    }
                    for c in chunks
                ],
            }
        return {"results": []}
