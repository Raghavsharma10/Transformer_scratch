"""
server/main.py
FastAPI HTTP server exposing the Python code suggestion endpoint.

Usage:
    pip install fastapi uvicorn
    python -m server.main \
        --checkpoint checkpoints/best.pt \
        --tokenizer  tokenizer/python_bpe.model \
        --host 127.0.0.1 --port 8000

Endpoint:
    POST /suggest
    Body : { "prefix": "def fib(n):\n    ", "k": 3, "max_tokens": 64 }
    Response: { "suggestions": ["...", "...", "..."], "latency_ms": 42.1 }

Health:
    GET /health → { "status": "ok", "model_params": 10123456 }
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

# FastAPI is imported inside the handler to allow the file to be parsed
# even when it's not installed (e.g., during unit testing of other modules).
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from inference.suggest import CodeSuggester, Suggestion


# ──────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────

if _FASTAPI_AVAILABLE:
    class SuggestRequest(BaseModel):
        prefix:     str              = Field(...,  description="Code typed so far")
        k:          int              = Field(3,    ge=1, le=5, description="Number of suggestions")
        max_tokens: int              = Field(64,   ge=8, le=256)
        deduplicate: bool            = Field(True, description="Remove near-duplicate completions")

    class SuggestResponse(BaseModel):
        suggestions: List[str]
        scores:      List[float]
        valid:       List[bool]
        latency_ms:  float

    class HealthResponse(BaseModel):
        status:       str
        model_params: int
        device:       str


# ──────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────

def create_app(suggester: CodeSuggester) -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(
        title="Python Code Suggester",
        description="Top-k Python code block suggestions powered by a custom transformer",
        version="1.0.0",
    )

    # Allow VS Code extension (localhost) to call this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_params=suggester.model.param_count(),
            device=str(suggester.device),
        )

    @app.post("/suggest", response_model=SuggestResponse)
    async def suggest(req: SuggestRequest) -> SuggestResponse:
        if not req.prefix.strip():
            raise HTTPException(status_code=400, detail="prefix must not be empty")

        t0 = time.perf_counter()
        results: List[Suggestion] = suggester.suggest(
            prefix=req.prefix,
            k=req.k,
            max_new_tokens=req.max_tokens,
            deduplicate=req.deduplicate,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        return SuggestResponse(
            suggestions=[r.completion for r in results],
            scores=[round(r.score, 4) for r in results],
            valid=[r.is_valid_py for r in results],
            latency_ms=round(total_ms, 1),
        )

    return app


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python code suggestion server")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer",  required=True,
                        help="Path to tokenizer model (.model)")
    parser.add_argument("--device",     default=None,
                        help="'cuda', 'cpu', or 'mps' (auto-detected if not set)")
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--reload",     action="store_true",
                        help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    suggester = CodeSuggester(args.checkpoint, args.tokenizer, device=args.device)
    app = create_app(suggester)

    print(f"\nServer starting at http://{args.host}:{args.port}")
    print(f"Docs at http://{args.host}:{args.port}/docs\n")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
