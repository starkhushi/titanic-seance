from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ghost, oracle

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Titanic Seance")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Profile(BaseModel):
    pclass: int = Field(3, ge=1, le=3)
    female: bool = False
    age: int = Field(30, ge=0, le=80)
    sibsp: int = Field(0, ge=0, le=4)
    parch: int = Field(0, ge=0, le=3)
    port: str = Field("S", pattern="^[SCQ]$")


class Speak(BaseModel):
    profile: Profile
    lang: str = Field("en", pattern="^(en|hi)$")
    question: str | None = Field(None, max_length=300)
    history: list[dict] = Field(default_factory=list, max_length=12)


@app.api_route("/", methods=["GET", "HEAD"])
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz/llm")
def healthz_llm() -> dict:
    """One tiny Gemma call, so a bad key or quota shows up without reading server logs."""
    key = os.environ.get("GEMINI_API_KEY", "")
    info = dict(key_ends=key[-4:] if len(key) > 12 else None)     # enough to tell keys apart, not to use one
    try:
        text = "".join(ghost.stream("Reply with one word.", "Say: ready", max_tokens=20, temperature=0.0))
        return dict(info, ok=True, reply=text.strip()[:40])
    except ghost.LLMError as e:
        return dict(info, ok=False, error=str(e)[:300])


@app.get("/healthz")
def healthz() -> dict:
    g = oracle.grid()
    return dict(ok=True, model=g["model"], profiles=len(g["probs"]), llm_backend=ghost.BACKEND,
                llm_model=ghost.GEMINI_MODEL if ghost.BACKEND == "gemini" else ghost.LLM_MODEL,
                llm_ready=bool(os.environ.get("GEMINI_API_KEY")) if ghost.BACKEND == "gemini" else True)


@app.post("/api/reading")
def api_reading(p: Profile) -> dict:
    return oracle.reading(p.model_dump())


@app.post("/api/speak")
def api_speak(body: Speak) -> StreamingResponse:
    r = oracle.reading(body.profile.model_dump())
    question = (body.question or "").strip() or None
    return StreamingResponse(ghost.stream_reading(r, body.lang, body.history, question),
                             media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
