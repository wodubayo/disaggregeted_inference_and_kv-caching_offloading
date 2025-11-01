import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx

PREFILL_URL = os.getenv("PREFILL_URL", "http://prefill:8010/v1")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # create one shared client for the whole app lifetime
    app.state.client = httpx.AsyncClient(timeout=120)
    try:
        yield
    finally:
        await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "prefill_url": PREFILL_URL, "model": MODEL_ID}

# ---- helpers ----
async def _json_or_raise(resp: httpx.Response):
    try:
        data = resp.json()
    except Exception:
        # bubble up backend raw text if it wasn’t JSON
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return JSONResponse(data, status_code=resp.status_code)

async def _openai_completions(payload: dict):
    r = await app.state.client.post(f"{PREFILL_URL}/completions", json=payload)
    return await _json_or_raise(r)

async def _proxy_get(path: str):
    r = await app.state.client.get(f"{PREFILL_URL}{path}")
    return await _json_or_raise(r)

# ---- OpenAI-compatible: list models ----
@app.get("/v1/models")
async def v1_models():
    # PREFILL_URL already ends in /v1, so /models → /v1/models
    return await _proxy_get("/models")

# convenience endpoint that takes just a prompt
@app.post("/generate")
async def generate(req: Request):
    """
    Accepts: {"prompt": "...", "max_tokens": 64, "temperature": 0.7}
    Proxies to backend /v1/completions using MODEL_ID.
    """
    body = await req.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")

    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": body.get("max_tokens", 128),
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "repetition_penalty": body.get("repetition_penalty"),
        "stop": body.get("stop"),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return await _openai_completions(payload)

# ---- OpenAI-compatible pass-throughs ----
@app.post("/v1/completions")
async def v1_completions(body: dict):
    return await _openai_completions(body)

@app.post("/v1/chat/completions")
async def v1_chat(body: dict):
    """
    Try real chat passthrough first (if backend ever exposes it).
    If backend returns 404, fall back to concatenating messages and calling /completions.
    """
    # direct passthrough attempt
    r = await app.state.client.post(f"{PREFILL_URL}/chat/completions", json=body)
    if r.status_code != 404:
        return await _json_or_raise(r)

    # fallback: translate chat → prompt
    msgs = body.get("messages") or []
    if not msgs:
        raise HTTPException(status_code=400, detail="Missing 'messages'.")

    prompt = "\n".join(m.get("content", "") for m in msgs)
    payload = {
        "model": body.get("model", MODEL_ID),
        "prompt": prompt,
        "max_tokens": body.get("max_tokens", 128),
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "repetition_penalty": body.get("repetition_penalty"),
        "stop": body.get("stop"),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return await _openai_completions(payload)
