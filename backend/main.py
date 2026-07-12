"""Fairwind API — the same routing logic as main.py (the Track 1 submission
entrypoint), exposed as a proper HTTP API for a decoupled frontend.

Reuses router/ unchanged: classifier, solvers, local_llm, fireworks_client.
"""
import os
import sys
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.classifier import classify
from router.solvers import try_solve_math, strip_code_fence
from router.fireworks_client import FireworksClient
from router import local_llm

LOCAL_LLM_CATEGORIES = {"sentiment", "ner", "factual", "summarization", "code_debug", "code_gen"}

app = FastAPI(title="Fairwind API")

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _preload():
    # Warm up local models once at container startup, not on the first
    # request - matches the intent of the webapp's st.cache_resource
    # preload, just via FastAPI's own startup hook.
    local_llm.preload()


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    category: str
    source: str  # "local (deterministic)" | "qwen3" | a Fireworks model id | "error" | "n/a"
    tokens: int
    elapsed: float
    answer: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "local_general_available": local_llm.available("factual"),
        "local_code_available": local_llm.available("code_gen"),
    }


@app.get("/api/models")
def models():
    has_creds = all(os.environ.get(k) for k in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"))
    fireworks_models = []
    if has_creds:
        try:
            client = FireworksClient()
            fireworks_models = client.allowed_models
        except Exception:
            pass
    return {
        "fireworks_models": fireworks_models,
        "local_models": [
            {"name": "Qwen3-4B-Instruct-2507", "categories": ["factual", "sentiment", "ner", "summarization", "code_debug", "code_gen"]},
        ],
        "has_fireworks_creds": has_creds,
    }


@app.post("/api/route", response_model=RouteResponse)
def route(req: RouteRequest):
    prompt = req.prompt.strip()
    started = time.time()
    category = classify(prompt)

    answer, source, tokens = None, None, 0

    if category == "math":
        math_answer = try_solve_math(prompt)
        if math_answer is not None:
            answer, source = math_answer, "local (deterministic)"
        # Word problems deliberately stay on Fireworks - see main.py for why.

    if answer is None and category in LOCAL_LLM_CATEGORIES:
        local_answer = local_llm.answer_confident(category, prompt)
        if local_answer is not None:
            if category in ("code_debug", "code_gen"):
                local_answer = strip_code_fence(local_answer)
            answer, source = local_answer, "qwen3"

    has_creds = all(os.environ.get(k) for k in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"))
    if answer is None and has_creds:
        try:
            client = FireworksClient()
            if category == "logic":
                source = client.pick_model("logic")
                answer = client._answer_logic(source, prompt)
            else:
                result = client.answer_all({category: [("live", prompt)]})
                answer = result.get("live", "")
                source = client.pick_model(category)
            tokens = client.total_tokens
        except Exception as exc:
            answer, source = f"(Fireworks call failed: {exc})", "error"
    elif answer is None:
        answer, source = "(no live Fireworks credentials configured)", "n/a"

    return RouteResponse(
        category=category, source=source or "n/a", tokens=tokens,
        elapsed=round(time.time() - started, 2), answer=answer or "",
    )
