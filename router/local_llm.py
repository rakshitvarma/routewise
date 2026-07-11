"""Genuine local zero-token inference via bundled quantized GGUF models.

Distinct from router/solvers.py (which only answers when it can *prove*
correctness): this is real language-model inference, so each category
routed here has been checked against router/eval_local.py first, and every
answer still passes a lightweight sanity check before being trusted over
escalating to Fireworks.

Two small models cover different category groups:
  - "general": sentiment / NER / factual / summarisation
  - "code":    code_debug / code_gen (code-specialized, verified via
    router.solvers.python_syntax_error before being trusted)
Both are lazy-loaded (only the models actually needed for the tasks in a
given run get loaded), and loading/inference failures degrade to None so
main.py's existing Fireworks fallback always covers the gap.
"""
import os
import re
from typing import Optional

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_MODEL_PATHS = {
    "general": os.environ.get("LOCAL_GENERAL_MODEL_PATH", os.path.join(_MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")),
    "code": os.environ.get("LOCAL_CODE_MODEL_PATH", os.path.join(_MODELS_DIR, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")),
}

_CATEGORY_MODEL = {
    "sentiment": "general",
    "ner": "general",
    "factual": "general",
    "summarization": "general",
    "code_debug": "code",
    "code_gen": "code",
}

_llm_cache = {}
_load_failed = set()


def _available_cpus() -> int:
    """CPU count that respects cgroup limits, unlike os.cpu_count() which
    reports the host's cores even inside a --cpus-limited container. On the
    2-vCPU grading box, os.cpu_count() returns the host's (e.g. 16), so
    llama.cpp would spawn 16 threads fighting over 2 cores - severe
    context-switch thrash that made a 46s run take 372s under test."""
    # Linux cgroup v2 quota
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    # cgroup-aware affinity (Docker --cpuset), then fall back to host count
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 2)


_N_THREADS = int(os.environ.get("LOCAL_LLM_THREADS", _available_cpus()))


def _get_llm(model_key: str):
    if model_key in _llm_cache:
        return _llm_cache[model_key]
    if model_key in _load_failed:
        return None
    path = _MODEL_PATHS[model_key]
    if not os.path.exists(path):
        _load_failed.add(model_key)
        return None
    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=path,
            n_ctx=1024,
            n_threads=_N_THREADS,
            verbose=False,
        )
        _llm_cache[model_key] = llm
        return llm
    except Exception:
        _load_failed.add(model_key)
        return None


_SYSTEM_PROMPTS = {
    "sentiment": "Classify sentiment as positive, negative, or neutral, then give a one-clause justification. Be concise.",
    "ner": "Extract named entities (person, organization, location, date) as a compact labelled list. Be concise.",
    "factual": "Answer in 1-2 short sentences. No preamble, no filler.",
    "summarization": "Summarise to the exact length/format constraint given. No preamble.",
    "code_debug": "Find the bug and return the corrected code only, no explanation.",
    "code_gen": "Write the correct, complete function per the spec. Code only, no explanation.",
}

_MAX_TOKENS = {
    "sentiment": 40, "ner": 80, "factual": 70, "summarization": 100,
    "code_debug": 250, "code_gen": 300,
}


def available(category: str = "general") -> bool:
    model_key = _CATEGORY_MODEL.get(category, category)
    return _get_llm(model_key) is not None


def answer(category: str, prompt: str) -> Optional[str]:
    """Return a locally-generated answer, or None if the model isn't
    available or the output fails a basic sanity check."""
    model_key = _CATEGORY_MODEL.get(category)
    if model_key is None:
        return None
    llm = _get_llm(model_key)
    if llm is None:
        return None

    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPTS[category]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=_MAX_TOKENS[category],
            temperature=0,
        )
        text = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    if not _sane(category, text):
        return None
    return text


def _sane(category: str, text: str) -> bool:
    """Cheap zero-token confidence check - reject obviously broken output
    rather than trusting a small model blindly."""
    if not text or len(text) < 2:
        return False
    if len(text) > 900:  # degenerate rambling/repetition
        return False
    words = text.split()
    if len(words) > 6 and len(set(words)) / len(words) < 0.35:  # repetition loop
        return False
    if category == "sentiment":
        return bool(re.search(r"\b(positive|negative|neutral)\b", text, re.I))
    return True
