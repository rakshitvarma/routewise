"""Genuine local zero-token inference via a bundled quantized GGUF model.

Distinct from router/solvers.py (which only answers when it can *prove*
correctness): this is real language-model inference, so each category
routed here has been checked against router/eval_local.py first, and every
answer still passes a confidence gate (see answer_confident) before being
trusted over escalating to Fireworks.

One consolidated model (Qwen3-4B-Instruct-2507) covers every locally-
eligible category - sentiment / NER / factual / summarisation /
code_debug / code_gen. Earlier versions used two separate 1.5B models
(general + code-specialized), but Qwen3-4B handles code well enough on
its own to drop the dedicated coder model, at a similar combined memory
footprint to a single model rather than two loaded models. Lazy-loaded
(only loaded if a task in a given run actually needs it), and loading/
inference failures degrade to None so main.py's Fireworks fallback always
covers the gap.
"""
import os
import re
from typing import Optional

_MODELS_DIR = os.environ.get(
    "LOCAL_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

_MODEL_PATHS = {
    "general": os.environ.get("LOCAL_GENERAL_MODEL_PATH", os.path.join(_MODELS_DIR, "qwen3-4b-instruct-2507-q4_k_m.gguf")),
}

# Same source/size as the Dockerfile's build-time download - used here so
# any environment without the weights pre-staged (e.g. the hosted demo,
# which can't commit ~2.5GB into git) can fetch them on first use instead
# of silently falling back to Fireworks for every "local" category.
_MODEL_DOWNLOAD = {
    "general": (
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        2497281120,
    ),
}


def _ensure_model_file(model_key: str) -> bool:
    """Download the GGUF weights on first use if they're not already
    present (e.g. this process isn't the Docker image, which downloads
    them at build time). Returns True once the file exists and is the
    expected size."""
    path = _MODEL_PATHS[model_key]
    if os.path.exists(path) and os.path.getsize(path) == _MODEL_DOWNLOAD[model_key][1]:
        return True

    url, expected_size = _MODEL_DOWNLOAD[model_key]
    try:
        import requests
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".part"
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        if os.path.getsize(tmp_path) != expected_size:
            os.remove(tmp_path)
            return False
        os.replace(tmp_path, path)
        return True
    except Exception:
        return False

_CATEGORY_MODEL = {
    "sentiment": "general",
    "ner": "general",
    "factual": "general",
    "summarization": "general",
    "code_debug": "general",
    "code_gen": "general",
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

    # Check llama_cpp is importable *before* possibly downloading ~1GB of
    # weights - environments without it installed (e.g. this hosted demo,
    # after llama-cpp-python was dropped from its requirements when
    # compiling it from source stalled indefinitely on Streamlit Cloud's
    # build infra) would otherwise download the full file just to fail
    # on import afterward.
    try:
        from llama_cpp import Llama
    except Exception:
        _load_failed.add(model_key)
        return None

    path = _MODEL_PATHS[model_key]
    if not (os.path.exists(path) and os.path.getsize(path) == _MODEL_DOWNLOAD[model_key][1]):
        if not _ensure_model_file(model_key):
            _load_failed.add(model_key)
            return None
    try:
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
    # factual bumped 130->200 and ner 80->100 after a stress test showed
    # the newer, more verbose Qwen3-4B model can run past the old budgets
    # (tuned for the smaller 1.5B models) on questions that invite a
    # longer answer, truncating mid-sentence - same failure mode already
    # fixed once for the Fireworks math category. Sentiment/summarization
    # are naturally self-limiting (one label+clause; explicit format
    # constraints) so left as-is.
    "sentiment": 40, "ner": 100, "factual": 200, "summarization": 100,
    "code_debug": 250, "code_gen": 300,
}


def preload() -> None:
    """Download + load the model up front (used by the webapp so the
    first user query isn't the one paying for the ~2.5GB download)."""
    _get_llm("general")


def available(category: str = "general") -> bool:
    model_key = _CATEGORY_MODEL.get(category, category)
    return _get_llm(model_key) is not None


def _generate(llm, category: str, prompt: str, temperature: float) -> Optional[str]:
    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPTS[category]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=_MAX_TOKENS[category],
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def answer(category: str, prompt: str) -> Optional[str]:
    """Return a locally-generated answer, or None if the model isn't
    available or the output fails a basic sanity check. Single sample,
    no confidence gate - kept for callers that just want the cheapest
    possible check (e.g. the webapp demo)."""
    model_key = _CATEGORY_MODEL.get(category)
    if model_key is None:
        return None
    llm = _get_llm(model_key)
    if llm is None:
        return None

    text = _generate(llm, category, prompt, temperature=0)
    if text is None or not _sane(category, text):
        return None
    return text


_SENTIMENT_LABEL_RE = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)


def _label_of(category: str, text: str) -> Optional[str]:
    if category == "sentiment":
        m = _SENTIMENT_LABEL_RE.search(text)
        return m.group(1).lower() if m else None
    return None


def _token_set(text: str) -> set:
    return {w.lower().strip(".,;:!?()[]\"'") for w in text.split() if len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Fallback thresholds, used only for a category the calibration curve
# (router/calibration.json, built by router/calibrate.py) doesn't cover
# yet - tuned conservatively since escalating a task that could have been
# answered correctly for free costs a few tokens, but trusting a
# locally-generated answer that later turns out wrong costs the accuracy
# gate - the more expensive mistake, by far.
_OVERLAP_THRESHOLD = 0.40
_VOTE_THRESHOLD = 0.66  # i.e. >=2 of 3 samples agreeing

# Target reliability once a real calibration curve is available: only
# trust a local answer when the fitted P(correct) at its confidence signal
# meets this bar.
_TARGET_RELIABILITY = 0.85

_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
_calibration_cache = None


def _load_calibration() -> dict:
    global _calibration_cache
    if _calibration_cache is not None:
        return _calibration_cache
    try:
        import json
        with open(_CALIBRATION_PATH) as f:
            _calibration_cache = json.load(f)
    except Exception:
        _calibration_cache = {}
    return _calibration_cache


def _calibrated_probability(points: list, signal: float) -> float:
    """Linear interpolation over an isotonic curve's (signal, p_correct)
    breakpoints, sorted ascending by signal."""
    if not points:
        return 0.0
    if signal <= points[0][0]:
        return points[0][1]
    if signal >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= signal <= x1:
            if x1 == x0:
                return y0
            t = (signal - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def confidence_signal(category: str, primary: str, samples: list) -> Optional[float]:
    """Raw self-consistency agreement signal in [0, 1] - higher means the
    independent samples agreed more. Category-appropriate since there's no
    single generic way to compare free-text vs. discrete-label answers.
    Shared by answer_confident() (the runtime gate) and calibrate.py (which
    fits the isotonic curve this signal is looked up against).
    """
    if category == "sentiment":
        labels = [_label_of(category, s) for s in samples]
        labels = [l for l in labels if l is not None]
        if len(labels) < 2:
            return None
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        _, top_count = max(counts.items(), key=lambda kv: kv[1])
        return top_count / len(labels)

    if category in ("code_debug", "code_gen"):
        # Syntax validity across independent samples is the agreement
        # signal for code: if the model can only produce valid syntax
        # some of the time for this exact prompt, it's shaky on it.
        from router.solvers import looks_like_python, python_syntax_error
        valid_count = sum(
            1 for s in samples
            if looks_like_python(s) and python_syntax_error(s) is None
        )
        return valid_count / len(samples)

    # factual / ner / summarization: no clean discrete label to vote on,
    # so use token-overlap similarity between the primary sample and each
    # additional sample as a proxy for "the model keeps saying the same
    # thing" rather than confabulating differently each time.
    primary_tokens = _token_set(primary)
    overlaps = [_jaccard(primary_tokens, _token_set(s)) for s in samples[1:]]
    if not overlaps:
        return None
    return sum(overlaps) / len(overlaps)


def sample_answers(category: str, prompt: str) -> Optional[tuple]:
    """Draw the deterministic (temp=0) sample plus 2 stochastic ones.
    Returns (primary, samples) with samples[0] == primary, or None if the
    model's unavailable or the primary sample fails the basic sanity
    check. Shared by answer_confident() and calibrate.py so both compute
    the confidence signal from an identically-gathered set of samples.
    """
    model_key = _CATEGORY_MODEL.get(category)
    if model_key is None:
        return None
    llm = _get_llm(model_key)
    if llm is None:
        return None

    primary = _generate(llm, category, prompt, temperature=0)
    if primary is None or not _sane(category, primary):
        return None

    samples = [primary]
    for _ in range(2):
        extra = _generate(llm, category, prompt, temperature=0.7)
        if extra is not None and _sane(category, extra):
            samples.append(extra)

    if len(samples) < 2:
        # Couldn't even get a second opinion - not enough signal to trust.
        return None
    return primary, samples


def answer_confident(category: str, prompt: str) -> Optional[str]:
    """Self-consistency gated version of answer(): the deterministic
    (temperature=0) sample must also agree with 2 additional stochastic
    samples before being trusted. The agreement signal is looked up
    against a calibration curve fitted from real labelled observations
    (router/calibrate.py) when available for this category, falling back
    to a conservative fixed threshold otherwise. Disagreement/low
    calibrated confidence escalates to Fireworks (returns None) rather
    than risking a locally-generated wrong answer.
    """
    sampled = sample_answers(category, prompt)
    if sampled is None:
        return None
    primary, samples = sampled

    signal = confidence_signal(category, primary, samples)
    if signal is None:
        return None

    calibration = _load_calibration()
    curve = calibration.get(category)
    if curve is not None:
        if curve.get("force_escalate"):
            return None
        p_correct = _calibrated_probability(curve.get("points", []), signal)
        return primary if p_correct >= _TARGET_RELIABILITY else None

    threshold = _VOTE_THRESHOLD if category in ("sentiment", "code_debug", "code_gen") else _OVERLAP_THRESHOLD
    return primary if signal >= threshold else None


_MATH_EXTRACT_PROMPT = (
    "Extract ONLY the arithmetic expression (numbers and + - * / ^ % ( ) only, "
    "no words, no units, no currency symbols, no equals sign) that computes the "
    "final numeric answer to this word problem. Output nothing else."
)

# Rate/unit-conversion problems ("60 miles in 45 minutes, what's the speed in
# mph") need an extra reasoning step (converting minutes to hours) before the
# arithmetic even starts. Verified empirically: the local model extracted
# "60/45" for exactly this kind of problem - a wrong *expression*, not an
# arithmetic slip, so evaluate_expression's guaranteed-correct computation
# doesn't help. Rather than risk a wrong answer, skip local extraction
# entirely for this class and defer to Fireworks, already proven reliable
# on it.
_RATE_CONVERSION_RE = re.compile(
    r"\bper (hour|minute|second|day|week|month|year)\b|\bspeed\b|\brate of\b|"
    r"\bmiles per\b|\bkm per\b|\bkilometers per\b|\bconvert\b|\bmph\b|\bkph\b",
    re.I,
)


def try_solve_math_word_problem(prompt: str) -> Optional[str]:
    """Word problems ('a $40 item marked up 30%...') need language
    understanding to turn into an expression, which is exactly what a
    small local model is reasonably good at - but small models are known
    to make arithmetic mistakes doing the actual computation themselves.
    So: local model extracts the expression only, and the real arithmetic
    is done by solvers.evaluate_expression (deterministic, always correct).
    Returns None (falls through to Fireworks) if the model's output isn't a
    clean, evaluable expression, or the problem needs a conversion step the
    extraction step has been caught getting wrong - never guesses.
    """
    from router.solvers import evaluate_expression  # local import: avoid a cycle at module load

    if _RATE_CONVERSION_RE.search(prompt):
        return None

    llm = _get_llm("general")
    if llm is None:
        return None

    # Self-consistency (3 samples, majority vote on the computed *value* -
    # not the raw expression text, since "40*1.3*0.9" and "40*0.9*1.3" are
    # textually different but numerically identical). This has to sample at
    # temperature>0: unlike Fireworks (where we observed real run-to-run
    # variance even at temperature=0, likely from serving-side batching
    # non-determinism), local llama.cpp inference is a single, unbatched
    # request and is genuinely deterministic at temperature=0 - repeating an
    # identical call would just reproduce the same answer 3/3 times and
    # prove nothing. Sampling lets the model actually explore different
    # reasoning paths so disagreement is a real signal, not theater.
    values = []
    for _ in range(3):
        try:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _MATH_EXTRACT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=60,
                temperature=0.7,
            )
            expr = result["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
        val = evaluate_expression(expr)
        if val is not None:
            values.append(val)

    if not values:
        return None

    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best_value, best_count = max(counts.items(), key=lambda kv: kv[1])
    return best_value if best_count >= 2 else None


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
