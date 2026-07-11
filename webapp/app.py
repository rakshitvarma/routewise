"""RouteWise demo — live showcase of the Track 1 routing agent.

Reuses the exact same router package (classifier, solvers, fireworks_client)
that runs inside the submitted Docker image, so this demo reflects real
behavior rather than a reimplementation.
"""
import os
import sys
import time
import html as htmlmod

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.classifier import classify
from router.solvers import try_solve_math
from router.fireworks_client import FireworksClient

st.set_page_config(page_title="RouteWise", page_icon="🧭", layout="centered")

# Bridge Streamlit secrets into the environment variables FireworksClient
# reads, without touching the router package used by the submitted image.
for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
    if key not in os.environ and key in st.secrets:
        os.environ[key] = st.secrets[key]

# ---------------------------------------------------------------------------
# Visual metadata: category icon/color, and per-model "badge" (a small
# gradient monogram avatar rather than a real trademarked logo).
# ---------------------------------------------------------------------------
CATEGORY_META = {
    "math": ("🧮", "#7C5CFC"),
    "factual": ("📖", "#5CC8FC"),
    "sentiment": ("💬", "#FC5C8D"),
    "summarization": ("📝", "#5CFCA8"),
    "ner": ("🏷️", "#FCC85C"),
    "code_debug": ("🐛", "#FC8D5C"),
    "code_gen": ("⚙️", "#8D5CFC"),
    "logic": ("🧩", "#5CFCE0"),
}

MODEL_META = [
    ("minimax", "MiniMax M3", "M", "#7C5CFC", "#4E7CFC"),
    ("kimi", "Kimi K2.7", "K", "#20E3B2", "#0D8ABC"),
    ("gemma", "Gemma", "G", "#FBC02D", "#FF7043"),
    ("local", "Local (0 tokens)", "⚡", "#43E97B", "#38F9D7"),
]


def model_badge(model_id: str) -> str:
    low = (model_id or "").lower()
    for hint, name, glyph, c1, c2 in MODEL_META:
        if hint in low:
            return (
                f'<div class="rw-model">'
                f'<div class="rw-avatar" style="background:linear-gradient(135deg,{c1},{c2})">{glyph}</div>'
                f'<span>{name}</span></div>'
            )
    short = model_id.split("/")[-1] if model_id else "n/a"
    return (
        f'<div class="rw-model">'
        f'<div class="rw-avatar" style="background:linear-gradient(135deg,#5A5F73,#3A3E4D)">?</div>'
        f'<span>{htmlmod.escape(short)}</span></div>'
    )


def category_pill(category: str) -> str:
    icon, color = CATEGORY_META.get(category, ("❔", "#9AA0AE"))
    return (
        f'<span class="rw-pill" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{icon}&nbsp;{category}</span>'
    )


EXAMPLES = [
    ("math", "Math", "A store marks up a $40 item by 30% and then offers a 10% discount on the marked-up price. What is the final price?"),
    ("factual", "Factual", "Explain what a black hole is in simple terms."),
    ("sentiment", "Sentiment", "Classify the sentiment: 'The food was okay, nothing special, but the service was excellent.'"),
    ("summarization", "Summary", "Summarise the following in one short sentence: Researchers found that participants who slept less than six hours a night for a week showed slower reaction times and reduced memory recall compared to a control group that slept eight hours."),
    ("ner", "NER", "Extract all named entities from: 'Marie Curie won the Nobel Prize in Physics in 1903 while working in Paris.'"),
    ("code_debug", "Code debug", "Find and fix the bug: ```def is_even(n):\n    return n % 2 == 1```"),
    ("code_gen", "Code gen", "Write a Python function is_palindrome(s) that returns True if a string reads the same forwards and backwards, ignoring case and spaces."),
    ("logic", "Logic", "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed', but all labels are wrong. You may pick one fruit from one box to determine the correct labels. Which box should you pick from, and why?"),
]

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .rw-hero {
        padding: 1.6rem 0 0.4rem 0;
    }
    .rw-hero h1 {
        font-size: 2.6rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #A78BFA, #60E6D8 60%, #5CC8FC);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .rw-hero p { color: #9AA0AE; font-size: 1rem; margin-top: 0.35rem; max-width: 640px; }

    .rw-stats { display: flex; gap: 0.7rem; margin: 1rem 0 1.4rem 0; flex-wrap: wrap; }
    .rw-stat {
        flex: 1; min-width: 130px; background: linear-gradient(160deg, #1B1F2B, #14161F);
        border: 1px solid #262B3A; border-radius: 16px; padding: 0.9rem 1rem;
    }
    .rw-stat .n { font-size: 1.5rem; font-weight: 700; color: #EDEDF2; }
    .rw-stat .l { font-size: 0.75rem; color: #8A8FA3; text-transform: uppercase; letter-spacing: 0.04em; }

    .rw-pill {
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
    }
    .rw-chip {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 500; color: #C7CBDA;
        background: #232838; border: 1px solid #2E3448; margin-left: 6px;
    }

    .rw-model { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #D5D8E3; }
    .rw-avatar {
        width: 26px; height: 26px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-size: 0.8rem;
        font-weight: 700; color: #0E1117; flex-shrink: 0;
        box-shadow: 0 0 0 2px #0E111744;
    }

    .rw-card {
        background: linear-gradient(160deg, #191D29, #12141C);
        border-radius: 16px; padding: 1.1rem 1.3rem; margin-top: 0.9rem;
        border: 1px solid #262B3A;
    }
    .rw-card .rw-top { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
    .rw-answer {
        white-space: pre-wrap; margin-top: 0.7rem; padding-top: 0.7rem;
        border-top: 1px solid #262B3A; color: #E6E6EA; font-size: 0.94rem; line-height: 1.5;
    }

    .rw-hist-card {
        background: #14161F; border: 1px solid #232838; border-radius: 12px;
        padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
    }
    .rw-hist-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
    .rw-hist-prompt { color: #8A8FA3; font-size: 0.82rem; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="rw-hero"><h1>🧭 RouteWise</h1>'
    '<p>Hybrid token-efficient routing agent — built for AMD Developer Hackathon '
    'Act II, Track 1. Classifies and solves for free where it can, and only pays '
    'for Fireworks inference on tasks that genuinely need it.</p></div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

stats_box = st.empty()


def render_stats():
    n = len(st.session_state.history)
    total = st.session_state.total_tokens
    avg = round(total / n) if n else 0
    stats_box.markdown(
        f'<div class="rw-stats">'
        f'<div class="rw-stat"><div class="n">{total}</div><div class="l">Tokens spent</div></div>'
        f'<div class="rw-stat"><div class="n">{n}</div><div class="l">Queries run</div></div>'
        f'<div class="rw-stat"><div class="n">{avg}</div><div class="l">Avg tokens / query</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


render_stats()

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. **Local classifier** (regex, 0 tokens) routes the task into one "
        "of 8 categories.\n"
        "2. **Deterministic math solver** answers bare arithmetic instantly "
        "for free — word problems fall through to Fireworks.\n"
        "3. **Logic puzzles** get 3-call self-consistency (majority vote) — "
        "cheap insurance against a demonstrated flaky-reasoning failure mode.\n"
        "4. **Everything else** is merged by model into as few Fireworks "
        "calls as possible (not one call per category)."
    )
    st.divider()
    st.markdown("**Models in play**", help="Only the models actually reachable via ALLOWED_MODELS are used.")
    for hint, name, glyph, c1, c2 in MODEL_META:
        st.markdown(
            f'<div class="rw-model" style="margin-bottom:6px">'
            f'<div class="rw-avatar" style="background:linear-gradient(135deg,{c1},{c2})">{glyph}</div>'
            f'<span>{name}</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown(
        "[GitHub repo](https://github.com/rakshitvarma/routewise) · "
        "[Docker image](https://github.com/rakshitvarma/routewise/pkgs/container/routewise)"
    )

st.subheader("Try it")
cols = st.columns(4)
for i, (cat, label, ex_prompt) in enumerate(EXAMPLES):
    icon, _ = CATEGORY_META.get(cat, ("❔", "#9AA0AE"))
    if cols[i % 4].button(f"{icon} {label}", use_container_width=True):
        st.session_state.prompt_input = ex_prompt

prompt = st.text_area(
    "Task prompt", key="prompt_input", height=100,
    placeholder="Type a task, or click an example above...",
)
run = st.button("Route & Answer →", type="primary")

has_creds = all(os.environ.get(k) for k in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"))
if not has_creds:
    st.info(
        "Fireworks credentials aren't configured for this demo instance, so only "
        "the zero-token local paths (math, classification) will run live. "
        "Everything else will show the routing decision without a live answer.",
        icon="ℹ️",
    )

if run and prompt.strip():
    category = classify(prompt)
    started = time.time()

    local_answer = try_solve_math(prompt) if category == "math" else None

    answer, model_used, tokens_used = None, None, 0
    if local_answer is not None:
        answer, model_used = local_answer, "local (deterministic)"
    elif has_creds:
        try:
            client = FireworksClient()
            if category == "logic":
                model_used = client.pick_model("logic")
                answer = client._answer_logic(model_used, prompt)
            else:
                result = client.answer_all({category: [("live", prompt)]})
                answer = result.get("live", "")
                model_used = client.pick_model(category)
            tokens_used = client.total_tokens
            st.session_state.total_tokens += tokens_used
        except Exception as exc:
            answer = f"(Fireworks call failed: {exc})"
            model_used = "error"
    else:
        answer, model_used = "(no live credentials configured for this demo)", "n/a"

    elapsed = time.time() - started
    safe_answer = htmlmod.escape(answer or "")

    st.markdown(
        f'<div class="rw-card">'
        f'<div class="rw-top">'
        f'{category_pill(category)}'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'{model_badge(model_used)}'
        f'<span class="rw-chip">{tokens_used} tokens</span>'
        f'<span class="rw-chip">{elapsed:.1f}s</span>'
        f'</div></div>'
        f'<div class="rw-answer">{safe_answer}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.session_state.history.insert(0, {
        "prompt": prompt,
        "category": category,
        "model": model_used,
        "tokens": tokens_used,
        "answer": answer,
    })
    render_stats()

if st.session_state.history:
    st.subheader("Session history")
    for i, entry in enumerate(st.session_state.history):
        prompt_preview = entry["prompt"][:90] + ("…" if len(entry["prompt"]) > 90 else "")
        st.markdown(
            f'<div class="rw-hist-card">'
            f'<div class="rw-hist-top">'
            f'{category_pill(entry["category"])}'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'{model_badge(entry["model"])}'
            f'<span class="rw-chip">{entry["tokens"]} tokens</span>'
            f'</div></div>'
            f'<div class="rw-hist-prompt">{htmlmod.escape(prompt_preview)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("View answer", expanded=False):
            st.markdown(entry["answer"])
