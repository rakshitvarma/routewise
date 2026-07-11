"""Thin Fireworks API client: batches many tasks into one chat completion.

All calls go through FIREWORKS_BASE_URL as required by the submission
guide. Batching (many task_ids answered by a single call) is the biggest
token lever available, matching the top leaderboard entry's approach.
"""
import json
import os
import re
from typing import Dict, List, Tuple

import requests

_REASONING_CAPABLE_HINTS = ("gpt-oss", "deepseek", "qwq", "r1", "minimax", "kimi")

# Real ALLOWED_MODELS for this hackathon (from Discord, 2026-07-11):
#   minimax-m3, kimi-k2p7-code, gemma-4-31b-it, gemma-4-26b-a4b-it,
#   gemma-4-31b-it-nvfp4
# Gemma models on this account incur a ~$20 one-time deployment cost EACH
# the first time they're invoked. With a $50 total credit budget, calling
# more than one Gemma variant (or calling one repeatedly during dev
# iteration before it's warm) risks burning most of the budget before we
# even submit. So: only ever target ONE specific Gemma model, and only
# for ONE category, and only when explicitly enabled via env var.
_GEMMA_BONUS_CATEGORY = "sentiment"
_ENABLE_GEMMA_BONUS = os.environ.get("ENABLE_GEMMA_BONUS", "false").lower() == "true"
# a4b = "4B active params" MoE variant — cheapest of the three Gemma
# options to run once deployed, so prefer it over the dense 31b variants.
_PREFERRED_GEMMA_HINT = "a4b"

_SYSTEM_PROMPTS = {
    "factual": "Answer each question in 1-2 short sentences. No preamble, no filler.",
    "sentiment": "Classify sentiment (positive/negative/neutral) and give a one-clause justification.",
    "summarization": "Summarise each passage to the exact length/format constraint given. No preamble.",
    "ner": "Extract named entities (person, org, location, date) as a compact labelled list.",
    "code_debug": "Find the bug and return the corrected code only, no explanation unless asked.",
    "code_gen": "Write the correct, complete function per the spec. Code only.",
    "logic": "Solve the constraint puzzle. State the final answer clearly, satisfying every condition.",
    "math": "Solve step-by-step internally, then give only the final numeric/short answer.",
}

_MAX_TOKENS = {
    "factual": 70,
    "sentiment": 40,
    "summarization": 100,
    "ner": 80,
    "code_debug": 300,
    "code_gen": 350,
    "logic": 150,
    "math": 60,
}


class FireworksClient:
    def __init__(self):
        self.api_key = os.environ["FIREWORKS_API_KEY"]
        self.base_url = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
        self.allowed_models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
        if not self.allowed_models:
            raise RuntimeError("ALLOWED_MODELS is empty")
        self.total_tokens = 0
        self.total_calls = 0

    def _find(self, hint: str):
        for m in self.allowed_models:
            if hint in m.lower():
                return m
        return None

    def _gemma_model(self):
        return self._find(_PREFERRED_GEMMA_HINT) or next(
            (m for m in self.allowed_models if "gemma" in m.lower()), None
        )

    def pick_model(self, category: str) -> str:
        if category == _GEMMA_BONUS_CATEGORY and _ENABLE_GEMMA_BONUS:
            gemma = self._gemma_model()
            if gemma:
                return gemma
        if category in ("code_debug", "code_gen"):
            kimi = self._find("kimi")
            if kimi:
                return kimi
        minimax = self._find("minimax")
        if minimax:
            return minimax
        # Fall back to any non-Gemma model to avoid an accidental deploy cost.
        non_gemma = [m for m in self.allowed_models if "gemma" not in m.lower()]
        return non_gemma[0] if non_gemma else self.allowed_models[0]

    def _build_user_prompt(self, items: List[Tuple[str, str]]) -> str:
        # json_object response_format requires an OBJECT at the root, not an
        # array. Earlier version used a generic '{"task_id": "answer"}'
        # example in the instruction text, and the model pattern-matched the
        # literal placeholder word "task_id" as the key instead of
        # substituting the real id - spelling out the *actual* required keys
        # avoids that ambiguity entirely.
        ids = ", ".join(f'"{tid}"' for tid, _ in items)
        lines = [f"Return ONLY a JSON object with exactly these keys: {ids}. "
                 "Each key's value is your answer to that task. No other text.", ""]
        for task_id, prompt in items:
            lines.append(f'"{task_id}": {prompt}')
        return "\n".join(lines)

    def _complete(self, model: str, system: str, user: str, max_tokens: int,
                   json_mode: bool = True) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # reasoning_effort="none" combined with json_object response_format
        # causes MiniMax-M3 to degenerate into repeating its own output until
        # max_tokens is hit (verified empirically) - only safe to suppress
        # reasoning on the plain-text (non-JSON) path.
        if not json_mode and any(h in model.lower() for h in _REASONING_CAPABLE_HINTS):
            payload["reasoning_effort"] = "none"

        def _post():
            return requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

        resp = _post()
        if resp.status_code == 400:
            # Some models may reject response_format/reasoning_effort params
            # outright - retry once with only the bare essentials before
            # giving up (cheaper than losing the whole category to a 400).
            payload.pop("response_format", None)
            payload.pop("reasoning_effort", None)
            resp = _post()
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_calls += 1
        return data["choices"][0]["message"]["content"]

    def answer_batch(self, category: str, items: List[Tuple[str, str]]) -> Dict[str, str]:
        """items: list of (task_id, prompt). Returns {task_id: answer}."""
        if not items:
            return {}

        model = self.pick_model(category)

        if len(items) == 1:
            # No JSON wrapper needed (and no ambiguous placeholder key to
            # confuse the model with) when there's only one task to answer.
            task_id, prompt = items[0]
            content = self._complete(
                model, _SYSTEM_PROMPTS[category], prompt,
                _MAX_TOKENS[category], json_mode=False,
            )
            return {task_id: content.strip()}

        content = self._complete(
            model,
            _SYSTEM_PROMPTS[category],
            self._build_user_prompt(items),
            _MAX_TOKENS[category] * len(items) + 20 * len(items),
        )
        result = self._parse_answers(content, [tid for tid, _ in items])

        # Defense in depth: a degenerate/repeating response (or a parse
        # failure) shows up as every task_id mapping to the same long blob.
        # Rather than submit that same wrong answer for every task in the
        # batch, fall back to answering each one individually - costs more
        # tokens but guarantees distinct, correct-shaped answers.
        distinct_values = {v for v in result.values()}
        if len(items) > 1 and len(distinct_values) == 1 and len(next(iter(distinct_values))) > 200:
            return {tid: self.answer_batch(category, [(tid, p)])[tid] for tid, p in items}
        return result

    def fix_code(self, category: str, prompt: str, broken_answer: str, error: str) -> str:
        """One corrective call for a single task whose code failed to parse."""
        model = self.pick_model(category)
        system = _SYSTEM_PROMPTS[category] + " Return ONLY the corrected code, no JSON, no markdown fences."
        user = (
            f"Original task: {prompt}\n\nYour previous answer had a Python syntax "
            f"error ({error}):\n{broken_answer}\n\nReturn the corrected, syntactically "
            f"valid code."
        )
        return self._complete(model, system, user, _MAX_TOKENS[category], json_mode=False)

    @staticmethod
    def _parse_answers(content: str, expected_ids: List[str]) -> Dict[str, str]:
        match = re.search(r"\{.*\}", content, re.S)
        raw = match.group(0) if match else content
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
            if isinstance(parsed, list):  # tolerate the old array shape too
                return {str(item["task_id"]): str(item["answer"]) for item in parsed}
            raise ValueError("unexpected JSON shape")
        except Exception:
            # Degrade gracefully: don't crash the whole run over one bad batch.
            return {tid: content.strip() for tid in expected_ids}
