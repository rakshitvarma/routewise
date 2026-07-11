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
        lines = ["Answer every task below. Return ONLY a JSON array like "
                 '[{"task_id": "...", "answer": "..."}], one entry per task_id, no other text.', ""]
        for task_id, prompt in items:
            lines.append(f'task_id={task_id}: {prompt}')
        return "\n".join(lines)

    def answer_batch(self, category: str, items: List[Tuple[str, str]]) -> Dict[str, str]:
        """items: list of (task_id, prompt). Returns {task_id: answer}."""
        if not items:
            return {}

        model = self.pick_model(category)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPTS[category]},
                {"role": "user", "content": self._build_user_prompt(items)},
            ],
            "max_tokens": _MAX_TOKENS[category] * max(1, len(items)),
            "temperature": 0,
        }
        if any(h in model.lower() for h in _REASONING_CAPABLE_HINTS):
            payload["reasoning_effort"] = "none"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_calls += 1

        content = data["choices"][0]["message"]["content"]
        return self._parse_answers(content, [tid for tid, _ in items])

    @staticmethod
    def _parse_answers(content: str, expected_ids: List[str]) -> Dict[str, str]:
        match = re.search(r"\[.*\]", content, re.S)
        raw = match.group(0) if match else content
        try:
            parsed = json.loads(raw)
            return {str(item["task_id"]): str(item["answer"]) for item in parsed}
        except Exception:
            # Degrade gracefully: don't crash the whole run over one bad batch.
            return {tid: content.strip() for tid in expected_ids}
