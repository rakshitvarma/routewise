"""Thin Fireworks API client: batches many tasks into one chat completion.

All calls go through FIREWORKS_BASE_URL as required by the submission
guide. Batching (many task_ids answered by a single call) is the biggest
token lever available, matching the top leaderboard entry's approach.
"""
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# A plain factual/trivia prompt and an open-ended "derive/prove/explain in
# detail" request both classify as "factual" (correctly - there's no
# separate official category for them), but forcing the same terse
# "1-2 sentences" instruction onto both means derivation-style requests get
# cut short even when the token budget would allow more. This regex lets
# just those requests use a richer prompt/budget without touching the
# terse behavior tuned (and calibration-tested) for ordinary factual QA.
_DETAILED_RE = re.compile(
    r"\bderiv(e|ation)\b|\bprove\b|\bproof\b|\bstep[- ]by[- ]step\b|"
    r"\bexplain in detail\b|\bwalk through\b|\bshow your work\b|\bin depth\b",
    re.I,
)


def _prompt_key(category: str, prompt: str) -> str:
    """Category key used to look up _SYSTEM_PROMPTS/_MAX_TOKENS - same as
    `category` for everything except a detailed-explanation-style factual
    request, which gets its own richer entry."""
    if category == "factual" and _DETAILED_RE.search(prompt):
        return "factual_detailed"
    return category


# Conservative leading/trailing politeness stripper. A competitor (TERA)
# uses aggressive prompt compression to cut input tokens; that pays off for
# chatty real-user prompts but the Track 1 eval prompts are terse benchmark
# strings with little filler, and aggressive stripping risks corrupting the
# quoted text / passages that sentiment/NER/summarization are graded on. So
# this only trims obvious conversational wrappers at the very start/end, and
# only when the prompt carries no quoted content or embedded passage to
# preserve - a near-zero-risk trim that helps in the rare case a task does
# carry filler and does nothing (the common case) when it doesn't.
_LEAD_POLITE_RE = re.compile(
    r"^(?:hi|hello|hey)[,!.]?\s+|"
    r"^(?:please|kindly)\s+|"
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+|kindly\s+)?|"
    r"^(?:i\s+(?:would\s+like|'?d\s+like|want)\s+you\s+to)\s+|"
    r"^(?:kindly\s+)?help\s+me\s+(?:to\s+)?",
    re.I,
)
_TRAIL_POLITE_RE = re.compile(
    r"\s*(?:thanks?|thank\s+you)(?:\s+(?:so\s+much|very\s+much|a\s+lot|in\s+advance))?[.!]*\s*$",
    re.I,
)


def _compress_prompt(category: str, prompt: str) -> str:
    """Trim leading/trailing politeness, but never for summarization (the
    passage must stay verbatim) and never when the prompt embeds quoted
    text (sentiment/NER inputs, exact-reply instructions) that could be
    graded literally."""
    if category == "summarization":
        return prompt
    if any(q in prompt for q in ('"', "'", "`", "“", "‘")):
        return prompt
    trimmed = _LEAD_POLITE_RE.sub("", prompt, count=1)
    trimmed = _TRAIL_POLITE_RE.sub("", trimmed, count=1)
    trimmed = trimmed.strip()
    # Only accept the trim if it actually removed something and left a
    # substantive prompt behind - never return an empty/near-empty string.
    return trimmed if len(trimmed) >= 8 and trimmed != prompt.strip() else prompt


_SYSTEM_PROMPTS = {
    "factual": "Answer each question in 1-2 short sentences. No preamble, no filler.",
    "factual_detailed": (
        "Give the derivation/step-by-step reasoning as requested, but as "
        "concisely as correctness allows - each step in one short line, no "
        "restating the question, no filler commentary between steps. Use "
        "LaTeX ($...$ or $$...$$) for math notation and fenced code blocks "
        "for code. Do not truncate before reaching the final result."
    ),
    "sentiment": "Classify sentiment (positive/negative/neutral) and give a one-clause justification.",
    "summarization": "Summarise each passage to the exact length/format constraint given. No preamble.",
    "ner": "Extract named entities (person, org, location, date) as a compact labelled list.",
    "code_debug": "Find the bug and return the corrected code only, no explanation unless asked.",
    "code_gen": "Write the correct, complete function per the spec. Code only.",
    "logic": "Solve the constraint puzzle. On the first line, state the final answer wrapped in double asterisks like **answer**, satisfying every condition. Then briefly justify in plain text - no headers, no tables, no bullet lists.",
    "math": "Solve step by step. Keep each step to one short line (no long explanations), "
            "then give the final numeric answer clearly labeled on its own line at the end.",
}

_MAX_TOKENS = {
    "factual": 130,
    # 300 was tried and reverted: MiniMax's actual derivation style (LaTeX,
    # per-step headers) didn't compress as much as the tightened prompt
    # asked for, so it hit truncation and triggered the retry-at-2x-budget
    # safety net - which costs a full second generation, making the total
    # *worse* (1004 tokens) than the original 450-budget version (612).
    # 400 leaves headroom to avoid that expensive retry while still cutting
    # some slack versus 450.
    "factual_detailed": 400,
    "sentiment": 40,
    "summarization": 100,
    "ner": 80,
    "code_debug": 300,
    "code_gen": 350,
    "logic": 220,
    # Multi-step word problems (percentages, multi-quarter totals, unit
    # conversions) legitimately need a few lines of working plus the final
    # answer - 60 was tuned for "just the final number" but real prompts
    # show brief steps regardless of instruction, and got hard-truncated
    # mid-line. 150 comfortably fits a handful of one-line steps.
    "math": 150,
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
        # Shared session: connection pooling/keep-alive across the many
        # sequential and concurrent calls a single run makes.
        self._session = requests.Session()

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

    def _fallback_model(self, failed_model: str):
        """A different allowed (non-Gemma, to avoid the deploy cost) model to
        retry on when `failed_model` errors - a single unservable/rate-limited
        model must degrade to "answered by the other model" rather than "every
        task in that bucket gets an empty answer"."""
        for m in self.allowed_models:
            if m != failed_model and "gemma" not in m.lower():
                return m
        return None

    def _complete(self, model: str, system: str, user: str, max_tokens: int,
                   json_mode: bool = True, _is_fallback: bool = False) -> str:
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
            return self._session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                # The submission guide caps response time per request at 30s.
                # A 60s client timeout meant we'd sit waiting past the point
                # the judging proxy had likely already given up - fail fast
                # instead so a slow call surfaces as an exception (caught by
                # answer_all's retry/fallback) rather than a long, pointless
                # wait on a connection the proxy may have already dropped.
                timeout=25,
            )

        try:
            resp = _post()
            if resp.status_code == 400:
                # Some models may reject response_format/reasoning_effort params
                # outright - retry once with only the bare essentials before
                # giving up (cheaper than losing the whole category to a 400).
                payload.pop("response_format", None)
                payload.pop("reasoning_effort", None)
                resp = _post()
            resp.raise_for_status()
        except Exception:
            # This specific model may be down/rate-limited while others are
            # fine - one shot on a different allowed model before propagating.
            fallback = None if _is_fallback else self._fallback_model(model)
            if fallback is None:
                raise
            print(f"[warn] {model} failed, retrying via {fallback}", file=sys.stderr)
            return self._complete(fallback, system, user, max_tokens,
                                  json_mode=json_mode, _is_fallback=True)
        data = resp.json()

        usage = data.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_calls += 1
        content = data["choices"][0]["message"]["content"]

        # Reasoning models can leak their chain-of-thought as <think>...</think>
        # (or leave an unclosed <think> when truncated) - never let that reach
        # the grader as part of the answer text.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
        content = re.sub(r"<think>.*\Z", "", content, flags=re.S).strip()

        # A hard-truncated single answer (finish_reason=length) risks the
        # accuracy gate far more than one retry's tokens - retry once with
        # double the budget, but only on the plain-text path (batched JSON
        # budgets are already sized as a sum over their items).
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        if finish == "length" and not json_mode and not _is_fallback:
            print(f"[warn] {model} answer truncated at {max_tokens} tokens, retrying with x2", file=sys.stderr)
            return self._complete(model, system, user, max_tokens * 2,
                                  json_mode=False, _is_fallback=True)
        return content

    def answer_batch(self, category: str, items: List[Tuple[str, str]]) -> Dict[str, str]:
        """items: list of (task_id, prompt). Returns {task_id: answer}."""
        if not items:
            return {}

        model = self.pick_model(category)

        if len(items) == 1:
            # No JSON wrapper needed (and no ambiguous placeholder key to
            # confuse the model with) when there's only one task to answer.
            task_id, prompt = items[0]
            key = _prompt_key(category, prompt)
            content = self._complete(
                model, _SYSTEM_PROMPTS[key], prompt,
                _MAX_TOKENS[key], json_mode=False,
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

    def answer_all(self, buckets: Dict[str, List[Tuple[str, str]]]) -> Dict[str, str]:
        """Merge every category routed to the same model into one call,
        instead of one call per category - the top leaderboard entry's own
        breakdown lists this exact merge as one of its biggest token-count
        reductions ("Merged all other categories into one direct batch").
        Falls back per-item on parse gaps or degenerate/repeated output.
        """
        results: Dict[str, str] = {}

        # Logic puzzles get dedicated self-consistency handling (see
        # _answer_logic below) rather than joining the generic merged-batch
        # path: verified empirically that a single call can flip between a
        # correct answer and one that directly contradicts its own stated
        # constraints (model non-determinism on harder reasoning, not a
        # batching artifact - reproduced both isolated and batched).
        logic_items = buckets.get("logic", [])
        if logic_items:
            model = self.pick_model("logic")
            # Independent per-task self-consistency runs - run multiple
            # logic tasks concurrently too, not just the 3 calls inside each.
            # ThreadPoolExecutor.map()'s iterator raises at the position of
            # the first failing task and *silently discards every result
            # after it*, even ones that finished successfully in the
            # background (verified empirically) - as_completed + a per-future
            # try/except is required so one logic task's failure can't wipe
            # out every other logic task's already-correct answer.
            with ThreadPoolExecutor(max_workers=max(1, len(logic_items))) as pool:
                future_to_task = {
                    pool.submit(self._answer_logic, model, prompt): task_id
                    for task_id, prompt in logic_items
                }
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        results[task_id] = future.result()
                    except Exception as exc:
                        print(f"[warn] logic task {task_id} failed entirely: {exc}", file=sys.stderr)

        model_groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for category, items in buckets.items():
            if category == "logic":
                continue
            model = self.pick_model(category)
            for task_id, prompt in items:
                # Trim conversational filler once here so every downstream
                # path (single/batch/corrective) sends the leaner prompt.
                model_groups[model].append((task_id, category, _compress_prompt(category, prompt)))

        def _resolve_group(group_key: str, entries: List[Tuple[str, str, str]]) -> Dict[str, str]:
            model = group_key.split("::", 1)[0]
            if len(entries) == 1:
                task_id, category, prompt = entries[0]
                key = _prompt_key(category, prompt)
                content = self._complete(
                    model, _SYSTEM_PROMPTS[key], prompt,
                    _MAX_TOKENS[key], json_mode=False,
                )
                return {task_id: content.strip()}

            ids = ", ".join(f'"{tid}"' for tid, _, _ in entries)
            lines = [
                f"Return ONLY a JSON object with exactly these keys: {ids}. "
                "Each task below has a bracketed style instruction - follow it "
                "exactly. No other text outside the JSON object.", "",
            ]
            max_tokens = 20 * len(entries)
            for task_id, category, prompt in entries:
                key = _prompt_key(category, prompt)
                lines.append(f'"{task_id}" [{_SYSTEM_PROMPTS[key]}]: {prompt}')
                max_tokens += _MAX_TOKENS[key]

            content = self._complete(
                model,
                "Answer a batch of unrelated tasks, each with its own style instruction.",
                "\n".join(lines), max_tokens,
            )
            parsed = self._parse_answers(content, [tid for tid, _, _ in entries])

            distinct_values = {v for v in parsed.values()}
            degenerate = len(distinct_values) == 1 and len(next(iter(distinct_values))) > 200
            group_results = {}
            for task_id, category, prompt in entries:
                answer = "" if degenerate else parsed.get(task_id, "").strip()
                if not answer:
                    # Missing key or degenerate batch: one corrective
                    # single-task call rather than submitting an empty/wrong
                    # answer that's a guaranteed miss on the accuracy gate.
                    key = _prompt_key(category, prompt)
                    answer = self._complete(
                        model, _SYSTEM_PROMPTS[key], prompt,
                        _MAX_TOKENS[key], json_mode=False,
                    ).strip()
                group_results[task_id] = answer
            return group_results

        groups = [(k, v) for k, v in model_groups.items() if v]
        if groups:
            # Each group is an independent HTTP call (often to a different
            # model) - run them concurrently instead of one after another.
            # pool.map()'s iterator raises at the position of the first
            # group that throws and *silently discards every group's result
            # after it in iteration order*, even ones that already completed
            # successfully in the background (verified empirically: a single
            # failing model group could wipe out every other category's
            # already-correct answers - a very plausible cause of a total
            # accuracy-gate failure from one transient Fireworks error).
            # as_completed + a per-future try/except isolates each group's
            # failure to just that group's own tasks.
            with ThreadPoolExecutor(max_workers=len(groups)) as pool:
                future_to_key = {pool.submit(_resolve_group, k, v): k for k, v in groups}
                for future in as_completed(future_to_key):
                    group_key = future_to_key[future]
                    try:
                        results.update(future.result())
                    except Exception as exc:
                        print(f"[warn] Fireworks group {group_key} failed entirely: {exc}", file=sys.stderr)

        return results

    def _answer_logic(self, model: str, prompt: str) -> str:
        """Self-consistency (3 independent calls, majority vote) for
        constraint puzzles. Costs ~3x the tokens of a single call, but this
        is spent only on however few logic tasks appear (typically a small
        slice of the 19 fixed tasks), and directly targets a demonstrated
        failure mode: this exact class of puzzle flipped between a correct
        answer and a self-contradictory one across otherwise-identical calls.
        """
        # The 3 calls are independent (same prompt, sampled separately) - run
        # them concurrently rather than serialized, to cut 3x network
        # round-trip latency down to ~1x. Same tokens, same accuracy.
        # Isolated per-sample (as_completed + try/except, not pool.map): if
        # one sample errors, the other 1-2 successful samples should still
        # be usable for majority voting rather than the whole task failing.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    lambda: self._complete(
                        model, _SYSTEM_PROMPTS["logic"], prompt, _MAX_TOKENS["logic"], json_mode=False
                    ).strip()
                )
                for _ in range(3)
            ]
            responses = []
            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as exc:
                    print(f"[warn] a logic self-consistency sample failed: {exc}", file=sys.stderr)
        if not responses:
            raise RuntimeError("all 3 logic self-consistency samples failed")

        def key_of(text: str) -> str:
            m = re.search(r"\*\*([^*]+)\*\*", text)
            if m:
                return m.group(1).strip().lower()
            m = re.search(r"\b([A-Z][a-z]+)\b", text)
            return m.group(1).lower() if m else text[:30].lower()

        keys = [key_of(r) for r in responses]
        counts = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        top_key = max(counts, key=counts.get)
        if counts[top_key] >= 2:
            for r, k in zip(responses, keys):
                if k == top_key:
                    return r
        return responses[0]  # no majority (all 3 disagreed) - best effort

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
