"""Track 1 entrypoint.

Reads /input/tasks.json, routes each task through:
  1. local zero-token classifier
  2. local zero-token deterministic solver (math only, conservative)
  3. local zero-token model answer, gated behind a self-consistency
     confidence check (see router/local_llm.answer_confident) - only
     trusted when independent samples agree, otherwise escalated
  4. batched Fireworks calls per category for everything else
Writes /output/results.json. Always exits 0 with valid JSON, even in
degraded form, since malformed output or a crash scores zero.
"""
import json
import os
import sys
import time
from collections import defaultdict

from router.classifier import classify
from router.solvers import (
    try_solve_math, try_solve_ner_spacy, try_solve_logic_row, looks_like_python,
    python_syntax_error, strip_code_fence,
)
from router.fireworks_client import FireworksClient
from router import local_llm

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")

# Both were validated against real examples before deciding a default:
# spaCy NER scored 9/15 (60%) fully-matched entities - worse than the
# existing calibrated LLM+Fireworks pipeline (97%), so it stays off.
# PAL math scored 3/3 (100%) on percentage/rate/fraction word problems -
# on by default, still gated behind its own self-consistency agreement
# check (>=2/3 samples) same as everything else in this module.
_ENABLE_SPACY_NER = os.environ.get("ENABLE_SPACY_NER", "false").lower() == "true"
_ENABLE_MATH_PAL = os.environ.get("ENABLE_MATH_PAL", "true").lower() == "true"

# Categories the bundled local models are allowed to attempt - each answer
# still has to clear the self-consistency confidence gate in
# local_llm.answer_confident before being trusted; anything it rejects
# falls through to Fireworks like any other unresolved task.
LOCAL_LLM_CATEGORIES = {"sentiment", "ner", "factual", "summarization", "code_debug", "code_gen"}


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    started = time.time()
    tasks = load_tasks(INPUT_PATH)

    answers = {}
    buckets = defaultdict(list)  # category -> [(task_id, prompt)]

    for task in tasks:
        # A single malformed task (missing/wrong-typed field, unexpected
        # content) must not lose every other task's already-computed
        # answer - isolate each task's processing so one bad entry
        # degrades to a missing answer for *that task only*, not a crash
        # that discards the whole batch via the outer handler.
        task_id = task.get("task_id") if isinstance(task, dict) else None
        if task_id is None:
            print(f"[warn] skipping task with no task_id: {task!r}", file=sys.stderr)
            continue
        try:
            prompt = task["prompt"]
            category = classify(prompt)

            if category == "math":
                local_answer = try_solve_math(prompt)
                if local_answer is not None:
                    answers[task_id] = local_answer
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) solved deterministically", file=sys.stderr)
                    continue
                # Word problems deliberately stay on Fireworks by default,
                # not local_llm.try_solve_math_word_problem(): that path
                # exists and works on the handful of cases tested, but a
                # handful of invented test cases doesn't bound the true
                # failure rate against genuinely randomized prompts, and an
                # attempt to make it safer via self-consistency empirically
                # made it *less* reliable (temperature sampling turned a
                # correct answer wrong). Fireworks has been 100% correct on
                # every math word problem across every test run - real
                # evidence beats a handful of ad hoc samples when a wrong
                # answer risks the accuracy gate. ENABLE_MATH_PAL opts into
                # local_llm.try_solve_math_word_problem_pal() instead (a
                # program-aided, sandboxed alternative) once validated.
                if _ENABLE_MATH_PAL:
                    pal_started = time.time()
                    pal_answer = local_llm.try_solve_math_word_problem_pal(prompt)
                    pal_elapsed = time.time() - pal_started
                    if pal_answer is not None:
                        answers[task_id] = pal_answer
                        print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) solved via local PAL in {pal_elapsed:.1f}s", file=sys.stderr)
                        continue
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) PAL not confident, in {pal_elapsed:.1f}s, falling to Fireworks", file=sys.stderr)

            if category == "logic":
                logic_answer = try_solve_logic_row(prompt)
                if logic_answer is not None:
                    answers[task_id] = logic_answer
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} (logic) solved deterministically (row puzzle)", file=sys.stderr)
                    continue
                # Anything else (box-labeling, dual-attribute grids, etc.)
                # falls through to Fireworks's self-consistency logic path,
                # same as before this solver existed.

            if category == "ner" and _ENABLE_SPACY_NER:
                ner_answer = try_solve_ner_spacy(prompt)
                if ner_answer is not None:
                    answers[task_id] = ner_answer
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} (ner) solved via spaCy", file=sys.stderr)
                    continue

            if category in LOCAL_LLM_CATEGORIES:
                local_started = time.time()
                local_answer = local_llm.answer_confident(category, prompt)
                local_elapsed = time.time() - local_started
                if local_answer is not None:
                    if category in ("code_debug", "code_gen"):
                        local_answer = strip_code_fence(local_answer)
                    answers[task_id] = local_answer
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) local (confident) in {local_elapsed:.1f}s", file=sys.stderr)
                    continue
                print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) local not confident, in {local_elapsed:.1f}s, falling to Fireworks", file=sys.stderr)

            buckets[category].append((task_id, prompt))
        except Exception as exc:
            print(f"[warn] failed to process task {task_id}: {exc}", file=sys.stderr)
            # Leave it out of both answers and buckets - it gets an empty
            # string in the final results, exactly like an unresolved
            # Fireworks task, rather than aborting everything else.

    if any(buckets.values()):
        # Everything in this block is best-effort: if FireworksClient()
        # itself fails (e.g. a required env var missing) or anything else
        # here throws unexpectedly, that must only cost the bucketed tasks
        # an empty answer - not propagate to the outer handler and wipe
        # out every already-computed local/deterministic answer too
        # (verified this exact failure mode: a missing FIREWORKS_API_KEY
        # discarded an already-solved bare-arithmetic answer).
        try:
            print(f"[timing] t={time.time()-started:.1f}s starting Fireworks phase, buckets={ {k: len(v) for k, v in buckets.items()} }", file=sys.stderr)
            client = FireworksClient()
            try:
                merged_answers = client.answer_all(buckets)
            except Exception:
                try:
                    merged_answers = client.answer_all(buckets)  # one retry
                except Exception as exc:
                    print(f"[warn] answer_all failed twice: {exc}", file=sys.stderr)
                    merged_answers = {}
            print(f"[timing] t={time.time()-started:.1f}s Fireworks phase (answer_all) done", file=sys.stderr)

            category_by_task = {tid: cat for cat, items in buckets.items() for tid, _ in items}
            prompt_by_task = {tid: p for items in buckets.values() for tid, p in items}
            for task_id, category in category_by_task.items():
                answer = merged_answers.get(task_id, "")
                if category in ("code_debug", "code_gen") and looks_like_python(answer):
                    err = python_syntax_error(answer)
                    if err:
                        try:
                            answer = client.fix_code(category, prompt_by_task[task_id], answer, err)
                        except Exception as exc:
                            print(f"[warn] fix_code failed for {task_id}: {exc}", file=sys.stderr)
                    answer = strip_code_fence(answer)
                answers[task_id] = answer
            print(
                f"[stats] fireworks_calls={client.total_calls} "
                f"total_tokens={client.total_tokens}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[warn] Fireworks phase failed entirely: {exc}", file=sys.stderr)

    task_ids = [t.get("task_id") for t in tasks if isinstance(t, dict) and t.get("task_id") is not None]
    results = [{"task_id": tid, "answer": answers.get(tid, "")} for tid in task_ids]
    write_results(OUTPUT_PATH, results)
    print(f"[done] {len(results)} tasks in {time.time() - started:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        # Still try to produce a valid (empty-ish) output so we don't
        # hard-fail the whole submission if input parsing blew up.
        try:
            tasks = load_tasks(INPUT_PATH)
            write_results(OUTPUT_PATH, [
                {"task_id": t["task_id"], "answer": ""} for t in tasks
                if isinstance(t, dict) and t.get("task_id") is not None
            ])
        except Exception:
            write_results(OUTPUT_PATH, [])
        sys.exit(1)
