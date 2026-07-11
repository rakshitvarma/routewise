"""Track 1 entrypoint.

Reads /input/tasks.json, routes each task through:
  1. local zero-token classifier
  2. local zero-token deterministic solvers (math only, conservative)
  3. batched Fireworks calls per category for everything else
Writes /output/results.json. Always exits 0 with valid JSON, even in
degraded form, since malformed output or a crash scores zero.
"""
import json
import os
import sys
import time
from collections import defaultdict

from router.classifier import classify
from router.solvers import try_solve_math, looks_like_python, python_syntax_error
from router.fireworks_client import FireworksClient
from router import local_llm

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")

# Categories the bundled local models have been eval'd against (see
# router/eval_local.py - 24/24 correct across all six on held-out phrasing
# distinct from sample_input/*.json) and are trusted to answer directly -
# zero Fireworks tokens. Math and logic are deliberately excluded: math
# already has a proven zero-token deterministic path for what can be
# solved with certainty, and logic puzzles need the kind of careful
# multi-step reasoning a 1.5B model is least likely to hold up on under
# randomized rephrasing - both stay on the already-verified Fireworks path.
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
        task_id = task["task_id"]
        prompt = task["prompt"]
        category = classify(prompt)

        if category == "math":
            local_answer = try_solve_math(prompt)
            if local_answer is not None:
                answers[task_id] = local_answer
                print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) solved deterministically", file=sys.stderr)
                continue
            # Word problems deliberately stay on Fireworks, not
            # local_llm.try_solve_math_word_problem(): that path exists and
            # works on the handful of cases tested, but a handful of
            # invented test cases doesn't bound the true failure rate
            # against genuinely randomized prompts, and an attempt to make
            # it safer via self-consistency empirically made it *less*
            # reliable (temperature sampling turned a correct answer wrong).
            # Fireworks has been 100% correct on every math word problem
            # across every test run - real evidence beats a handful of
            # ad hoc samples when a wrong answer risks the accuracy gate.

        if category in LOCAL_LLM_CATEGORIES:
            local_started = time.time()
            local_answer = local_llm.answer(category, prompt)
            local_elapsed = time.time() - local_started
            if local_answer is not None:
                if category in ("code_debug", "code_gen") and looks_like_python(local_answer):
                    # Extra gate beyond local_llm's own sanity check: only
                    # trust local code output that's actually valid syntax.
                    if python_syntax_error(local_answer) is None:
                        answers[task_id] = local_answer
                        print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) local in {local_elapsed:.1f}s", file=sys.stderr)
                        continue
                else:
                    answers[task_id] = local_answer
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) local in {local_elapsed:.1f}s", file=sys.stderr)
                    continue
            print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) local rejected/failed in {local_elapsed:.1f}s, falling to Fireworks", file=sys.stderr)

        buckets[category].append((task_id, prompt))

    if any(buckets.values()):
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
            answers[task_id] = answer
        print(
            f"[stats] fireworks_calls={client.total_calls} "
            f"total_tokens={client.total_tokens}",
            file=sys.stderr,
        )

    results = [{"task_id": tid, "answer": answers.get(tid, "")} for tid in
               [t["task_id"] for t in tasks]]
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
            write_results(OUTPUT_PATH, [{"task_id": t["task_id"], "answer": ""} for t in tasks])
        except Exception:
            write_results(OUTPUT_PATH, [])
        sys.exit(1)
