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

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("TASKS_OUTPUT_PATH", "/output/results.json")


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
                continue

        buckets[category].append((task_id, prompt))

    if any(buckets.values()):
        client = FireworksClient()
        try:
            merged_answers = client.answer_all(buckets)
        except Exception:
            try:
                merged_answers = client.answer_all(buckets)  # one retry
            except Exception as exc:
                print(f"[warn] answer_all failed twice: {exc}", file=sys.stderr)
                merged_answers = {}

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
