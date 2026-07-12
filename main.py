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
import threading
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

# spaCy NER scored 9/15 (60%) fully-matched entities - worse than the
# existing calibrated LLM+Fireworks pipeline (97%), so it stays off.
#
# PAL math scored well on a handful of examples (3/3), but every clean
# (uncontended) test run this session showed each PAL attempt costing
# 20-45s - 3x local generations plus sandboxed execution, per task. A
# hidden evaluation set with more math word problems than we tested could
# compound that past the 10-minute cap. TIMEOUT scores zero; Fireworks has
# been 100% correct and fast (2-5s) on every math word problem tested -
# defaulting PAL off trades a modest, unproven token saving for a real,
# proven-safe path. Left available to opt into (e.g. for local demoing)
# via the env var.
_ENABLE_SPACY_NER = os.environ.get("ENABLE_SPACY_NER", "false").lower() == "true"
_ENABLE_MATH_PAL = os.environ.get("ENABLE_MATH_PAL", "false").lower() == "true"

# Deterministic solvers (try_solve_math, try_solve_logic_row, spaCy) are
# near-instant - no guard needed. The local-model paths are not: PAL math
# has been observed taking 20-45s+ per task when self-consistency doesn't
# immediately agree, and the confidence-gate's 3-sample local generation
# is a similar cost. Our own test sets stay well under the 10-minute cap,
# but a hidden evaluation set with more math/local-eligible tasks than we
# tested against could compound that per-task cost past it. Once the
# elapsed time crosses this budget, skip straight to the (fast, batched,
# parallel) Fireworks bucket for every remaining task rather than risk a
# TIMEOUT - a token cost is always recoverable, a TIMEOUT scores zero.
# 420s (7 min) leaves a 180s margin for the Fireworks phase + write step,
# tightened from 480s given real TIMEOUT reports on a submitted run.
_TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", "420"))

# Categories the bundled local models are allowed to attempt - each answer
# still has to clear the self-consistency confidence gate in
# local_llm.answer_confident before being trusted; anything it rejects
# falls through to Fireworks like any other unresolved task.
LOCAL_LLM_CATEGORIES = {"sentiment", "ner", "factual", "summarization", "code_debug", "code_gen"}

# Absolute last-resort backstop, independent of _TIME_BUDGET_SECONDS (which
# only stops *starting* new slow local attempts - it can't help if a call
# already in progress hangs, e.g. under host contention we've directly
# observed calls take 500s+). A background thread force-flushes whatever
# answers exist so far and hard-exits the whole process, regardless of what
# the main thread is stuck on - a SIGALRM-style in-thread interrupt doesn't
# work here (verified: llama.cpp's blocking native call never yields back
# to Python for a signal handler to fire), but killing the process from a
# separate thread does. 560s leaves a 40s margin under the 10-minute cap.
_HARD_DEADLINE_SECONDS = float(os.environ.get("HARD_DEADLINE_SECONDS", "560"))


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path, results):
    """Atomic write (write to a temp file, then os.replace) - a plain
    open(path, "w") leaves a truncated/corrupt file if the process is
    killed mid-write (external timeout kill, OOM, watchdog exit); the
    grader would then see invalid JSON instead of whatever we'd already
    computed. os.replace is atomic on both POSIX and Windows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def normalize_tasks(tasks):
    """Pair every input task with a task_id, generating a placeholder
    (task-N) for any task missing one instead of dropping it - the output
    must always have exactly one row per input row. A grader that expects
    len(results) == len(tasks) (or a 1:1 task_id correspondence it can't
    find) may treat a shorter results file as invalid, which is a far
    worse outcome than one row answered with an empty string."""
    normalized = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict) and task.get("task_id") is not None:
            task_id = task["task_id"]
        else:
            task_id = f"task-{i + 1}"
            print(f"[warn] task at index {i} has no task_id, using placeholder {task_id!r}: {task!r}", file=sys.stderr)
        normalized.append((task_id, task))
    return normalized


def main():
    started = time.time()
    tasks = load_tasks(INPUT_PATH)
    normalized_tasks = normalize_tasks(tasks)
    all_task_ids = [tid for tid, _ in normalized_tasks]

    # Write a valid placeholder result immediately, before any inference -
    # so even a crash/kill in the first second still leaves a minimally
    # valid (if empty) file on disk rather than nothing at all.
    write_results(OUTPUT_PATH, [{"task_id": tid, "answer": ""} for tid in all_task_ids])

    answers = {}
    buckets = defaultdict(list)  # category -> [(task_id, prompt)]
    # In-run dedup: identical prompts (whatever their task_ids) get computed
    # once and copied - the evaluation set can repeat prompts, and paying a
    # second local generation or Fireworks call for a byte-identical prompt
    # buys nothing. This is per-run working state only (nothing persisted or
    # baked into the image, which is what the no-caching rule prohibits).
    seen_prompts = {}   # prompt -> task_id whose answer to copy
    dup_of = {}         # task_id -> earlier task_id with the identical prompt

    def _flush_snapshot():
        snapshot = dict(answers)
        for dup_id, earlier_id in dup_of.items():
            snapshot[dup_id] = snapshot.get(earlier_id, "")
        write_results(OUTPUT_PATH, [{"task_id": tid, "answer": snapshot.get(tid, "")} for tid in all_task_ids])

    def _watchdog():
        time.sleep(_HARD_DEADLINE_SECONDS)
        print(f"[fatal] hard deadline ({_HARD_DEADLINE_SECONDS:.0f}s) reached - "
              f"force-flushing {len(answers)}/{len(all_task_ids)} answers and exiting", file=sys.stderr)
        try:
            _flush_snapshot()
        except Exception as exc:
            print(f"[fatal] watchdog flush failed: {exc}", file=sys.stderr)
        os._exit(0)  # a stuck call in the main thread can't be interrupted - kill the process instead

    threading.Thread(target=_watchdog, daemon=True).start()

    for task_id, task in normalized_tasks:
        # A single malformed task (missing/wrong-typed field, unexpected
        # content) must not lose every other task's already-computed
        # answer - isolate each task's processing so one bad entry
        # degrades to a missing answer for *that task only*, not a crash
        # that discards the whole batch via the outer handler.
        try:
            prompt = task["prompt"]

            earlier = seen_prompts.get(prompt)
            if earlier is not None:
                # Identical prompt already being handled - reuse its answer
                # (resolved after the Fireworks phase, since the earlier task
                # may itself still be waiting in a bucket at this point).
                dup_of[task_id] = earlier
                print(f"[timing] t={time.time()-started:.1f}s {task_id} deduplicated (same prompt as {earlier})", file=sys.stderr)
                continue
            seen_prompts[prompt] = task_id

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
                if _ENABLE_MATH_PAL and time.time() - started < _TIME_BUDGET_SECONDS:
                    pal_started = time.time()
                    pal_answer = local_llm.try_solve_math_word_problem_pal(prompt)
                    pal_elapsed = time.time() - pal_started
                    if pal_answer is not None:
                        answers[task_id] = pal_answer
                        print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) solved via local PAL in {pal_elapsed:.1f}s", file=sys.stderr)
                        continue
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) PAL not confident, in {pal_elapsed:.1f}s, falling to Fireworks", file=sys.stderr)
                elif _ENABLE_MATH_PAL:
                    print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) skipping PAL (time budget), falling to Fireworks", file=sys.stderr)

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

            if category in LOCAL_LLM_CATEGORIES and time.time() - started < _TIME_BUDGET_SECONDS:
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
            elif category in LOCAL_LLM_CATEGORIES:
                print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) skipping local attempt (time budget), falling to Fireworks", file=sys.stderr)

            buckets[category].append((task_id, prompt))
        except Exception as exc:
            print(f"[warn] failed to process task {task_id}: {exc}", file=sys.stderr)
            # Leave it out of both answers and buckets - it gets an empty
            # string in the final results, exactly like an unresolved
            # Fireworks task, rather than aborting everything else.

    # Computed unconditionally, before the Fireworks try block, so a last-
    # resort local fallback below can still find each bucketed task's
    # category/prompt even if FireworksClient() itself never constructs
    # (e.g. a missing/misnamed env var) or answer_all() fails entirely.
    category_by_task = {tid: cat for cat, items in buckets.items() for tid, _ in items}
    prompt_by_task = {tid: p for items in buckets.values() for tid, p in items}

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

    # Last-resort tier: an empty string is a guaranteed miss on the accuracy
    # gate, while an ungated local answer at least has a real chance of
    # being right. This only fires for tasks that reached this point with
    # no answer at all - i.e. Fireworks was unreachable/misconfigured, or a
    # specific model group failed entirely (see fireworks_client.py's
    # as_completed fix for why other groups no longer get taken down with
    # it) - never overrides an answer Fireworks or local already provided.
    for task_id, category in category_by_task.items():
        if answers.get(task_id):
            continue
        prompt = prompt_by_task.get(task_id)
        if prompt is None:
            continue
        try:
            fallback = local_llm.answer(category, prompt)
        except Exception as exc:
            print(f"[warn] last-resort local answer failed for {task_id}: {exc}", file=sys.stderr)
            fallback = None
        if fallback is not None:
            if category in ("code_debug", "code_gen"):
                fallback = strip_code_fence(fallback)
            answers[task_id] = fallback
            print(f"[timing] t={time.time()-started:.1f}s {task_id} ({category}) recovered via last-resort local answer", file=sys.stderr)

    for dup_id, earlier_id in dup_of.items():
        answers[dup_id] = answers.get(earlier_id, "")

    results = [{"task_id": tid, "answer": answers.get(tid, "")} for tid in all_task_ids]
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
                {"task_id": tid, "answer": ""} for tid, _ in normalize_tasks(tasks)
            ])
        except Exception:
            write_results(OUTPUT_PATH, [])
        sys.exit(1)
