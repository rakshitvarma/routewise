"""Fit a real calibration curve for the local-model confidence gate.

Runs each example in router/calibration_data.py through the same
self-consistency sampling the runtime gate uses, determines ground-truth
correctness (exact match for sentiment/NER, real code execution for
code_debug/code_gen, Fireworks-as-judge for factual/summarization), and
fits an isotonic regression per category mapping confidence signal ->
P(correct). Writes router/calibration.json, which local_llm.py loads at
runtime instead of the fixed heuristic thresholds.

Usage (inside the Docker image, where the models and llama-cpp-python
are already present):
    python -m router.calibrate
"""
import json
import os
import re
import sys

from router import calibration_data, local_llm
from router.solvers import strip_code_fence


def _isotonic_fit(pairs):
    """Pool-adjacent-violators: fit a monotonically non-decreasing curve
    through (signal, is_correct) points. Returns [(signal, p_correct), ...]
    sorted ascending by signal, safe to linear-interpolate between."""
    pairs = sorted(pairs, key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    ys = [float(p[1]) for p in pairs]
    n = len(ys)
    # blocks: [sum_y, count, start_idx, end_idx]
    blocks = [[ys[i], 1, i, i] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][0] / blocks[i][1]
        avg_next = blocks[i + 1][0] / blocks[i + 1][1]
        if avg_i > avg_next:
            merged = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1],
                      blocks[i][2], blocks[i + 1][3]]
            blocks[i:i + 2] = [merged]
            if i > 0:
                i -= 1
        else:
            i += 1
    fitted = [0.0] * n
    for s, c, start, end in blocks:
        avg = s / c
        for idx in range(start, end + 1):
            fitted[idx] = avg
    return list(zip(xs, fitted))


def _run_code_tests(code: str, func_name: str, tests: list) -> bool:
    code = strip_code_fence(code)
    namespace = {}
    try:
        exec(code, namespace)
    except Exception:
        return False
    if func_name not in namespace:
        return False
    try:
        for t in tests:
            exec(t, namespace)
        return True
    except Exception:
        return False


def _judge_via_fireworks(client, question: str, answer: str) -> bool:
    """LLM-as-judge fallback for categories with no fixed-string ground
    truth - mirrors the real evaluation's own methodology (an LLM judge),
    just run offline here to build calibration data instead of at
    submission time."""
    model = client.pick_model("factual")
    prompt = (
        f"Question: {question}\n\nAnswer given: {answer}\n\n"
        "Is this answer factually correct and does it directly address the "
        "question? Reply with exactly one word: YES or NO."
    )
    try:
        verdict = client._complete(model, "You are a strict grading assistant.", prompt, 5, json_mode=False)
    except Exception:
        return False
    return "yes" in verdict.strip().lower()


def _check_sentiment(example, primary) -> bool:
    label = local_llm._label_of("sentiment", primary)
    return label == example["expected_label"]


def _check_ner(example, primary) -> bool:
    text_low = primary.lower()
    return all(req.lower() in text_low for req in example["required"])


def _check_factual_grounded(example, primary) -> bool:
    """TriviaQA-style: correct if any accepted gold alias appears verbatim
    in the answer - stricter and more objective than an LLM judge, at the
    cost of penalizing a correct answer phrased in an unrecognized way."""
    text_low = primary.lower()
    return any(alias.lower() in text_low for alias in example["gold_aliases"])


def _check_summarization(example, primary) -> bool:
    if "expected_sentences" in example:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", primary.strip()) if s]
        return len(sentences) == example["expected_sentences"]
    if "expected_bullets" in example:
        bullets = [l for l in primary.splitlines() if l.strip().startswith(("-", "*", "•"))]
        if len(bullets) != example["expected_bullets"]:
            return False
        return all(len(re.sub(r"^[-*•]\s*", "", b).split()) <= example["max_bullet_words"] for b in bullets)
    return True


def main():
    from router.fireworks_client import FireworksClient
    client = FireworksClient()

    local_llm.preload()

    dataset = (
        [("sentiment", e, "exact") for e in calibration_data.SENTIMENT]
        + [("ner", e, "exact") for e in calibration_data.NER]
        + [("code_debug", e, "exec") for e in calibration_data.CODE_DEBUG]
        + [("code_gen", e, "exec") for e in calibration_data.CODE_GEN]
        + [("factual", e, "grounded") for e in calibration_data.FACTUAL_GROUNDED]
        + [("factual", e, "judged") for e in calibration_data.FACTUAL_JUDGED]
        + [("summarization", e, "exact") for e in calibration_data.SUMMARIZATION]
    )

    observations = {}  # category -> [(signal, is_correct), ...]
    for category, example, check_kind in dataset:
        prompt = example["prompt"]
        sampled = local_llm.sample_answers(category, prompt)
        if sampled is None:
            print(f"[calibrate] {category}: no sample at all for {prompt[:50]!r}", file=sys.stderr)
            continue
        primary, samples = sampled
        signal = local_llm.confidence_signal(category, primary, samples)
        if signal is None:
            continue

        if category == "sentiment":
            correct = _check_sentiment(example, primary)
        elif category == "ner":
            correct = _check_ner(example, primary)
        elif check_kind == "grounded":
            correct = _check_factual_grounded(example, primary)
        elif category in ("code_debug", "code_gen"):
            correct = _run_code_tests(primary, example["func_name"], example["tests"])
        elif category == "summarization":
            correct = _check_summarization(example, primary)
        else:  # factual
            correct = _judge_via_fireworks(client, prompt, primary)

        observations.setdefault(category, []).append((signal, correct))
        print(f"[calibrate] {category}: signal={signal:.2f} correct={correct} prompt={prompt[:60]!r}", file=sys.stderr)

    calibration = {}
    for category, obs in observations.items():
        n = len(obs)
        n_correct = sum(1 for _, c in obs if c)
        accuracy = n_correct / n if n else 0.0
        print(f"[calibrate] {category}: {n_correct}/{n} correct overall ({accuracy:.0%})", file=sys.stderr)
        if n < 4 or accuracy < 0.5:
            # Not enough data, or the category is unreliable regardless of
            # confidence signal - force every task in it to Fireworks
            # rather than fit a curve on too little/too weak evidence.
            calibration[category] = {"force_escalate": True, "n": n, "accuracy": accuracy}
            continue
        points = _isotonic_fit(obs)
        calibration[category] = {"force_escalate": False, "n": n, "accuracy": accuracy, "points": points}

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"[calibrate] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
