"""Zero-token deterministic solvers.

These only answer when they can be *sure* they're correct — anything
ambiguous falls through to the Fireworks batch. Wrong free answers are
worse than costing a few tokens, since a single accuracy-gate failure
takes the whole submission off the leaderboard.
"""
import ast
import itertools
import operator
import re
from typing import Optional

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_EXPR_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*[-+*/^%]\s*[-+]?\d+(?:\.\d+)?)+")


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _format_result(result):
    if isinstance(result, float):
        result = round(result, 6)  # clear binary float noise (e.g. 46.800000000000004)
        if result.is_integer():
            result = int(result)
    return str(result)


def try_solve_math(prompt: str) -> Optional[str]:
    """Solve prompts that are *just* a bare arithmetic expression.

    Deliberately conservative: word problems ("if a shirt costs $50 and is
    discounted 20%...") need real language understanding to extract the
    right operation, so they're left to Fireworks rather than risking a
    wrong local answer.
    """
    text = prompt.strip().rstrip("?.! ")
    # Strip common lead-ins like "What is" / "Calculate" / "Solve:"
    text = re.sub(r"^(what is|calculate|compute|solve|evaluate)\s*:?\s*", "", text, flags=re.I)

    match = _EXPR_RE.fullmatch(text.replace("^", "**").replace(" ", ""))
    if not match:
        return None
    return evaluate_expression(text)


def evaluate_expression(expr: str) -> Optional[str]:
    """Safely evaluate a bare arithmetic expression string (numbers and
    +-*/^%() only). Used both for prompts that are already a bare
    expression, and for expressions a local model extracted from a word
    problem - in the latter case the model's output is trusted for
    *extraction only*; the actual arithmetic is still done here, in code,
    so a small model's well-known weakness at doing the arithmetic itself
    can't produce a wrong answer.
    """
    text = (expr or "").strip()
    # Models sometimes append "= <their own computed value>" despite being
    # told not to - discard that and evaluate only the expression part
    # ourselves, since trusting the model's own arithmetic is exactly what
    # this function exists to avoid.
    text = text.split("=")[0]
    text = text.replace("^", "**").replace(",", "").replace(" ", "").rstrip("?.!")
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval")
        result = _safe_eval(tree.body)
    except Exception:
        return None
    return _format_result(result)


# Code debugging / generation deliberately have no local deterministic
# *answering* path: verifying arbitrary submitted code needs sandboxed
# execution and inferred test cases, which is high-effort and unreliable to
# build correctly under time pressure. Both categories are always routed to
# the Fireworks batch. We do, however, cheaply verify the syntax of what
# comes back (zero tokens) so an obviously broken answer can trigger one
# corrective call instead of silently failing the accuracy gate.

_PY_HINT_RE = re.compile(r"\bdef \w+\(|\bpython\b|\breturn\b", re.I)
_OTHER_LANG_RE = re.compile(
    r"\bpublic (class|static)\b|#include|\bfunction\s*\w*\s*\(|\bconsole\.log\b|"
    r"\bSystem\.out\b|\bvar \w+\s*=|\blet \w+\s*=", re.I
)


def looks_like_python(text: str) -> bool:
    return bool(_PY_HINT_RE.search(text)) and not _OTHER_LANG_RE.search(text)


def strip_code_fence(code: str) -> str:
    """Remove a leading/trailing markdown code fence. The model is asked
    for "code only," but often wraps it in ```python ... ``` anyway - if
    the returned answer text still has that fence, anything that tries to
    exec() it literally hits a SyntaxError on the fence markers themselves,
    even though the code inside is fine."""
    return re.sub(r"^```(python)?\s*|\s*```$", "", code.strip(), flags=re.I)


# Deterministic solver for one specific, common logic-puzzle archetype:
# N named entities in a row of N positions, with left/right/end/adjacency
# clues, asking who ends up at some position. Every "logic" task currently
# costs a full Fireworks self-consistency call (router/fireworks_client.py
# _answer_logic, 3x calls) regardless of how simple the puzzle is - a
# competitor's writeup describes solving this exact archetype (and
# assignment/ordering puzzles) deterministically for zero tokens, brute-
# forcing every permutation and only committing when *exactly one*
# satisfies every clue. Scoped narrowly (this one archetype, not every
# logic-puzzle shape) since a wrong parse that silently commits to a wrong
# answer is worse than the Fireworks tokens it would have cost - anything
# that doesn't cleanly match returns None and falls through as before.
_ENTITY_LIST_RE = re.compile(
    r"[-–—]\s*((?:[A-Z][\w'.]*\s*,\s*)*(?:[A-Z][\w'.]*\s*,?\s*and\s+)?[A-Z][\w'.]*)\s*[-–—]"
)


def _split_entity_list(raw: str) -> list:
    parts = re.split(r",\s*and\s+|,\s*|\s+and\s+", raw)
    return [p.strip() for p in parts if p.strip()]


def _extract_row_entities(prompt: str) -> Optional[list]:
    m = _ENTITY_LIST_RE.search(prompt)
    if not m:
        return None
    names = _split_entity_list(m.group(1))
    return names if len(names) >= 3 else None


def _entity_alternation(names: list) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


def _parse_row_constraints(prompt: str, names: list):
    """Returns a list of constraint-check functions (perm -> bool), where
    perm is a tuple of names ordered leftmost..rightmost, or None if any
    clue sentence doesn't match a recognized pattern (abstain)."""
    ents = _entity_alternation(names)
    n = len(names)
    constraints = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prompt) if s.strip()]

    for sent in sentences:
        if "?" in sent:
            continue  # the question, handled separately

        m = re.search(rf"({ents})\s+(?:is|sits?)\s+not at either end", sent, re.I)
        if m:
            name = m.group(1)
            constraints.append(lambda p, name=name: p[0] != name and p[-1] != name)
            continue

        m = re.search(rf"({ents})\s+sits?\s+immediately to the left of\s+({ents})", sent, re.I)
        if m:
            x, y = m.group(1), m.group(2)
            constraints.append(lambda p, x=x, y=y: p.index(x) == p.index(y) - 1)
            continue

        m = re.search(rf"({ents})\s+sits?\s+immediately to the right of\s+({ents})", sent, re.I)
        if m:
            x, y = m.group(1), m.group(2)
            constraints.append(lambda p, x=x, y=y: p.index(x) == p.index(y) + 1)
            continue

        m = re.search(rf"({ents})\s+sits?\s+at the leftmost\s+\w+", sent, re.I)
        if m:
            name = m.group(1)
            constraints.append(lambda p, name=name: p[0] == name)
            continue

        m = re.search(rf"({ents})\s+sits?\s+at the rightmost\s+\w+", sent, re.I)
        if m:
            name = m.group(1)
            constraints.append(lambda p, name=name: p[-1] == name)
            continue

        m = re.search(rf"({ents})\s+is not next to\s+({ents})", sent, re.I)
        if m:
            x, y = m.group(1), m.group(2)
            constraints.append(lambda p, x=x, y=y: abs(p.index(x) - p.index(y)) != 1)
            continue

        m = re.search(rf"({ents})\s+is next to\s+({ents})", sent, re.I)
        if m:
            x, y = m.group(1), m.group(2)
            constraints.append(lambda p, x=x, y=y: abs(p.index(x) - p.index(y)) == 1)
            continue

        m = re.search(rf"({ents})\s+sits?\s+to the left of\s+({ents})", sent, re.I)
        if m:
            x, y = m.group(1), m.group(2)
            constraints.append(lambda p, x=x, y=y: p.index(x) < p.index(y))
            continue

        m = re.search(rf"({ents})\s+sits?\s+to the right of\s+({ents})", sent, re.I)
        if m:
            x, y = m.group(1), m.group(2)
            constraints.append(lambda p, x=x, y=y: p.index(x) > p.index(y))
            continue

        if _ENTITY_LIST_RE.search(sent):
            continue  # the introductory sentence naming the entities

        # Some other sentence that doesn't match any known clue shape -
        # can't be sure we aren't ignoring a real constraint, so abstain
        # entirely rather than risk a wrong answer.
        return None

    return constraints


def _answer_row_question(prompt: str, names: list, solution: tuple) -> Optional[str]:
    ents = _entity_alternation(names)
    q_match = re.search(r"[^.!?]*\?", prompt)
    if not q_match:
        return None
    question = q_match.group(0)

    if re.search(r"who sits? at the rightmost", question, re.I):
        return solution[-1]
    if re.search(r"who sits? at the leftmost", question, re.I):
        return solution[0]
    m = re.search(rf"who sits?\s+immediately to the left of\s+({ents})", question, re.I)
    if m:
        idx = solution.index(m.group(1)) - 1
        return solution[idx] if 0 <= idx < len(solution) else None
    m = re.search(rf"who sits?\s+immediately to the right of\s+({ents})", question, re.I)
    if m:
        idx = solution.index(m.group(1)) + 1
        return solution[idx] if 0 <= idx < len(solution) else None
    return None


def try_solve_logic_row(prompt: str) -> Optional[str]:
    """N entities in a row of N positions with left/right/end/adjacency
    clues (e.g. "Four coworkers - Dev, Priya, Sam, and Lee - sit in a row
    of four desks. Dev is not at either end. ... Who sits at the rightmost
    desk?"). Brute-forces every permutation of entities to positions,
    keeping only those consistent with every parsed clue - answers only if
    exactly one permutation survives, otherwise returns None (falls
    through to Fireworks, same as any other logic puzzle today)."""
    names = _extract_row_entities(prompt)
    if names is None:
        return None

    constraints = _parse_row_constraints(prompt, names)
    if constraints is None or not constraints:
        return None

    solutions = [
        perm for perm in itertools.permutations(names)
        if all(c(perm) for c in constraints)
    ]
    if len(solutions) != 1:
        return None

    return _answer_row_question(prompt, names, solutions[0])


def python_syntax_error(code: str) -> Optional[str]:
    """Return None if `code` parses as valid Python, else the error message."""
    try:
        ast.parse(strip_code_fence(code))
        return None
    except SyntaxError as exc:
        return str(exc)


# Named entity extraction via spaCy's statistical NER model instead of the
# general local LLM: a purpose-built model for this exact task, zero tokens
# either way, competitor evidence (AMD hackathon writeups) suggests it's
# both faster and at least as reliable as asking a general-purpose LLM. Kept
# behind ENABLE_SPACY_NER (main.py) rather than replacing the existing
# LLM+Fireworks path outright: the current pipeline already calibrates to
# ~97% on ner, and spaCy's accuracy on this exact task/prompt distribution
# hasn't been validated yet - safest to compare before trusting it blindly.
_spacy_nlp = None
_spacy_load_failed = False

_SPACY_LABEL_MAP = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "DATE": "Date",
}
_LABEL_ORDER = ["Person", "Organization", "Location", "Date"]


def _get_spacy_nlp():
    global _spacy_nlp, _spacy_load_failed
    if _spacy_nlp is not None or _spacy_load_failed:
        return _spacy_nlp
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_md")
    except Exception:
        _spacy_load_failed = True
        _spacy_nlp = None
    return _spacy_nlp


def try_solve_ner_spacy(prompt: str) -> Optional[str]:
    """Extract named entities with spaCy. Returns None (falls through to the
    LLM/Fireworks path) if spaCy isn't available or finds nothing of the
    requested types - never returns a confident-looking "no entities found"
    that might just be a miss."""
    nlp = _get_spacy_nlp()
    if nlp is None:
        return None
    doc = nlp(prompt)
    grouped: dict = {}
    for ent in doc.ents:
        label = _SPACY_LABEL_MAP.get(ent.label_)
        if label is None:
            continue
        bucket = grouped.setdefault(label, [])
        if ent.text not in bucket:
            bucket.append(ent.text)
    if not grouped:
        return None
    lines = [f"{label}: {', '.join(grouped[label])}" for label in _LABEL_ORDER if label in grouped]
    return "\n".join(lines)
