"""Zero-token deterministic solvers.

These only answer when they can be *sure* they're correct — anything
ambiguous falls through to the Fireworks batch. Wrong free answers are
worse than costing a few tokens, since a single accuracy-gate failure
takes the whole submission off the leaderboard.
"""
import ast
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

    expr = text.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body if hasattr(tree, "body") else tree)
    except Exception:
        return None

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


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


def python_syntax_error(code: str) -> Optional[str]:
    """Return None if `code` parses as valid Python, else the error message."""
    stripped = re.sub(r"^```(python)?\s*|\s*```$", "", code.strip(), flags=re.I)
    try:
        ast.parse(stripped)
        return None
    except SyntaxError as exc:
        return str(exc)
