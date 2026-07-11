"""Zero-token local task classifier.

Routes each task to one of the 8 Track 1 categories using regex/keyword
heuristics. No model calls, no tokens spent.
"""
import re

CATEGORIES = (
    "math",
    "code_debug",
    "code_gen",
    "logic",
    "sentiment",
    "summarization",
    "ner",
    "factual",
)

_CODE_BLOCK_RE = re.compile(r"```|def \w+\(|function \w+\(|class \w+[:(]")
_BUG_WORDS = re.compile(r"\b(bug|fix|error|incorrect|debug|broken|wrong output|traceback)\b", re.I)
_GEN_WORDS = re.compile(
    r"\b(implement|write code|generate code)\b|\bwrite\b.{0,25}\bfunction\b", re.I
)
_MATH_WORDS = re.compile(
    r"\bpercent\w*\b|\bsum\b|\bproduct\b|\baverage\b|\bmean\b|\bmedian\b|\bprofit\w*\b|"
    r"\bdiscount\w*\b|\bratio\b|\binterest\b|\bmultipl\w*\b|\bdivid\w*\b|\bsubtract\w*\b|"
    r"\bhow many\b|\bhow much\b|\bcalculat\w*\b|\bprojection\w*\b|\btotal cost\b|"
    r"\bper (hour|day|week|month|year)\b",
    re.I,
)
_MATH_EXPR_RE = re.compile(r"\d+\s*[-+*/^%]\s*\d+")
_LOGIC_WORDS = re.compile(
    r"\b(if and only if|all of|none of|exactly one|must be true|constraint|"
    r"who is the|which one|deduce|puzzle|either .* or|cannot both)\b", re.I
)
_SENTIMENT_WORDS = re.compile(
    r"\b(sentiment|positive|negative|neutral|feel about|opinion of|review)\b", re.I
)
_SUMMARY_WORDS = re.compile(
    r"\b(summar(y|ise|ize)|condense|tl;dr|in one sentence|in \d+ words|shorten)\b", re.I
)
_NER_WORDS = re.compile(
    r"\b(named entit(y|ies)|extract (the )?(person|organi[sz]ation|location|date|entities)|"
    r"identify all (people|places|organizations|dates))\b", re.I
)


def classify(prompt: str) -> str:
    """Return one of CATEGORIES for the given task prompt."""
    text = prompt.strip()

    if _NER_WORDS.search(text):
        return "ner"
    if _SUMMARY_WORDS.search(text):
        return "summarization"
    if _SENTIMENT_WORDS.search(text):
        return "sentiment"
    if _CODE_BLOCK_RE.search(text) and _BUG_WORDS.search(text):
        return "code_debug"
    if _GEN_WORDS.search(text) or (_CODE_BLOCK_RE.search(text) and not _BUG_WORDS.search(text)):
        return "code_gen"
    if _LOGIC_WORDS.search(text):
        return "logic"
    if _MATH_WORDS.search(text) or _MATH_EXPR_RE.search(text):
        return "math"
    return "factual"
