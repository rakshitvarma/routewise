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
    r"who is the|who sits|who owns|who likes|which one|which box|deduce|puzzle|"
    r"either .* or|cannot both|immediately to the (left|right)|leftmost|rightmost|"
    r"you may (pick|choose|select)|labels? (is|are|were) wrong)\b", re.I
)
# Deliberately excludes bare "positive"/"negative"/"neutral" next to
# "number(s)"/"integer(s)"/"value(s)" - those show up in math/code prompts
# ("all-negative numbers") and would otherwise falsely steal the category
# from a stronger, more specific signal like a code block.
_SENTIMENT_WORDS = re.compile(
    r"\bsentiment\b|\bfeel about\b|\bopinion of\b|\breview\b|"
    r"\b(positive|negative|neutral)\b(?!\s*(numbers?|integers?|values?))",
    re.I,
)
_SUMMARY_WORDS = re.compile(
    r"\b(summar(y|ise|ize)|condense|tl;dr|in one sentence|in \d+ (words|sentences)|shorten)\b", re.I
)
_NER_WORDS = re.compile(
    r"\bnamed entit(y|ies)\b|"
    r"\b(extract|identify|list)\b.{0,40}\b(people|person|organi[sz]ations?|locations?|dates?|entities)\b",
    re.I,
)


def classify(prompt: str) -> str:
    """Return one of CATEGORIES for the given task prompt."""
    text = prompt.strip()

    # Code detection runs first: a code block plus a bug/fix word is a much
    # more specific, unambiguous signal than a stray keyword match from
    # another category (e.g. "negative" appearing inside "all-negative
    # numbers" in a bug report should not be able to steal this to sentiment).
    if _CODE_BLOCK_RE.search(text) and _BUG_WORDS.search(text):
        return "code_debug"
    if _GEN_WORDS.search(text) or (_CODE_BLOCK_RE.search(text) and not _BUG_WORDS.search(text)):
        return "code_gen"
    if _NER_WORDS.search(text):
        return "ner"
    if _SUMMARY_WORDS.search(text):
        return "summarization"
    if _SENTIMENT_WORDS.search(text):
        return "sentiment"
    if _LOGIC_WORDS.search(text):
        return "logic"
    if _MATH_WORDS.search(text) or _MATH_EXPR_RE.search(text):
        return "math"
    return "factual"
