# RouteWise

**A hybrid token-efficient routing agent** — built for AMD Developer
Hackathon Act II, Track 1 (Hybrid Token-Efficient Routing Agent).

A Dockerized agent that answers all 8 Track 1 task categories (factual
knowledge, math reasoning, sentiment classification, summarisation, NER,
code debugging, logical/deductive reasoning, code generation) while
minimizing tokens billed through Fireworks AI. RouteWise classifies and
solves what it can for free, and only pays for Fireworks inference on the
tasks that genuinely need it — the same core idea generalizes beyond this
hackathon to any team routing requests across multiple LLM providers to
cut inference spend without sacrificing accuracy.

## Architecture

```
tasks.json -> classifier (regex, 0 tokens) -> math? -> deterministic solver (0 tokens, confident only)
                                            -> otherwise -> batched per-category Fireworks call -> results.json
```

1. **Local zero-token classifier** (`router/classifier.py`) routes every
   task into one of the 8 categories using regex/keyword heuristics — no
   model call, no tokens.
2. **Deterministic math solver** (`router/solvers.py`) answers *only*
   prompts that reduce to a bare arithmetic expression (e.g. `12 * 8 + 4`),
   verified via a safe AST evaluator. Word problems requiring language
   understanding (discounts, percentages framed as text, projections) are
   deliberately left to Fireworks — guessing wrong locally costs an
   accuracy-gate failure, which is far more expensive than a few tokens.
3. **Batched Fireworks calls** (`router/fireworks_client.py`): every
   unresolved task is grouped by category and answered in a *single* API
   call per category (not per task), with `reasoning_effort: "none"` on
   reasoning-capable models, terse category-specific system prompts, and
   per-category `max_tokens` caps. Batching multiple tasks into one call is
   the single biggest token lever available under this scoring scheme.
4. Responses are parsed as a strict JSON array (`{task_id, answer}`); if a
   batch's output doesn't parse, the batch is retried once, then degrades
   to a best-effort answer rather than crashing — a non-zero exit or
   malformed `results.json` scores zero, which is worse than one weak
   answer.

## Model routing

Models are read from `ALLOWED_MODELS` at runtime (never hardcoded):

| Category | Model |
|---|---|
| code_debug, code_gen | `kimi-k2p7-code` (code-specialized) |
| everything else | `minimax-m3` |

**Gemma is deliberately not used in the default configuration.** Gemma
models on this account are billed per-hour while deployed (not per-token),
and return a 404 if not deployed at call time — given a $50 total credit
budget and no control over exactly when the evaluation harness invokes the
container, routing production traffic through an on-demand-billed model
risked either burning the budget keeping it warm, or a hard failure if it
wasn't deployed at evaluation time. Reliability and token-efficiency on the
core 8 categories were prioritized over the Gemma bonus. The routing logic
supports a Gemma path (`ENABLE_GEMMA_BONUS=true` routes `sentiment` to
`gemma-4-26b-a4b-it`, the cheapest of the three available Gemma variants)
for anyone who wants to enable it deliberately.

## Running locally

```bash
cp .env.example .env   # fill in FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
docker build -t routewise .
docker run --rm \
  -v "$(pwd)/sample_input:/input" \
  -v "$(pwd)/output:/output" \
  --env-file .env \
  routewise
cat output/results.json
```

## Building for submission (linux/amd64)

```bash
docker buildx build --platform linux/amd64 --tag <registry>/<image>:latest --push .
```

## Contract

- Reads tasks from `/input/tasks.json`: `[{"task_id": "...", "prompt": "..."}]`
- Writes `/output/results.json`: `[{"task_id": "...", "answer": "..."}]`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` from
  the environment only — no hardcoded keys or model IDs, no bundled `.env`.
- Exits 0 on success; always attempts to write valid (if degraded) JSON
  even on partial failure.
