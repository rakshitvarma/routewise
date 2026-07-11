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

**Live demo:** https://routewise-amd.streamlit.app/
**Docker image:** `ghcr.io/rakshitvarma/routewise:latest`

## Architecture

```
tasks.json -> classifier (regex, 0 tokens) -> math? -> deterministic solver (0 tokens, confident only)
                                            -> logic? -> self-consistent (3-call majority vote)
                                            -> otherwise -> merged per-model Fireworks batch -> results.json
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
3. **Merged Fireworks batches** (`router/fireworks_client.py`): every
   unresolved task is grouped by *model* (not by category) and answered in
   as few calls as possible — e.g. factual, sentiment, summarisation, and
   NER all route to the same model and are merged into a single call, with
   per-task style instructions embedded inline. `code_debug`/`code_gen`
   merge into a second call. This mirrors the top leaderboard entry's own
   documented technique of merging categories into one direct batch, and
   cuts a typical 19-task run from ~8 calls down to 2-3.
4. **Logic puzzles get dedicated self-consistency** instead of joining the
   merged batch: 3 independent calls, majority vote on the stated
   conclusion. This was added after empirically reproducing a real failure
   — the same seating-arrangement puzzle, called with identical inputs,
   non-deterministically flipped between a correct answer and one that
   directly contradicted its own stated constraint. Since a single wrong
   answer here is one of only ~19 total scored tasks, the extra tokens are
   worth it.
5. Responses are parsed as strict JSON; a missing key for any task in a
   merged batch, or a degenerate/repeating response, triggers one
   corrective single-task call for just that task rather than submitting
   an empty or wrong answer for the whole batch. A non-zero exit or
   malformed `results.json` scores zero, which is worse than a few extra
   tokens spent recovering.

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

## Live demo

A small Streamlit app (`webapp/app.py`) lets you try the router live —
pick an example task per category (or write your own) and see the
classification, routing decision (local vs. which Fireworks model), token
cost, and answer, plus a running session history. It imports the exact
same `router/` package used by the submitted Docker image, so it reflects
real behavior rather than a reimplementation.

Run it locally with `streamlit run webapp/app.py` after copying
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` with real
Fireworks credentials (or export the same three env vars used by the
Docker image).

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
