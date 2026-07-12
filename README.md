# Fairwind

**A hybrid token-efficient routing agent** — built for AMD Developer
Hackathon Act II, Track 1 (Hybrid Token-Efficient Routing Agent).

A Dockerized agent that answers all 8 Track 1 task categories (factual
knowledge, math reasoning, sentiment classification, summarisation, NER,
code debugging, logical/deductive reasoning, code generation) while
minimizing tokens billed through Fireworks AI. Fairwind classifies and
solves what it can for free — via deterministic code or two bundled local
models — and only pays for Fireworks inference on the tasks that genuinely
need it. The same core idea generalizes beyond this hackathon to any team
routing requests across multiple LLM providers (local or hosted) to cut
inference spend without sacrificing accuracy.

**Live demo:** https://fairwind-frontend-841323378171.us-central1.run.app
**Docker image:** `ghcr.io/rakshitvarma/fairwind:latest`

## Architecture

```
tasks.json -> classifier (regex, 0 tokens)
           -> math?                       -> deterministic solver (0 tokens, confident only) -> Fireworks fallback
           -> sentiment/ner/factual/       -> local Qwen2.5-1.5B-Instruct (0 tokens, sanity-checked)
              summarization?
           -> code_debug/code_gen?        -> local Qwen2.5-Coder-1.5B-Instruct (0 tokens, syntax-verified)
           -> logic?                      -> Fireworks, self-consistent (3-call majority vote)
           -> anything the above rejected -> merged per-model Fireworks batch
                                                                            -> results.json
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
3. **One bundled local model** (`router/local_llm.py`, Qwen3-4B-Instruct-2507,
   ~2.5GB GGUF, CPU inference via `llama-cpp-python`) answers six of the
   eight categories entirely for free: factual, sentiment, NER,
   summarisation, code_debug, code_gen. Every local answer is gated behind
   a self-consistency confidence check (`answer_confident`) rather than a
   fixed sanity check: the deterministic sample has to agree with 2 more
   stochastic samples, and that agreement signal is looked up against a
   calibration curve fitted from real labelled observations
   (`router/calibrate.py`) before being trusted. Anything that fails the
   gate falls through to Fireworks exactly like an unsolved math
   expression does — the local tier is pure upside, never a hard
   dependency.
4. **Math word problems and logic puzzles stay on Fireworks** rather than
   the local models, deliberately. Math word problems already had a
   reliable Fireworks track record; logic puzzles get **dedicated
   self-consistency** (3 independent calls, majority vote) after
   empirically reproducing a real failure — the same seating-arrangement
   puzzle, called with identical inputs, non-deterministically flipped
   between a correct answer and one that directly contradicted its own
   stated constraint. A 1.5B local model is the *least* likely component
   to hold up on this category under randomized rephrasing, so it's kept
   off the local path entirely.
5. **Merged Fireworks batches** (`router/fireworks_client.py`) for
   whatever local didn't resolve: grouped by *model* (not by category) and
   answered in as few calls as possible, with per-task style instructions
   embedded inline. Mirrors the top leaderboard entry's own documented
   technique of merging categories into one direct batch.
6. Responses are parsed as strict JSON; a missing key for any task in a
   merged batch, or a degenerate/repeating response, triggers one
   corrective single-task call for just that task rather than submitting
   an empty or wrong answer for the whole batch. A non-zero exit or
   malformed `results.json` scores zero, which is worse than a few extra
   tokens spent recovering.

### Local model verification

Per the org's own guidance, local models are a fully valid scoring
strategy (answers count toward accuracy; only Fireworks-routed tokens
count toward the token score) — but final scoring uses **randomized
prompt variants**, so a model that just memorizes the visible examples is
worthless. `router/eval_local.py` is a standalone eval harness (run via
`docker run --rm --entrypoint python <image> -m router.eval_local`) using
prompts with deliberately different phrasing/domains than
`sample_input/*.json`, to catch overfitting rather than reward it. Result
on the last run: **24/24 correct** across all six locally-routed
categories on held-out phrasing. On the 17-task diverse test set end to
end, **13 of 17 tasks were answered entirely locally for zero tokens**,
with only math word problems and logic puzzles reaching Fireworks (2,149
tokens total, down from 3,336 in an earlier Fireworks-only version of this
same test set).

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

**https://fairwind-frontend-841323378171.us-central1.run.app**

A React/Vite/Tailwind frontend calling a FastAPI backend
(`backend/main.py`), both deployed on Google Cloud Run. Pick an example
task per category (or write your own) and see the classification, routing
decision (local vs. which Fireworks model), token cost, and answer, plus a
running session history. The backend wraps the exact same `router/`
package used by the submitted Docker image, so it reflects real behavior
rather than a reimplementation.

Run it locally:

```bash
# backend
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8080

# frontend
cd frontend && npm install
echo "VITE_API_URL=http://localhost:8080" > .env.local
npm run dev
```

## Running locally

```bash
cp .env.example .env   # fill in FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
docker build -t fairwind .
docker run --rm \
  -v "$(pwd)/sample_input:/input" \
  -v "$(pwd)/output:/output" \
  --env-file .env \
  fairwind
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
