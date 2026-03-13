# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

QA Studio is a **self-improving response generation platform**. The goal is to automatically improve the **system prompt** used to generate LLM responses, creating a closed-loop cycle:

1. **Evaluate** existing LLM-generated responses (from UltraFeedback dataset) using a judge with rubric-based scoring
2. **Analyze failure patterns** — what goes wrong in the responses (over_verbose, hallucination, logic_error, etc.)
3. **Improve the system/generation prompt** — use pattern analysis to suggest a better system prompt that fixes the identified issues
4. **A/B test** — generate NEW responses with the improved system prompt, evaluate them, compare with baseline
5. **Deploy** — if the improved system prompt produces better responses, promote it to production

**The thing being improved is the SYSTEM PROMPT (response generation prompt), NOT the judge/evaluation prompt.** The judge is a measurement tool — it stays fixed. The system prompt is what gets better over time.

### What each component does

| Component | Role | What it does NOT do |
|---|---|---|
| **Judge/Evaluator** (`pipeline.py`) | Measures response quality (fixed rubric) | Does NOT get improved by the cycle |
| **Pattern Analyzer** (`pattern_analyzer.py`) | Finds recurring response quality issues | — |
| **Prompt Suggester** (`prompt_suggester.py`) | Suggests improvements to the **system prompt** | Must NOT target the judge prompt |
| **A/B Experiment** (`experiment.py`) | Generates responses with old vs new system prompt, evaluates both | — |
| **Approval Workflow** (`approval_workflow.py`) | Deploys improved **system prompt** to Langfuse | Must NOT deploy judge prompts |

### Dataset

Primary dataset: UltraFeedback — 4 candidate responses per question from different LLMs (alpaca-7b, vicuna, starchat, falcon) with GPT-4 annotations as ground truth. These are pre-existing LLM-generated responses used for the initial evaluation phase.

## Project Overview (Technical)

## Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Commands

### Docker (PostgreSQL)
```bash
docker compose up -d                           # start Postgres on port 5433
docker compose ps                              # verify healthy
docker compose down                            # stop
```

### Backend (Python/FastAPI)
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000     # dev server
pytest                                         # all tests (uses SQLite)
pytest tests/test_mock_provider.py -v          # single test file
pytest --cov=app                               # with coverage
python scripts/migrate_to_postgres.py          # one-time SQLite → Postgres migration
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev       # dev server on port 3001
npm run build     # production build
npm run lint      # ESLint
```

### Data Preparation & Ingestion
```bash
cd backend

# Prepare UltraFeedback dataset (4 model responses per instruction, with GPT-4 annotations)
python scripts/prepare_ultrafeedback.py --dev-size 2000 --test-size 500 --ab-size 300

# Ingest data
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./sample_data/ultrafeedback_dev.jsonl", "split": "dev"}'

# Run evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate/run" \
  -H "Content-Type: application/json" \
  -d '{"dataset_split": "dev", "prompt_version": "v1", "model_version": "mock", "docs_version": "v1"}'
```

## Architecture

**Backend** (`backend/app/`): FastAPI app with PostgreSQL (SQLAlchemy). Routes split across `api/endpoints/`. Config via `pydantic-settings` from `.env`. SQLite still used for tests via `conftest.py`.

**Evaluation Pipeline** (`services/pipeline.py`): Per-item flow: prepare → mask PII (regex) → classify (taxonomy) → RAG retrieve (masked) → judge (masked, gates + scores) → sampling decision (human queue). PII-masked text is used for all downstream steps (RAG, judge) — raw text never leaves the masking boundary. The judge is a fixed measurement tool — it does NOT get modified by the self-improvement cycle.

**Self-Improvement Cycle**: The cycle improves the **system/generation prompt**, not the judge. It refreshes dataset evaluations when coverage is stale or missing, analyzes failure tags from those dataset evaluations, proposes a better system prompt, generates new baseline/candidate responses for A/B judging, then routes the winning prompt through approval and deployment.

**LLM Providers** (`providers/`): Strategy pattern via `factory.py`. `MockProvider` for testing (keyword-based classification, prompt-aware generation, rule-based judging), `OpenAIProvider` for production. Set via `LLM_PROVIDER` env var.

**RAG** (`rag/`): `RAGIndexer` builds FAISS index from markdown docs in `docs/` (policies, help center, rubrics). `RAGRetriever` queries the index. Mock embeddings when no OpenAI key.

**Taxonomy** (`core/taxonomy.py`): 8 Q/A task labels (reasoning, math, classification, summarization, extraction, creative_writing, coding, open_qa) with required slots per label. 12 failure tags.

**Rubric** (`core/rubric.py`): 2 gates (factual_safety, hallucination) + 4 scores (instruction_following, reasoning_quality, completeness, clarity, 1-5 scale). Sampling rules drive human queue routing.

**Frontend** (`frontend/`): Next.js 14 with pages for items, experiments, and human review queue. API client in `src/lib/api.ts` points to backend port 8000.

**Instrumentation**: Langfuse tracing with graceful fallback — works fully offline without Langfuse keys.

**n8n Integration**: `n8n_workflow.json` at root — schedule-driven workflow (ingest → evaluate → check queue → Slack alert).

## Key Data Files

- `sample_data/ultrafeedback_{dev,test,ab_eval}.jsonl` — UltraFeedback splits (scenario_id, candidate_source, GPT-4 annotation metadata)
- `docs/rubrics/` — scoring rubric documents (primary RAG source)
- `docs/policies/` and `docs/help_center/` — supplementary docs (fallback when rubrics absent)

### Ingest Format Support

The ingest parser (`_parse_and_ingest`) supports three JSONL shapes:
- **Standard**: `question`, `response`, `system_prompt` (also `instruction`/`output`/`query`/`answer`)
- **Legacy ticket**: `conversation[0].content` → question, `candidate_response` → response
- **UltraFeedback extras**: `scenario_id` (groups candidates), `candidate_source` (model name), `metadata.annotations` (GPT-4 scores as ground truth)

Duplicate rows are skipped on ingest when `external_id` matches an existing record.

### Key Behaviors

- **evaluate/run** skips items that already have an evaluation for the same (prompt_version, model_version, docs_version) triple.
- **Pattern analysis** deduplicates by using only the latest dataset evaluation per item to avoid count inflation and excludes experiment evaluations from the prompt-improvement corpus.
- **Novel tag routing** persists across pipeline instances (process-level set) so the first occurrence is truly novel.
- **Mock provider** generates prompt-sensitive responses. A/B deltas in mock mode should come from response changes caused by the system prompt, not from synthetic judge bias.

### Self-Improvement Invariants

- **The judge prompt is FIXED.** The self-improvement cycle must never modify, suggest changes to, or deploy changes to the judge/evaluation prompt. The judge is a measurement instrument.
- **The system prompt is what improves.** Pattern analysis → suggestion → A/B test → deploy all target the system/generation prompt.
- **A/B testing requires generation.** To test an improved system prompt, the experiment must generate new responses with it (not just re-evaluate existing responses with a different judge prompt).
- **Langfuse prompt registry** stores versioned system prompts (not judge prompts). The `production` label points to the current best system prompt.

## Environment

Backend requires `.env` in `backend/` (copy from `.env.example`). Key settings:

- `DATABASE_URL=postgresql://qa_studio:qa_studio@localhost:5433/qa_studio` (requires `docker compose up -d`)
- `LLM_PROVIDER=mock` for local dev (no API keys needed)
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY` for production
- Langfuse keys are optional

Frontend requires `.env.local` in `frontend/` (copy from `.env.example`).

## Efficiency Rules

- **Don't re-read files already in context.** If you just read a file in this conversation, use the cached content.
- **Use haiku for simple subagent tasks** (file searches, running tests, quick lookups). Reserve opus for complex multi-step reasoning.
- **Skip preamble.** No "Sure, I'd be happy to help!" — go straight to the answer or action.
- **Batch parallel tool calls.** If reading 3 files or running independent commands, do them all in one message.
- **n8n workflows** are managed via the n8n-mcp tools (list, get, update, test). The live instance is at localhost:5678. Key workflow IDs: Evaluation Pipeline = `k8IANqznWP1Sq4mc`, Self-Improvement Cycle = `BhJkrpSlZwnDVVHY`.
