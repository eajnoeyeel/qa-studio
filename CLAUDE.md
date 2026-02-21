# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QA Studio — automated quality assessment platform for Q/A responses. Evaluates LLM-generated answers through gate checks (factual safety, hallucination) and 4-dimension scoring (instruction_following, reasoning_quality, completeness, clarity), with RAG-based evidence retrieval, human-in-the-loop review, and A/B experimentation. Uses OpenOrca dataset format (system_prompt, question, response).

## Commands

### Backend (Python/FastAPI)
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000     # dev server
pytest                                         # all tests
pytest tests/test_mock_provider.py -v          # single test file
pytest --cov=app                               # with coverage
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
# Prepare OpenOrca dataset
cd backend
python scripts/prepare_openorca.py

# Ingest data
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./sample_data/items_dev.jsonl", "split": "dev"}'

# Run evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate/run" \
  -H "Content-Type: application/json" \
  -d '{"dataset_split": "dev", "prompt_version": "v1", "model_version": "mock", "docs_version": "v1"}'
```

## Architecture

**Backend** (`backend/app/`): FastAPI app with SQLite (SQLAlchemy). All routes in a single `api/routes.py`. Config via `pydantic-settings` from `.env`.

**Evaluation Pipeline** (`services/pipeline.py`): Per-item flow: prepare → mask PII (regex) → classify (taxonomy) → RAG retrieve → judge (gates + scores) → sampling decision (human queue).

**LLM Providers** (`providers/`): Strategy pattern via `factory.py`. `MockProvider` for testing (keyword-based), `OpenAIProvider` for production. Set via `LLM_PROVIDER` env var.

**RAG** (`rag/`): `RAGIndexer` builds FAISS index from markdown docs in `docs/` (policies, help center, rubrics). `RAGRetriever` queries the index. Mock embeddings when no OpenAI key.

**Taxonomy** (`core/taxonomy.py`): 8 Q/A task labels (reasoning, math, classification, summarization, extraction, creative_writing, coding, open_qa) with required slots per label. 12 failure tags.

**Rubric** (`core/rubric.py`): 2 gates (factual_safety, hallucination) + 4 scores (instruction_following, reasoning_quality, completeness, clarity, 1-5 scale). Sampling rules drive human queue routing.

**Frontend** (`frontend/`): Next.js 14 with pages for items, experiments, and human review queue. API client in `src/lib/api.ts` points to backend port 8000.

**Instrumentation**: Langfuse tracing with graceful fallback — works fully offline without Langfuse keys.

**n8n Integration**: `n8n_workflow.json` at root — schedule-driven workflow (ingest → evaluate → check queue → Slack alert).

## Key Data Files

- `sample_data/items_dev.jsonl` — dev split (OpenOrca format: system_prompt, question, response)
- `sample_data/items_test.jsonl` — test split
- `sample_data/items_ab_eval.jsonl` — A/B experiment split
- `docs/policies/` and `docs/help_center/` — RAG source documents (markdown with frontmatter)
- `docs/rubrics/` — scoring rubric documents per task type

## Environment

Backend requires `.env` in `backend/` (copy from `.env.example`). Key settings:
- `LLM_PROVIDER=mock` for local dev (no API keys needed)
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY` for production
- Langfuse keys are optional

Frontend requires `.env.local` in `frontend/` (copy from `.env.example`).
