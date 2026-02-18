# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SaaS CS QA Studio — automated quality assessment platform for customer service responses in SaaS collaboration tools (Notion-like). Evaluates CS agent replies through gate checks (policy safety, overclaim) and 4-dimension scoring, with RAG-based evidence retrieval, human-in-the-loop review, and A/B experimentation.

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

### Data Ingestion & Evaluation
```bash
# Ingest sample data
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./sample_data/tickets_dev.jsonl", "split": "dev"}'

# Run evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate/run" \
  -H "Content-Type: application/json" \
  -d '{"dataset_split": "dev", "prompt_version": "v1", "model_version": "mock", "docs_version": "v1"}'
```

## Architecture

**Backend** (`backend/app/`): FastAPI app with SQLite (SQLAlchemy). All routes in a single `api/routes.py`. Config via `pydantic-settings` from `.env`.

**Evaluation Pipeline** (`services/pipeline.py`): Per-ticket flow: normalize → mask PII (regex) → classify (taxonomy) → RAG retrieve → judge (gates + scores) → sampling decision (human queue).

**LLM Providers** (`providers/`): Strategy pattern via `factory.py`. `MockProvider` for testing (keyword-based), `OpenAIProvider` for production. Set via `LLM_PROVIDER` env var.

**RAG** (`rag/`): `RAGIndexer` builds FAISS index from markdown docs in `docs/` (policies + help center). `RAGRetriever` queries the index. Mock embeddings when no OpenAI key.

**Taxonomy** (`core/taxonomy.py`): 8 ticket labels (billing_seats, billing_refund, workspace_access, etc.) with required slots per label. 12 failure tags.

**Rubric** (`core/rubric.py`): 2 gates (policy_safety, overclaim) + 4 scores (understanding, info_strategy, actionability, communication, 1-5 scale). Sampling rules drive human queue routing.

**Frontend** (`frontend/`): Next.js 14 with pages for tickets, experiments, and human review queue. API client in `src/lib/api.ts` points to backend port 8000.

**Instrumentation**: Langfuse tracing with graceful fallback — works fully offline without Langfuse keys.

**n8n Integration**: `n8n_workflow.json` at root — schedule-driven workflow (ingest → evaluate → check queue → Slack alert).

## Key Data Files

- `sample_data/tickets_dev.jsonl` — dev split
- `sample_data/tickets_test.jsonl` — test split
- `sample_data/tickets_ab_eval.jsonl` — A/B experiment split
- `docs/policies/` and `docs/help_center/` — RAG source documents (markdown with frontmatter)

## Environment

Backend requires `.env` in `backend/` (copy from `.env.example`). Key settings:
- `LLM_PROVIDER=mock` for local dev (no API keys needed)
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY` for production
- Langfuse keys are optional

Frontend requires `.env.local` in `frontend/` (copy from `.env.example`).
