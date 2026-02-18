# SaaS CS QA Studio

**Automated Customer Service Quality Assessment for SaaS Collaboration Tools (Notion-like)**

A comprehensive platform for automating quality evaluation of customer service responses, with support for human-in-the-loop review, A/B experimentation, and RAG-based evidence retrieval.

## Features

- **Automated Evaluation**: Gate checks (policy safety, overclaim) + 4-dimension scoring (understanding, info_strategy, actionability, communication)
- **RAG-based Evidence**: LangChain-powered document retrieval for evidence-based overclaim detection
- **Human Review Queue**: Smart sampling for edge cases requiring human judgment
- **A/B Experiments**: Compare prompt/model versions with statistical analysis
- **Langfuse Observability**: Full tracing with graceful fallback when unavailable
- **n8n Integration**: HTTP endpoints designed for workflow orchestration

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) OpenAI API key for real embeddings/LLM
- (Optional) Langfuse account for production tracing

### Backend Setup

```bash
cd backend

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Initialize database and build RAG index
python -c "from app.main import app; print('Initialized')"

# Run server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Run development server
npm run dev
```

Access the UI at http://localhost:3001

## Environment Variables

Create `.env` file in `backend/` directory:

```env
# Application
APP_NAME=SaaS CS QA Studio
DEBUG=true
DATABASE_URL=sqlite:///./cs_qa_studio.db

# Langfuse (optional - graceful fallback if not set)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# LLM Provider (mock for testing, openai for production)
LLM_PROVIDER=mock
OPENAI_API_KEY=sk-...

# RAG Settings
DOCS_PATH=./docs
VECTOR_STORE_PATH=./vector_store
EMBEDDING_MODEL=text-embedding-3-small
```

## API Endpoints

### Ingest Batch

```bash
# Ingest tickets from JSONL file
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./sample_data/tickets_dev.jsonl", "split": "dev"}'
```

### Run Evaluation

```bash
curl -X POST "http://localhost:8000/api/v1/evaluate/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_split": "dev",
    "prompt_version": "v1",
    "model_version": "mock",
    "docs_version": "v1"
  }'
```

### A/B Experiment

```bash
curl -X POST "http://localhost:8000/api/v1/experiment/ab" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "prompt_v1_vs_v2",
    "dataset_split": "ab_eval",
    "docs_version": "v1",
    "config_a": {"prompt_version": "v1", "model_version": "mock"},
    "config_b": {"prompt_version": "v2", "model_version": "mock"}
  }'
```

### Get Human Queue

```bash
curl "http://localhost:8000/api/v1/human/queue?limit=20"
```

### Submit Human Review

```bash
curl -X POST "http://localhost:8000/api/v1/human/review" \
  -H "Content-Type: application/json" \
  -d '{
    "queue_item_id": "...",
    "evaluation_id": "...",
    "gold_label": "billing_seats",
    "gold_scores": {"understanding": 4, "actionability": 3},
    "notes": "Correct classification, but missing seat pricing info"
  }'
```

### Report Summary

```bash
curl "http://localhost:8000/api/v1/reports/summary?dataset_split=dev"
```

## Data Format

### Ticket JSONL Format

```json
{
  "id": "ticket_001",
  "conversation": [
    { "role": "user", "content": "I need to add 5 seats to our plan." },
    { "role": "user", "content": "We're onboarding new team members." }
  ],
  "candidate_response": "Happy to help! Go to Settings → Billing...",
  "metadata": { "category": "billing_seats" }
}
```

### Document Markdown Format

```markdown
---
doc_id: help_billing
title: Billing Management
version: v1
tags: [billing_seats, billing_refund]
source_url: https://help.example.com/billing
---

# Billing Management

Content here...
```

## n8n Workflow Integration

### Workflow Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌───────────────┐
│ Schedule    │────▶│ Ingest Batch │────▶│ Run Eval    │────▶│ Check Queue   │
│ Trigger     │     │ (weekly)     │     │ (daily)     │     │ (if needed)   │
└─────────────┘     └──────────────┘     └─────────────┘     └───────────────┘
                                                                      │
                                                                      ▼
                                              ┌─────────────────────────────────┐
                                              │ Slack Notification if           │
                                              │ gate_fail_rate > 10%            │
                                              └─────────────────────────────────┘
```

### n8n HTTP Node Configurations

#### 1. Ingest Batch Node

```
Method: POST
URL: http://localhost:8000/api/v1/ingest/batch
Body (JSON):
{
  "file_path": "/data/tickets_latest.jsonl",
  "split": "dev"
}
```

#### 2. Run Evaluation Node

```
Method: POST
URL: http://localhost:8000/api/v1/evaluate/run
Body (JSON):
{
  "dataset_split": "dev",
  "prompt_version": "v1",
  "model_version": "mock",
  "docs_version": "v1",
  "sampling_config": {
    "gate_fail_to_human": true,
    "low_score_threshold": 2
  }
}
```

#### 3. A/B Experiment Node

```
Method: POST
URL: http://localhost:8000/api/v1/experiment/ab
Body (JSON):
{
  "name": "weekly_ab_{{ $now.format('YYYY-MM-DD') }}",
  "dataset_split": "ab_eval",
  "docs_version": "v1",
  "config_a": {
    "prompt_version": "v1",
    "model_version": "mock"
  },
  "config_b": {
    "prompt_version": "v2",
    "model_version": "mock"
  }
}
```

#### 4. Get Summary Report Node

```
Method: GET
URL: http://localhost:8000/api/v1/reports/summary?dataset_split=dev
```

#### 5. Check Human Queue Node

```
Method: GET
URL: http://localhost:8000/api/v1/human/queue?limit=50
```

### Sample n8n Workflow JSON

See `n8n_workflow.json` in the repository root for a complete importable workflow.

## Evaluation Rubric

### Gates (Pass/Fail)

| Gate            | Description                                              |
| --------------- | -------------------------------------------------------- |
| `policy_safety` | No PII exposure, no security bypass, proper verification |
| `overclaim`     | All claims supported by documentation, no false promises |

### Scores (1-5)

| Score           | Description                                       |
| --------------- | ------------------------------------------------- |
| `understanding` | Correctly identifies customer's issue and context |
| `info_strategy` | Asks appropriate clarifying questions             |
| `actionability` | Provides clear, executable next steps             |
| `communication` | Professional, empathetic tone                     |

### Failure Tags

`intent_miss`, `missing_slot`, `no_next_step`, `policy_pii`, `overclaim`,
`escalation_needed`, `tool_needed`, `tone_issue`, `contradiction`,
`sso_admin_required`, `permission_model_mismatch`, `billing_context_missing`

## Taxonomy

| Label                | Description                               |
| -------------------- | ----------------------------------------- |
| `billing_seats`      | Seat-based pricing, adding/removing users |
| `billing_refund`     | Refund requests, billing disputes         |
| `workspace_access`   | Workspace access issues, invitations      |
| `permission_sharing` | Permission settings, sharing controls     |
| `login_sso`          | Login issues, SSO/SAML configuration      |
| `import_export_sync` | Data import/export, third-party sync      |
| `bug_report`         | Technical bugs, unexpected behavior       |
| `feature_request`    | Feature suggestions, capability inquiries |

## Project Structure

```
SaaS CS QA Studio/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Config, rubric, taxonomy
│   │   ├── db/            # Repository pattern
│   │   ├── models/        # Pydantic schemas, SQLAlchemy models
│   │   ├── providers/     # LLM provider interfaces
│   │   ├── rag/           # RAG indexer and retriever
│   │   ├── services/      # Pipeline, instrumentation, experiment
│   │   └── main.py        # FastAPI app
│   ├── tests/             # Pytest tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js pages
│   │   ├── components/    # React components
│   │   └── lib/           # API client
│   └── package.json
├── docs/
│   ├── policies/          # Internal policy documents
│   └── help_center/       # Help center article summaries
├── sample_data/           # Sample ticket JSONL files
├── scripts/               # Utility scripts
└── README.md
```

## Testing

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_mock_provider.py -v
```

## Development

### Adding New Document

1. Create markdown file in `docs/policies/` or `docs/help_center/`
2. Include frontmatter with `doc_id`, `title`, `version`, `tags`
3. Rebuild index: `curl -X POST http://localhost:8000/api/v1/documents/reindex`

### Adding New Taxonomy Label

1. Add to `backend/app/core/taxonomy.py`:
   - Add to `TaxonomyLabel` enum
   - Add required slots to `REQUIRED_SLOTS`
   - Add description to `LABEL_DESCRIPTIONS`
2. Update mock provider keyword matching in `providers/mock.py`
3. Update frontend filter options

### Adding New Failure Tag

1. Add to `FailureTag` enum in `taxonomy.py`
2. Update mock provider detection logic in `providers/mock.py`
3. (Optional) Update sampling rules if should trigger human queue

## License

MIT License - See LICENSE file

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
