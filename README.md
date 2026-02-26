# QA Studio

**Automated Quality Assessment for LLM-Generated Q/A Responses**

Evaluates LLM-generated answers through gate checks and multi-dimension scoring, with RAG-based evidence retrieval, human-in-the-loop review, A/B experimentation, and an automated self-improvement cycle.

## Features

- **Automated Evaluation**: Gate checks (factual_safety, hallucination) + 4-dimension scoring (instruction_following, reasoning_quality, completeness, clarity)
- **RAG-based Evidence**: FAISS vector index for evidence-based hallucination detection
- **Human Review Queue**: Smart sampling for edge cases (gate failures, low scores, novel tags)
- **A/B Experiments**: Compare prompt/model versions with statistical analysis
- **Self-Improvement Cycle**: Pattern analysis, prompt suggestions, proposals, automated A/B testing
- **Langfuse Observability**: Full tracing with graceful fallback when unavailable
- **n8n Integration**: Schedule-driven workflows for automated pipeline execution

## Quick Start

### Backend Setup

```bash
cd backend
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Access the UI at http://localhost:3001

### Data Preparation

```bash
cd backend
pip install datasets>=2.14.0

# Prepare UltraFeedback dataset (4 model responses per instruction, GPT-4 annotations)
python scripts/prepare_ultrafeedback.py --dev-size 2000 --test-size 500 --ab-size 300
```

### Ingest & Evaluate

```bash
# Ingest data
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./sample_data/ultrafeedback_dev.jsonl", "split": "dev"}'

# Run evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate/run" \
  -H "Content-Type: application/json" \
  -d '{"dataset_split": "dev", "prompt_version": "v1", "model_version": "mock", "docs_version": "v1"}'
```

## Environment Variables

Create `.env` in `backend/`:

```env
APP_NAME=QA Studio
DEBUG=true
DATABASE_URL=sqlite:///./qa_studio.db

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# LLM Provider (mock for testing, openai for production)
LLM_PROVIDER=mock
OPENAI_API_KEY=sk-...

# RAG
DOCS_PATH=./docs
VECTOR_STORE_PATH=./vector_store
EMBEDDING_MODEL=text-embedding-3-small
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ingest/batch` | Ingest JSONL from server path |
| POST | `/api/v1/ingest/upload` | Ingest uploaded JSONL/CSV file |
| POST | `/api/v1/evaluate/run` | Run evaluation on items |
| GET | `/api/v1/items` | List items (pagination, split/scenario/source filters) |
| GET | `/api/v1/items/{id}` | Get item with evaluations |
| GET | `/api/v1/items/scenario/{id}` | Get all candidates for a scenario |
| GET | `/api/v1/evaluations` | List evaluations (filter by item/version) |
| POST | `/api/v1/experiment/ab` | Run A/B experiment |
| POST | `/api/v1/experiment/multi` | Run N-way comparison |
| GET | `/api/v1/experiments` | List experiments |
| GET | `/api/v1/human/queue` | Get pending human review items |
| POST | `/api/v1/human/review` | Submit human review |
| GET | `/api/v1/reports/summary` | Get evaluation summary report |
| POST | `/api/v1/analysis/patterns` | Run failure pattern analysis |
| POST | `/api/v1/suggestions/generate` | Generate prompt improvement suggestions |
| GET | `/api/v1/proposals` | List prompt proposals |
| POST | `/api/v1/proposals` | Create prompt proposal |
| GET | `/api/v1/documents` | List RAG-indexed documents |
| POST | `/api/v1/documents/reindex` | Rebuild RAG index |
| GET | `/api/v1/health` | Health check |

## Data Format

### JSONL Ingest Format

```json
{
  "id": "uf_evol_instruct_42_alpaca-7b",
  "scenario_id": "uf_evol_instruct_42",
  "candidate_source": "alpaca-7b",
  "system_prompt": "You are a helpful assistant.",
  "question": "Explain the concept of recursion.",
  "response": "Recursion is when a function calls itself...",
  "metadata": {
    "dataset": "ultrafeedback",
    "overall_score": 4,
    "annotations": {"instruction_following": 5, "truthfulness": 3}
  }
}
```

### Evaluation Rubric

**Gates (Pass/Fail)**

| Gate | Description |
|------|-------------|
| `factual_safety` | No harmful instructions, PII leaks, or unsafe content |
| `hallucination` | No fabricated facts, citations, or unsubstantiated claims |

**Scores (1-5)**

| Score | Description |
|-------|-------------|
| `instruction_following` | Follows every aspect of the instruction |
| `reasoning_quality` | Sound reasoning with clear logical steps |
| `completeness` | Addresses every aspect of the question |
| `clarity` | Clear, well-organized, easy to follow |

### Taxonomy

8 Q/A task labels: `reasoning`, `math`, `classification`, `summarization`, `extraction`, `creative_writing`, `coding`, `open_qa`

12 failure tags: `instruction_miss`, `incomplete_answer`, `hallucination`, `logic_error`, `format_violation`, `over_verbose`, `under_verbose`, `wrong_language`, `unsafe_content`, `citation_missing`, `off_topic`, `partial_answer`

## Project Structure

```
QA Studio/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── chains/        # LCEL chain wrappers (optional)
│   │   ├── core/          # Config, rubric, taxonomy
│   │   ├── db/            # Repository pattern
│   │   ├── models/        # Pydantic schemas, SQLAlchemy models
│   │   ├── providers/     # LLM provider interfaces (mock, openai)
│   │   ├── rag/           # RAG indexer and retriever
│   │   ├── services/      # Pipeline, experiment, pattern analysis, suggestions
│   │   └── main.py        # FastAPI app entry point
│   ├── tests/             # Pytest tests
│   ├── scripts/           # Data preparation (prepare_ultrafeedback.py)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js pages (items, experiments, review)
│   │   ├── components/    # React components
│   │   └── lib/           # API client
│   └── package.json
├── docs/
│   ├── rubrics/           # Scoring rubric documents (primary RAG source)
│   ├── policies/          # Policy documents
│   └── help_center/       # Help center articles
├── sample_data/           # Generated JSONL splits (gitignored)
├── n8n_workflow.json      # Self-improvement cycle workflow
├── CLAUDE.md              # Claude Code project context
└── README.md
```

## Testing

```bash
cd backend
source .venv/bin/activate
PYTHONPATH=. pytest -v          # all tests
PYTHONPATH=. pytest --cov=app   # with coverage
```

## License

MIT License
