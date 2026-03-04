"""Ingest endpoints."""
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ...db.repository import EvalItemRepository
from ...models.schemas import DatasetSplit, EvalItemCreate, IngestRequest, IngestResponse
from ..deps import get_db

router = APIRouter()



def _resolve_ingest_path(raw_path: str) -> Path:
    """Resolve and validate ingest path against approved project data roots."""
    project_root = Path(__file__).resolve().parents[4]
    allowed_roots = [
        (project_root / "backend" / "data").resolve(),
        (project_root / "sample_data").resolve(),
    ]

    candidate = Path(raw_path).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail=(
                "File path is not allowed. Use files under "
                "'backend/data' or 'sample_data'."
            ),
        )

    return candidate


async def _parse_and_ingest(
    data_lines: List[str],
    split: DatasetSplit,
    db: Session,
) -> IngestResponse:
    """Parse JSONL lines and ingest eval items."""
    item_repo = EvalItemRepository(db)
    errors = []
    items_to_create = []

    for i, line in enumerate(data_lines):
        try:
            data = json.loads(line)

            # Supports standard Q/A, UltraFeedback, and legacy ticket formats
            system_prompt = data.get("system_prompt", data.get("system", ""))
            question = data.get("question", data.get("instruction", data.get("query", "")))
            response = data.get("response", data.get("output", data.get("answer", "")))

            # Legacy ticket format: conversation[0].content → question,
            # candidate_response → response
            if not question and "conversation" in data:
                conv = data["conversation"]
                if isinstance(conv, list) and conv:
                    question = conv[0].get("content", "") if isinstance(conv[0], dict) else str(conv[0])
            if not response and "candidate_response" in data:
                response = data["candidate_response"]

            if not question or not response:
                errors.append(f"Line {i + 1}: Missing question or response field")
                continue

            item = EvalItemCreate(
                external_id=data.get("id", data.get("external_id")),
                system_prompt=system_prompt or None,
                question=question,
                response=response,
                metadata=data.get("metadata"),
                scenario_id=data.get("scenario_id"),
                candidate_source=data.get("candidate_source"),
                split=split,
            )
            items_to_create.append(item)

        except json.JSONDecodeError as e:
            errors.append(f"Line {i + 1}: JSON parse error - {e}")
        except Exception as e:
            errors.append(f"Line {i + 1}: {str(e)}")

    ingested = 0
    if items_to_create:
        ingested = item_repo.create_batch(items_to_create)

    return IngestResponse(
        ingested_count=ingested,
        split=split,
        errors=errors[:10],
    )


@router.post("/ingest/batch", response_model=IngestResponse)
async def ingest_batch(
    request: IngestRequest,
    db: Session = Depends(get_db),
):
    """Ingest batch of eval items from server file path."""
    file_path = _resolve_ingest_path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    with open(file_path) as f:
        if str(file_path).endswith(".jsonl"):
            data_lines = f.read().strip().split("\n")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .jsonl")

    return await _parse_and_ingest(data_lines, request.split, db)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    split: DatasetSplit = Form(DatasetSplit.DEV),
    db: Session = Depends(get_db),
):
    """Ingest batch of eval items from file upload."""
    content = await file.read()
    content_str = content.decode("utf-8")

    if file.filename.endswith(".jsonl"):
        data_lines = content_str.strip().split("\n")
    elif file.filename.endswith(".csv"):
        import csv
        import io

        data_lines = []
        reader = csv.DictReader(io.StringIO(content_str))
        for row in reader:
            data_lines.append(json.dumps(row))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use .jsonl or .csv")

    return await _parse_and_ingest(data_lines, split, db)
