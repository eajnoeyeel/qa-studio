"""Item and evaluation listing endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...db.repository import EvalItemRepository, EvaluationRepository, JudgeOutputRepository
from ...models.schemas import DatasetSplit, EvalItemListResponse, ScenarioItemsResponse
from ..deps import get_db

router = APIRouter()


@router.get("/items", response_model=EvalItemListResponse)
async def list_items(
    split: Optional[DatasetSplit] = None,
    scenario_id: Optional[str] = None,
    candidate_source: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List eval items with pagination and optional filters."""
    repo = EvalItemRepository(db)
    items, total = repo.get_all(
        split=split,
        scenario_id=scenario_id,
        candidate_source=candidate_source,
        page=page,
        page_size=page_size,
    )

    return EvalItemListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/items/scenario/{scenario_id}", response_model=ScenarioItemsResponse)
async def get_items_by_scenario(
    scenario_id: str,
    db: Session = Depends(get_db),
):
    """Get all candidate items for a scenario."""
    repo = EvalItemRepository(db)
    items = repo.get_by_scenario(scenario_id)
    return ScenarioItemsResponse(scenario_id=scenario_id, items=items, count=len(items))


@router.get("/items/{item_id}")
async def get_item(
    item_id: str,
    db: Session = Depends(get_db),
):
    """Get eval item by ID with evaluations."""
    item_repo = EvalItemRepository(db)
    eval_repo = EvaluationRepository(db)
    judge_repo = JudgeOutputRepository(db)

    item = item_repo.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    evaluations = eval_repo.get_by_item(item_id)

    enriched_evals = []
    for eval_item in evaluations:
        judge = judge_repo.get_by_evaluation(eval_item.id)
        enriched_evals.append(
            {
                **eval_item.model_dump(),
                "judge_output": judge.model_dump() if judge else None,
            }
        )

    return {
        "item": item.model_dump(),
        "evaluations": enriched_evals,
    }


@router.get("/evaluations")
async def list_evaluations(
    item_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    model_version: Optional[str] = None,
    docs_version: str = Query("v1", description="Docs version (defaults to 'v1')"),
    db: Session = Depends(get_db),
):
    """List evaluations with optional filters."""
    repo = EvaluationRepository(db)

    if item_id:
        evaluations = repo.get_by_item(item_id)
    elif prompt_version and model_version:
        evaluations = repo.get_by_version(prompt_version, model_version, docs_version)
    else:
        raise HTTPException(status_code=400, detail="Provide item_id or version filters")

    return {"evaluations": [e.model_dump() for e in evaluations]}
