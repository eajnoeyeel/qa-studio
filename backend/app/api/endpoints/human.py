"""Human review and report endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...db.repository import EvalItemRepository, HumanQueueRepository, HumanReviewRepository
from ...models.schemas import DatasetSplit, HumanQueueItem, HumanReviewCreate, HumanReviewInDB, ReportSummaryResponse
from ..deps import get_db

router = APIRouter()


@router.get("/human/queue", response_model=list[HumanQueueItem])
async def get_human_queue(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get pending items in human review queue."""
    repo = HumanQueueRepository(db)
    return repo.get_pending(limit=limit, offset=offset)


@router.post("/human/review", response_model=HumanReviewInDB)
async def submit_human_review(
    review: HumanReviewCreate,
    db: Session = Depends(get_db),
):
    """Submit human review (gold label)."""
    review_repo = HumanReviewRepository(db)
    queue_repo = HumanQueueRepository(db)

    queue_item = queue_repo.get(review.queue_item_id)
    if not queue_item:
        raise HTTPException(status_code=404, detail="Queue item not found")

    if queue_item.evaluation_id != review.evaluation_id:
        raise HTTPException(
            status_code=400,
            detail="evaluation_id does not match the queue item",
        )

    try:
        queue_repo.mark_reviewed(review.queue_item_id, commit=False)
        created_review = review_repo.create(
            HumanReviewCreate(
                queue_item_id=review.queue_item_id,
                evaluation_id=queue_item.evaluation_id,
                reviewer_id=review.reviewer_id,
                gold_label=review.gold_label,
                gold_gates=review.gold_gates,
                gold_scores=review.gold_scores,
                gold_tags=review.gold_tags,
                notes=review.notes,
            ),
            commit=False,
        )
        db.commit()
        return created_review
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit human review: {e}")


@router.get("/reports/summary", response_model=ReportSummaryResponse)
async def get_report_summary(
    dataset_split: DatasetSplit,
    db: Session = Depends(get_db),
):
    """Get summary report for evaluations."""
    item_repo = EvalItemRepository(db)
    queue_repo = HumanQueueRepository(db)

    stats = item_repo.get_summary_stats(dataset_split)
    total_evals = stats["total_evaluations"]
    pending_count = queue_repo.count_pending(split=dataset_split)

    return ReportSummaryResponse(
        dataset_split=dataset_split,
        total_evaluations=total_evals,
        gate_fail_rate=stats["gate_fail_count"] / total_evals if total_evals > 0 else 0,
        avg_scores=stats.get("avg_scores", {}),
        tag_distribution=dict(sorted(stats.get("tag_counts", {}).items(), key=lambda x: -x[1])),
        human_queue_stats={"pending": pending_count},
    )
