"""Repositories for human queue and human reviews."""
from typing import List, Optional

from sqlalchemy.orm import Session

from ...models.database import (
    EvalItemModel,
    HumanQueueModel,
    HumanReviewModel,
    json_deserializer,
    json_serializer,
)
from ...models.schemas import (
    DatasetSplit,
    HumanQueueItem,
    HumanQueueReason,
    HumanReviewCreate,
    HumanReviewInDB,
)
from .common import generate_id


class HumanQueueRepository:
    """Repository for human queue operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, queue_id: str) -> Optional[HumanQueueItem]:
        model = self.db.query(HumanQueueModel).filter(HumanQueueModel.id == queue_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_pending(self, limit: int = 50, offset: int = 0) -> List[HumanQueueItem]:
        models = (
            self.db.query(HumanQueueModel)
            .filter(HumanQueueModel.reviewed == False)
            .order_by(HumanQueueModel.priority.desc(), HumanQueueModel.created_at)
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [self._to_schema(m) for m in models]

    def get_by_item(self, item_id: str) -> List[HumanQueueItem]:
        models = self.db.query(HumanQueueModel).filter(HumanQueueModel.item_id == item_id).all()
        return [self._to_schema(m) for m in models]

    def create(
        self,
        item_id: str,
        evaluation_id: str,
        reason: HumanQueueReason,
        priority: int = 0,
        commit: bool = True,
    ) -> HumanQueueItem:
        model = HumanQueueModel(
            id=generate_id(),
            item_id=item_id,
            evaluation_id=evaluation_id,
            reason=reason,
            priority=priority,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        else:
            self.db.flush()
        return self._to_schema(model)

    def mark_reviewed(self, queue_id: str, commit: bool = True) -> bool:
        model = self.db.query(HumanQueueModel).filter(HumanQueueModel.id == queue_id).first()
        if not model:
            return False
        model.reviewed = True
        if commit:
            self.db.commit()
        else:
            self.db.flush()
        return True

    def count_pending(self, split: Optional[DatasetSplit] = None) -> int:
        query = self.db.query(HumanQueueModel).filter(HumanQueueModel.reviewed == False)
        if split:
            query = query.join(
                EvalItemModel, HumanQueueModel.item_id == EvalItemModel.id
            ).filter(EvalItemModel.split == split)
        return query.count()

    def _to_schema(self, model: HumanQueueModel) -> HumanQueueItem:
        return HumanQueueItem(
            id=model.id,
            item_id=model.item_id,
            evaluation_id=model.evaluation_id,
            reason=model.reason,
            priority=model.priority,
            created_at=model.created_at,
            reviewed=model.reviewed,
        )


class HumanReviewRepository:
    """Repository for human review operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, review_id: str) -> Optional[HumanReviewInDB]:
        model = self.db.query(HumanReviewModel).filter(HumanReviewModel.id == review_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_evaluation(self, eval_id: str) -> Optional[HumanReviewInDB]:
        model = self.db.query(HumanReviewModel).filter(HumanReviewModel.evaluation_id == eval_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def create(self, review: HumanReviewCreate, commit: bool = True) -> HumanReviewInDB:
        model = HumanReviewModel(
            id=generate_id(),
            queue_item_id=review.queue_item_id,
            evaluation_id=review.evaluation_id,
            reviewer_id=review.reviewer_id,
            gold_label=review.gold_label,
            gold_gates_json=json_serializer(review.gold_gates) if review.gold_gates else None,
            gold_scores_json=json_serializer(review.gold_scores) if review.gold_scores else None,
            gold_tags_json=json_serializer(review.gold_tags) if review.gold_tags else None,
            notes=review.notes,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        else:
            self.db.flush()
        return self._to_schema(model)

    def _to_schema(self, model: HumanReviewModel) -> HumanReviewInDB:
        return HumanReviewInDB(
            id=model.id,
            queue_item_id=model.queue_item_id,
            evaluation_id=model.evaluation_id,
            reviewer_id=model.reviewer_id,
            gold_label=model.gold_label,
            gold_gates=json_deserializer(model.gold_gates_json) if model.gold_gates_json else None,
            gold_scores=json_deserializer(model.gold_scores_json) if model.gold_scores_json else None,
            gold_tags=json_deserializer(model.gold_tags_json) if model.gold_tags_json else None,
            notes=model.notes,
            created_at=model.created_at,
        )
