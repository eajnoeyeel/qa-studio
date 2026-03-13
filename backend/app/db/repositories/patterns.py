"""Repositories for failure pattern persistence."""
from typing import List, Optional

from sqlalchemy.orm import Session

from ...models.database import FailurePatternModel, json_deserializer, json_serializer
from ...models.schemas import DatasetSplit, FailurePattern
from .common import generate_id


class FailurePatternRepository:
    """Repository for failure pattern operations."""

    def __init__(self, db: Session):
        self.db = db

    def create_batch(self, patterns: List[dict], analysis_run_id: str) -> int:
        """Batch-create failure patterns from analysis results."""
        models = []
        for p in patterns:
            model = FailurePatternModel(
                id=generate_id(),
                analysis_run_id=analysis_run_id,
                tags_json=json_serializer(p["tags"]),
                frequency=p["frequency"],
                avg_scores_json=json_serializer(p.get("avg_scores", {})),
                taxonomy_labels_json=json_serializer(p.get("taxonomy_labels", {})),
                dataset_split=p.get("dataset_split"),
                prompt_version=p.get("prompt_version"),
                model_version=p.get("model_version"),
            )
            models.append(model)
        self.db.add_all(models)
        self.db.commit()
        return len(models)

    def get_latest(
        self,
        top_k: int = 10,
        dataset_split: Optional[DatasetSplit] = None,
    ) -> List[FailurePattern]:
        """Get patterns from the most recent analysis run."""
        latest_query = self.db.query(FailurePatternModel)
        if dataset_split is not None:
            latest_query = latest_query.filter(FailurePatternModel.dataset_split == dataset_split)
        latest = latest_query.order_by(
            FailurePatternModel.created_at.desc()
        ).first()
        if not latest:
            return []
        run_id = latest.analysis_run_id
        query = self.db.query(FailurePatternModel).filter(
            FailurePatternModel.analysis_run_id == run_id
        )
        if dataset_split is not None:
            query = query.filter(FailurePatternModel.dataset_split == dataset_split)
        models = query.order_by(FailurePatternModel.frequency.desc()).limit(top_k).all()
        return [self._to_schema(m) for m in models]

    def get_latest_run_id(self, dataset_split: Optional[DatasetSplit] = None) -> Optional[str]:
        """Get the most recent analysis run ID."""
        query = self.db.query(FailurePatternModel)
        if dataset_split is not None:
            query = query.filter(FailurePatternModel.dataset_split == dataset_split)
        latest = query.order_by(
            FailurePatternModel.created_at.desc()
        ).first()
        return latest.analysis_run_id if latest else None

    def _to_schema(self, model: FailurePatternModel) -> FailurePattern:
        return FailurePattern(
            id=model.id,
            analysis_run_id=model.analysis_run_id,
            tags=json_deserializer(model.tags_json) or [],
            frequency=model.frequency,
            avg_scores=json_deserializer(model.avg_scores_json) or {},
            taxonomy_labels=json_deserializer(model.taxonomy_labels_json) or {},
            dataset_split=model.dataset_split,
            prompt_version=model.prompt_version,
            model_version=model.model_version,
            created_at=model.created_at,
        )
