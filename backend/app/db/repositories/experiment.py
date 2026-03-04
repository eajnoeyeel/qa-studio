"""Repositories for experiments and multi-comparison results."""
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ...models.database import (
    ExperimentModel,
    ExperimentResultModel,
    MultiComparisonResultModel,
    json_deserializer,
    json_serializer,
)
from ...models.schemas import (
    ExperimentConfig,
    ExperimentCreate,
    ExperimentInDB,
    ExperimentResult,
    ExperimentSummary,
)
from .common import generate_id


class ExperimentRepository:
    """Repository for experiment operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, exp_id: str) -> Optional[ExperimentInDB]:
        model = self.db.query(ExperimentModel).filter(ExperimentModel.id == exp_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_all(self) -> List[ExperimentInDB]:
        models = self.db.query(ExperimentModel).order_by(ExperimentModel.created_at.desc()).all()
        return [self._to_schema(m) for m in models]

    def create(self, experiment: ExperimentCreate) -> ExperimentInDB:
        model = ExperimentModel(
            id=generate_id(),
            name=experiment.name,
            dataset_split=experiment.dataset_split,
            docs_version=experiment.docs_version,
            config_a_json=json_serializer(experiment.config_a.model_dump()),
            config_b_json=json_serializer(experiment.config_b.model_dump()),
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def update_summary(self, exp_id: str, summary: ExperimentSummary) -> Optional[ExperimentInDB]:
        model = self.db.query(ExperimentModel).filter(ExperimentModel.id == exp_id).first()
        if not model:
            return None
        model.summary_json = json_serializer(summary.model_dump())
        model.completed_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: ExperimentModel) -> ExperimentInDB:
        config_a = ExperimentConfig(**json_deserializer(model.config_a_json))
        config_b = ExperimentConfig(**json_deserializer(model.config_b_json))
        summary = None
        if model.summary_json:
            summary = ExperimentSummary(**json_deserializer(model.summary_json))

        return ExperimentInDB(
            id=model.id,
            name=model.name,
            dataset_split=model.dataset_split,
            docs_version=model.docs_version,
            config_a=config_a,
            config_b=config_b,
            summary=summary,
            created_at=model.created_at,
            completed_at=model.completed_at,
        )


class ExperimentResultRepository:
    """Repository for experiment result operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_experiment(self, exp_id: str) -> List[ExperimentResult]:
        models = self.db.query(ExperimentResultModel).filter(
            ExperimentResultModel.experiment_id == exp_id
        ).all()
        return [self._to_schema(m) for m in models]

    def create(self, exp_id: str, result: ExperimentResult) -> ExperimentResult:
        model = ExperimentResultModel(
            id=generate_id(),
            experiment_id=exp_id,
            item_id=result.item_id,
            eval_a_id=result.eval_a_id,
            eval_b_id=result.eval_b_id,
            score_diff_json=json_serializer(result.score_diff),
            gate_diff_json=json_serializer(result.gate_diff),
            is_ambiguous=result.is_ambiguous,
            winner=result.winner,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: ExperimentResultModel) -> ExperimentResult:
        return ExperimentResult(
            item_id=model.item_id,
            eval_a_id=model.eval_a_id,
            eval_b_id=model.eval_b_id,
            score_diff=json_deserializer(model.score_diff_json),
            gate_diff=json_deserializer(model.gate_diff_json),
            is_ambiguous=model.is_ambiguous,
            winner=model.winner,
        )


class MultiComparisonRepository:
    """Repository for multi-comparison results."""

    def __init__(self, db: Session):
        self.db = db

    def create_item_result(
        self,
        experiment_id: str,
        item_id: str,
        config_results: Dict,
        rankings: List,
        winner_config_id: Optional[str],
    ) -> str:
        model = MultiComparisonResultModel(
            id=generate_id(),
            experiment_id=experiment_id,
            item_id=item_id,
            config_results_json=json_serializer(config_results),
            rankings_json=json_serializer(rankings),
            winner_config_id=winner_config_id,
        )
        self.db.add(model)
        self.db.commit()
        return model.id

    def get_by_experiment(self, experiment_id: str) -> List[Dict]:
        models = self.db.query(MultiComparisonResultModel).filter(
            MultiComparisonResultModel.experiment_id == experiment_id
        ).all()
        return [
            {
                "id": m.id,
                "item_id": m.item_id,
                "config_results": json_deserializer(m.config_results_json),
                "rankings": json_deserializer(m.rankings_json),
                "winner_config_id": m.winner_config_id,
            }
            for m in models
        ]
