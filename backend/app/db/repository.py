"""Repository pattern for database operations - allows SQLite/Postgres swapping."""
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Generic, List, Optional, Type, TypeVar
from sqlalchemy.orm import Session

from ..models.database import (
    Base, EvalItemModel, EvaluationModel, JudgeOutputModel,
    HumanQueueModel, HumanReviewModel, ExperimentModel,
    ExperimentResultModel, DocumentModel, TraceLogModel,
    FailurePatternModel, MultiComparisonResultModel, PromptProposalModel,
    json_serializer
)
from ..models.schemas import (
    EvalItemCreate, EvalItemInDB, EvaluationCreate, EvaluationInDB,
    JudgeOutput, JudgeOutputInDB, HumanQueueItem, HumanQueueReason,
    HumanReviewCreate, HumanReviewInDB, ExperimentCreate, ExperimentInDB,
    ExperimentResult, ExperimentSummary, DocumentMeta, DocumentInDB,
    DatasetSplit, ClassificationResult,
    FailurePattern, MultiComparisonSummary, ConfigRanking,
    PromptProposalCreate, PromptProposalInDB, ProposalStatus
)


T = TypeVar("T", bound=Base)


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository."""

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        pass

    @abstractmethod
    def get_all(self, **filters) -> List[T]:
        pass

    @abstractmethod
    def create(self, entity: T) -> T:
        pass

    @abstractmethod
    def update(self, id: str, **kwargs) -> Optional[T]:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass


class EvalItemRepository:
    """Repository for eval item operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, item_id: str) -> Optional[EvalItemInDB]:
        model = self.db.query(EvalItemModel).filter(EvalItemModel.id == item_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_external_id(self, external_id: str) -> Optional[EvalItemInDB]:
        model = self.db.query(EvalItemModel).filter(EvalItemModel.external_id == external_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_all(
        self,
        split: Optional[DatasetSplit] = None,
        scenario_id: Optional[str] = None,
        candidate_source: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> tuple[List[EvalItemInDB], int]:
        query = self.db.query(EvalItemModel)
        if split:
            query = query.filter(EvalItemModel.split == split)
        if scenario_id:
            query = query.filter(EvalItemModel.scenario_id == scenario_id)
        if candidate_source:
            query = query.filter(EvalItemModel.candidate_source == candidate_source)

        total = query.count()
        models = query.order_by(EvalItemModel.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
        return [self._to_schema(m) for m in models], total

    def get_by_split(self, split: DatasetSplit) -> List[EvalItemInDB]:
        models = self.db.query(EvalItemModel).filter(
            EvalItemModel.split == split
        ).order_by(EvalItemModel.created_at).all()
        return [self._to_schema(m) for m in models]

    def create(self, item: EvalItemCreate, commit: bool = True) -> EvalItemInDB:
        model = EvalItemModel(
            id=generate_id(),
            external_id=item.external_id,
            split=item.split,
            system_prompt=item.system_prompt,
            question=item.question,
            response=item.response,
            metadata_json=json_serializer(item.metadata) if item.metadata else None,
            scenario_id=item.scenario_id,
            candidate_source=item.candidate_source,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def create_batch(self, items: List[EvalItemCreate], skip_duplicates: bool = True) -> int:
        """Batch create eval items in a single transaction.

        When skip_duplicates=True, items with an external_id that already exists
        in the database are silently skipped.
        """
        existing_ids: set = set()
        if skip_duplicates:
            ext_ids = [i.external_id for i in items if i.external_id]
            if ext_ids:
                rows = self.db.query(EvalItemModel.external_id).filter(
                    EvalItemModel.external_id.in_(ext_ids)
                ).all()
                existing_ids = {r[0] for r in rows}

        models = []
        for item in items:
            if skip_duplicates and item.external_id and item.external_id in existing_ids:
                continue
            model = EvalItemModel(
                id=generate_id(),
                external_id=item.external_id,
                split=item.split,
                system_prompt=item.system_prompt,
                question=item.question,
                response=item.response,
                metadata_json=json_serializer(item.metadata) if item.metadata else None,
                scenario_id=item.scenario_id,
                candidate_source=item.candidate_source,
            )
            models.append(model)
        if models:
            self.db.add_all(models)
            self.db.commit()
        return len(models)

    def flush(self):
        """Commit pending changes."""
        self.db.commit()

    def update_masked(self, item_id: str, masked_text: str, commit: bool = True) -> Optional[EvalItemInDB]:
        model = self.db.query(EvalItemModel).filter(EvalItemModel.id == item_id).first()
        if not model:
            return None
        model.masked_text = masked_text
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def get_by_scenario(self, scenario_id: str) -> List[EvalItemInDB]:
        """Get all candidate items for a given scenario."""
        models = self.db.query(EvalItemModel).filter(
            EvalItemModel.scenario_id == scenario_id
        ).all()
        return [self._to_schema(m) for m in models]

    def _to_schema(self, model: EvalItemModel) -> EvalItemInDB:
        return EvalItemInDB(
            id=model.id,
            external_id=model.external_id,
            split=model.split,
            system_prompt=model.system_prompt,
            question=model.question,
            response=model.response,
            metadata=model.item_metadata,
            scenario_id=model.scenario_id,
            candidate_source=model.candidate_source,
            masked_text=model.masked_text,
            created_at=model.created_at,
        )


    def get_summary_stats(self, split: DatasetSplit) -> Dict:
        """Get aggregated evaluation stats for a split using JOINed queries."""
        from sqlalchemy import func
        from ..models.database import EvaluationModel, JudgeOutputModel

        # Total evaluations for this split
        total_evals = self.db.query(func.count(EvaluationModel.id)).join(
            EvalItemModel, EvalItemModel.id == EvaluationModel.item_id
        ).filter(EvalItemModel.split == split).scalar() or 0

        if total_evals == 0:
            return {
                "total_evaluations": 0,
                "gate_fail_count": 0,
                "scores_raw": [],
                "tags_raw": [],
            }

        # Get all judge outputs for this split in one query
        judge_rows = self.db.query(
            JudgeOutputModel.gates_json,
            JudgeOutputModel.scores_json,
            JudgeOutputModel.failure_tags_json,
        ).join(
            EvaluationModel, EvaluationModel.id == JudgeOutputModel.evaluation_id
        ).join(
            EvalItemModel, EvalItemModel.id == EvaluationModel.item_id
        ).filter(EvalItemModel.split == split).all()

        from ..models.database import json_deserializer
        gate_fails = 0
        score_sums: Dict[str, float] = {}
        score_counts: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}

        for gates_json, scores_json, tags_json in judge_rows:
            gates = json_deserializer(gates_json) or []
            scores = json_deserializer(scores_json) or []
            tags = json_deserializer(tags_json) or []

            if not all(g.get("passed", True) for g in gates):
                gate_fails += 1
            for s in scores:
                st = s.get("score_type", "")
                sv = s.get("score", 0)
                score_sums[st] = score_sums.get(st, 0) + sv
                score_counts[st] = score_counts.get(st, 0) + 1
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        avg_scores = {k: score_sums[k] / score_counts[k] for k in score_sums if score_counts[k] > 0}

        return {
            "total_evaluations": total_evals,
            "gate_fail_count": gate_fails,
            "avg_scores": avg_scores,
            "tag_counts": tag_counts,
        }


class EvaluationRepository:
    """Repository for evaluation operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, eval_id: str) -> Optional[EvaluationInDB]:
        model = self.db.query(EvaluationModel).filter(EvaluationModel.id == eval_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_item(self, item_id: str) -> List[EvaluationInDB]:
        models = self.db.query(EvaluationModel).filter(
            EvaluationModel.item_id == item_id
        ).order_by(EvaluationModel.created_at.desc()).all()
        return [self._to_schema(m) for m in models]

    def get_by_version(
        self,
        prompt_version: str,
        model_version: str,
        docs_version: str
    ) -> List[EvaluationInDB]:
        models = self.db.query(EvaluationModel).filter(
            EvaluationModel.prompt_version == prompt_version,
            EvaluationModel.model_version == model_version,
            EvaluationModel.docs_version == docs_version
        ).all()
        return [self._to_schema(m) for m in models]

    def get_evaluated_item_ids(
        self,
        prompt_version: str,
        model_version: str,
        docs_version: str,
    ) -> set:
        """Return set of item_ids that already have an evaluation for this version triple."""
        rows = self.db.query(EvaluationModel.item_id).filter(
            EvaluationModel.prompt_version == prompt_version,
            EvaluationModel.model_version == model_version,
            EvaluationModel.docs_version == docs_version,
        ).all()
        return {r[0] for r in rows}

    def create(self, evaluation: EvaluationCreate, trace_id: Optional[str] = None, commit: bool = True) -> EvaluationInDB:
        model = EvaluationModel(
            id=generate_id(),
            item_id=evaluation.item_id,
            prompt_version=evaluation.prompt_version,
            model_version=evaluation.model_version,
            docs_version=evaluation.docs_version,
            trace_id=trace_id,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        else:
            self.db.flush()
        return self._to_schema(model)

    def update_classification(self, eval_id: str, classification: ClassificationResult, commit: bool = True) -> Optional[EvaluationInDB]:
        model = self.db.query(EvaluationModel).filter(EvaluationModel.id == eval_id).first()
        if not model:
            return None
        model.classification_json = json_serializer(classification.model_dump())
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: EvaluationModel) -> EvaluationInDB:
        classification = None
        if model.classification:
            classification = ClassificationResult(**model.classification)

        judge = None
        if model.judge_output:
            judge = JudgeOutput(
                gates=model.judge_output.gates,
                scores=model.judge_output.scores,
                failure_tags=model.judge_output.failure_tags,
                summary_of_issue=model.judge_output.summary_of_issue,
                what_to_fix=model.judge_output.what_to_fix,
                rag_citations=model.judge_output.rag_citations,
            )

        return EvaluationInDB(
            id=model.id,
            item_id=model.item_id,
            prompt_version=model.prompt_version,
            model_version=model.model_version,
            docs_version=model.docs_version,
            classification=classification,
            judge_output=judge,
            trace_id=model.trace_id,
            created_at=model.created_at,
        )


class JudgeOutputRepository:
    """Repository for judge output operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, output_id: str) -> Optional[JudgeOutputInDB]:
        model = self.db.query(JudgeOutputModel).filter(JudgeOutputModel.id == output_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_evaluation(self, eval_id: str) -> Optional[JudgeOutputInDB]:
        model = self.db.query(JudgeOutputModel).filter(JudgeOutputModel.evaluation_id == eval_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def create(self, eval_id: str, output: JudgeOutput, commit: bool = True) -> JudgeOutputInDB:
        model = JudgeOutputModel(
            id=generate_id(),
            evaluation_id=eval_id,
            gates_json=json_serializer([g.model_dump() for g in output.gates]),
            scores_json=json_serializer([s.model_dump() for s in output.scores]),
            failure_tags_json=json_serializer(output.failure_tags),
            summary_of_issue=output.summary_of_issue,
            what_to_fix=output.what_to_fix,
            rag_citations_json=json_serializer(output.rag_citations),
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        else:
            self.db.flush()
        return self._to_schema(model)

    def _to_schema(self, model: JudgeOutputModel) -> JudgeOutputInDB:
        return JudgeOutputInDB(
            id=model.id,
            evaluation_id=model.evaluation_id,
            gates=[g for g in model.gates],
            scores=[s for s in model.scores],
            failure_tags=model.failure_tags,
            summary_of_issue=model.summary_of_issue,
            what_to_fix=model.what_to_fix,
            rag_citations=model.rag_citations,
            created_at=model.created_at,
        )


class HumanQueueRepository:
    """Repository for human queue operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_pending(self, limit: int = 50) -> List[HumanQueueItem]:
        models = self.db.query(HumanQueueModel).filter(
            HumanQueueModel.reviewed == False
        ).order_by(HumanQueueModel.priority.desc(), HumanQueueModel.created_at).limit(limit).all()
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
        from ..models.database import json_deserializer
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
        from ..models.database import json_serializer
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
        from ..models.database import json_deserializer
        from ..models.schemas import ExperimentConfig

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
        from ..models.database import json_deserializer
        return ExperimentResult(
            item_id=model.item_id,
            eval_a_id=model.eval_a_id,
            eval_b_id=model.eval_b_id,
            score_diff=json_deserializer(model.score_diff_json),
            gate_diff=json_deserializer(model.gate_diff_json),
            is_ambiguous=model.is_ambiguous,
            winner=model.winner,
        )


class DocumentRepository:
    """Repository for document operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, doc_id: str) -> Optional[DocumentInDB]:
        model = self.db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_version(self, version: str) -> List[DocumentInDB]:
        models = self.db.query(DocumentModel).filter(DocumentModel.version == version).all()
        return [self._to_schema(m) for m in models]

    def get_all(self) -> List[DocumentInDB]:
        models = self.db.query(DocumentModel).all()
        return [self._to_schema(m) for m in models]

    def create(self, doc: DocumentMeta, content: str) -> DocumentInDB:
        model = DocumentModel(
            doc_id=doc.doc_id,
            title=doc.title,
            content=content,
            source_url=doc.source_url,
            version=doc.version,
            tags_json=json_serializer(doc.tags),
            category=doc.category,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: DocumentModel) -> DocumentInDB:
        return DocumentInDB(
            doc_id=model.doc_id,
            title=model.title,
            content=model.content,
            source_url=model.source_url,
            version=model.version,
            tags=model.tags,
            category=model.category,
            created_at=model.created_at,
        )


class TraceLogRepository:
    """Repository for trace log operations (Langfuse fallback)."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        trace_id: str,
        span_name: str,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        commit: bool = True,
    ):
        model = TraceLogModel(
            id=generate_id(),
            trace_id=trace_id,
            span_name=span_name,
            input_json=json_serializer(input_data) if input_data else None,
            output_json=json_serializer(output_data) if output_data else None,
            latency_ms=latency_ms,
            error=error,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
        else:
            self.db.flush()
        return model

    def get_by_trace(self, trace_id: str):
        return self.db.query(TraceLogModel).filter(TraceLogModel.trace_id == trace_id).all()


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
                prompt_version=p.get("prompt_version"),
                model_version=p.get("model_version"),
            )
            models.append(model)
        self.db.add_all(models)
        self.db.commit()
        return len(models)

    def get_latest(self, top_k: int = 10) -> List[FailurePattern]:
        """Get patterns from the most recent analysis run."""
        latest = self.db.query(FailurePatternModel).order_by(
            FailurePatternModel.created_at.desc()
        ).first()
        if not latest:
            return []
        run_id = latest.analysis_run_id
        models = self.db.query(FailurePatternModel).filter(
            FailurePatternModel.analysis_run_id == run_id
        ).order_by(FailurePatternModel.frequency.desc()).limit(top_k).all()
        return [self._to_schema(m) for m in models]

    def get_latest_run_id(self) -> Optional[str]:
        """Get the most recent analysis run ID."""
        latest = self.db.query(FailurePatternModel).order_by(
            FailurePatternModel.created_at.desc()
        ).first()
        return latest.analysis_run_id if latest else None

    def _to_schema(self, model: FailurePatternModel) -> FailurePattern:
        from ..models.database import json_deserializer
        return FailurePattern(
            id=model.id,
            analysis_run_id=model.analysis_run_id,
            tags=json_deserializer(model.tags_json) or [],
            frequency=model.frequency,
            avg_scores=json_deserializer(model.avg_scores_json) or {},
            taxonomy_labels=json_deserializer(model.taxonomy_labels_json) or {},
            prompt_version=model.prompt_version,
            model_version=model.model_version,
            created_at=model.created_at,
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
        from ..models.database import json_deserializer
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


class ProposalRepository:
    """Repository for prompt proposal operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, proposal: PromptProposalCreate) -> PromptProposalInDB:
        from datetime import datetime as dt
        model = PromptProposalModel(
            id=generate_id(),
            prompt_name=proposal.prompt_name,
            current_version=proposal.current_version,
            proposed_prompt=proposal.proposed_prompt,
            status="pending",
            created_by=proposal.created_by,
            created_at=dt.utcnow(),
            updated_at=dt.utcnow(),
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def get(self, proposal_id: str) -> Optional[PromptProposalInDB]:
        model = self.db.query(PromptProposalModel).filter(
            PromptProposalModel.id == proposal_id
        ).first()
        return self._to_schema(model) if model else None

    def get_all(self, status=None, limit: int = 50) -> List[PromptProposalInDB]:
        """List proposals, optionally filtered by status (str or ProposalStatus enum)."""
        query = self.db.query(PromptProposalModel)
        if status is not None:
            # Accept both string and enum
            status_val = status.value if hasattr(status, "value") else status
            query = query.filter(PromptProposalModel.status == status_val)
        models = query.order_by(PromptProposalModel.created_at.desc()).limit(limit).all()
        return [self._to_schema(m) for m in models]

    def update_status(
        self,
        proposal_id: str,
        status,  # str or ProposalStatus enum
        extra: Optional[Dict] = None,
    ) -> Optional[PromptProposalInDB]:
        """Update proposal status and apply any extra field updates."""
        from datetime import datetime as dt
        model = self.db.query(PromptProposalModel).filter(
            PromptProposalModel.id == proposal_id
        ).first()
        if not model:
            return None
        # Handle both string and enum
        model.status = status.value if hasattr(status, "value") else status
        model.updated_at = dt.utcnow()
        # Apply extra fields
        for key, value in (extra or {}).items():
            if key == "test_experiment_id":
                model.test_experiment_id = value
            elif key == "proposed_langfuse_version":
                model.proposed_langfuse_version = value
            elif key == "improvement_metrics":
                model.improvement_metrics_json = json_serializer(value) if value is not None else None
            elif key == "deployed_at":
                model.deployed_at = value
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: PromptProposalModel) -> PromptProposalInDB:
        from ..models.database import json_deserializer
        return PromptProposalInDB(
            id=model.id,
            prompt_name=model.prompt_name,
            current_version=model.current_version,
            proposed_prompt=model.proposed_prompt,
            proposed_langfuse_version=model.proposed_langfuse_version,
            status=ProposalStatus(model.status),
            test_experiment_id=model.test_experiment_id,
            improvement_metrics=json_deserializer(model.improvement_metrics_json) or {},
            created_by=model.created_by or "auto",
            created_at=model.created_at,
            updated_at=model.updated_at,
            deployed_at=model.deployed_at,
        )
