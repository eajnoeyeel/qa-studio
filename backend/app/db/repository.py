"""Repository pattern for database operations - allows SQLite/Postgres swapping."""
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Generic, List, Optional, Type, TypeVar
from sqlalchemy.orm import Session

from ..models.database import (
    Base, TicketModel, EvaluationModel, JudgeOutputModel,
    HumanQueueModel, HumanReviewModel, ExperimentModel,
    ExperimentResultModel, DocumentModel, TraceLogModel,
    json_serializer
)
from ..models.schemas import (
    TicketCreate, TicketInDB, EvaluationCreate, EvaluationInDB,
    JudgeOutput, JudgeOutputInDB, HumanQueueItem, HumanQueueReason,
    HumanReviewCreate, HumanReviewInDB, ExperimentCreate, ExperimentInDB,
    ExperimentResult, ExperimentSummary, DocumentMeta, DocumentInDB,
    DatasetSplit, Message, ClassificationResult
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


class TicketRepository:
    """Repository for ticket operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, ticket_id: str) -> Optional[TicketInDB]:
        model = self.db.query(TicketModel).filter(TicketModel.id == ticket_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_external_id(self, external_id: str) -> Optional[TicketInDB]:
        model = self.db.query(TicketModel).filter(TicketModel.external_id == external_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_all(
        self,
        split: Optional[DatasetSplit] = None,
        page: int = 1,
        page_size: int = 50
    ) -> tuple[List[TicketInDB], int]:
        query = self.db.query(TicketModel)
        if split:
            query = query.filter(TicketModel.split == split)

        total = query.count()
        models = query.offset((page - 1) * page_size).limit(page_size).all()
        return [self._to_schema(m) for m in models], total

    def get_by_split(self, split: DatasetSplit) -> List[TicketInDB]:
        models = self.db.query(TicketModel).filter(TicketModel.split == split).all()
        return [self._to_schema(m) for m in models]

    def create(self, ticket: TicketCreate, commit: bool = True) -> TicketInDB:
        model = TicketModel(
            id=generate_id(),
            external_id=ticket.external_id,
            split=ticket.split,
            conversation_json=json_serializer([m.model_dump() for m in ticket.conversation]),
            candidate_response=ticket.candidate_response,
            metadata_json=json_serializer(ticket.metadata) if ticket.metadata else None,
        )
        self.db.add(model)
        if commit:
            self.db.commit()
            self.db.refresh(model)
        return self._to_schema(model)

    def create_batch(self, tickets: List[TicketCreate]) -> int:
        """Batch create tickets in a single transaction."""
        models = []
        for ticket in tickets:
            model = TicketModel(
                id=generate_id(),
                external_id=ticket.external_id,
                split=ticket.split,
                conversation_json=json_serializer([m.model_dump() for m in ticket.conversation]),
                candidate_response=ticket.candidate_response,
                metadata_json=json_serializer(ticket.metadata) if ticket.metadata else None,
            )
            models.append(model)
        self.db.add_all(models)
        self.db.commit()
        return len(models)

    def flush(self):
        """Commit pending changes."""
        self.db.commit()

    def update_normalized(self, ticket_id: str, normalized_text: str, masked_text: str) -> Optional[TicketInDB]:
        model = self.db.query(TicketModel).filter(TicketModel.id == ticket_id).first()
        if not model:
            return None
        model.normalized_text = normalized_text
        model.masked_text = masked_text
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: TicketModel) -> TicketInDB:
        return TicketInDB(
            id=model.id,
            external_id=model.external_id,
            split=model.split,
            conversation=[Message(**m) for m in model.conversation],
            candidate_response=model.candidate_response,
            metadata=model.ticket_metadata,
            normalized_text=model.normalized_text,
            masked_text=model.masked_text,
            created_at=model.created_at,
        )


class EvaluationRepository:
    """Repository for evaluation operations."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, eval_id: str) -> Optional[EvaluationInDB]:
        model = self.db.query(EvaluationModel).filter(EvaluationModel.id == eval_id).first()
        if not model:
            return None
        return self._to_schema(model)

    def get_by_ticket(self, ticket_id: str) -> List[EvaluationInDB]:
        models = self.db.query(EvaluationModel).filter(EvaluationModel.ticket_id == ticket_id).all()
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

    def create(self, evaluation: EvaluationCreate, trace_id: Optional[str] = None) -> EvaluationInDB:
        model = EvaluationModel(
            id=generate_id(),
            ticket_id=evaluation.ticket_id,
            prompt_version=evaluation.prompt_version,
            model_version=evaluation.model_version,
            docs_version=evaluation.docs_version,
            trace_id=trace_id,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def update_classification(self, eval_id: str, classification: ClassificationResult) -> Optional[EvaluationInDB]:
        model = self.db.query(EvaluationModel).filter(EvaluationModel.id == eval_id).first()
        if not model:
            return None
        model.classification_json = json_serializer(classification.model_dump())
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
            ticket_id=model.ticket_id,
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

    def create(self, eval_id: str, output: JudgeOutput) -> JudgeOutputInDB:
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
        self.db.commit()
        self.db.refresh(model)
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

    def get_by_ticket(self, ticket_id: str) -> List[HumanQueueItem]:
        models = self.db.query(HumanQueueModel).filter(HumanQueueModel.ticket_id == ticket_id).all()
        return [self._to_schema(m) for m in models]

    def create(
        self,
        ticket_id: str,
        evaluation_id: str,
        reason: HumanQueueReason,
        priority: int = 0
    ) -> HumanQueueItem:
        model = HumanQueueModel(
            id=generate_id(),
            ticket_id=ticket_id,
            evaluation_id=evaluation_id,
            reason=reason,
            priority=priority,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def mark_reviewed(self, queue_id: str) -> bool:
        model = self.db.query(HumanQueueModel).filter(HumanQueueModel.id == queue_id).first()
        if not model:
            return False
        model.reviewed = True
        self.db.commit()
        return True

    def count_pending(self) -> int:
        return self.db.query(HumanQueueModel).filter(HumanQueueModel.reviewed == False).count()

    def _to_schema(self, model: HumanQueueModel) -> HumanQueueItem:
        return HumanQueueItem(
            id=model.id,
            ticket_id=model.ticket_id,
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

    def create(self, review: HumanReviewCreate) -> HumanReviewInDB:
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
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: HumanReviewModel) -> HumanReviewInDB:
        return HumanReviewInDB(
            id=model.id,
            queue_item_id=model.queue_item_id,
            evaluation_id=model.evaluation_id,
            reviewer_id=model.reviewer_id,
            gold_label=model.gold_label,
            gold_gates=model.gold_gates_json,
            gold_scores=model.gold_scores_json,
            gold_tags=model.gold_tags_json,
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
            ticket_id=result.ticket_id,
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
            ticket_id=model.ticket_id,
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
        error: Optional[str] = None
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
        self.db.commit()
        return model

    def get_by_trace(self, trace_id: str):
        return self.db.query(TraceLogModel).filter(TraceLogModel.trace_id == trace_id).all()
