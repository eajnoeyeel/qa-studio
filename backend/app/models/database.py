"""SQLAlchemy database models."""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    String, Text, create_engine, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from .schemas import DatasetSplit, HumanQueueReason


Base = declarative_base()


class JSONEncodedDict(Text):
    """Custom type for JSON-encoded dictionaries."""
    pass


def json_serializer(obj: Any) -> str:
    """JSON serializer for DB storage."""
    return json.dumps(obj, default=str)


def json_deserializer(s: str) -> Any:
    """JSON deserializer for DB retrieval."""
    if s is None:
        return None
    return json.loads(s)


class TicketModel(Base):
    """Ticket database model."""
    __tablename__ = "tickets"

    id = Column(String, primary_key=True)
    external_id = Column(String, nullable=True, index=True)
    split = Column(SQLEnum(DatasetSplit), default=DatasetSplit.DEV, index=True)
    conversation_json = Column(Text, nullable=False)  # JSON
    candidate_response = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON
    normalized_text = Column(Text, nullable=True)
    masked_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluations = relationship("EvaluationModel", back_populates="ticket")

    @property
    def conversation(self) -> List[Dict]:
        return json_deserializer(self.conversation_json) or []

    @conversation.setter
    def conversation(self, value: List[Dict]):
        self.conversation_json = json_serializer(value)

    @property
    def ticket_metadata(self) -> Optional[Dict]:
        return json_deserializer(self.metadata_json)

    @ticket_metadata.setter
    def ticket_metadata(self, value: Optional[Dict]):
        self.metadata_json = json_serializer(value) if value else None


class EvaluationModel(Base):
    """Evaluation database model."""
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey("tickets.id"), nullable=False, index=True)
    prompt_version = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False, index=True)
    docs_version = Column(String, nullable=False, index=True)
    classification_json = Column(Text, nullable=True)  # JSON
    trace_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("TicketModel", back_populates="evaluations")
    judge_output = relationship("JudgeOutputModel", back_populates="evaluation", uselist=False)

    @property
    def classification(self) -> Optional[Dict]:
        return json_deserializer(self.classification_json)

    @classification.setter
    def classification(self, value: Optional[Dict]):
        self.classification_json = json_serializer(value) if value else None


class JudgeOutputModel(Base):
    """Judge output database model."""
    __tablename__ = "judge_outputs"

    id = Column(String, primary_key=True)
    evaluation_id = Column(String, ForeignKey("evaluations.id"), nullable=False, unique=True)
    gates_json = Column(Text, nullable=False)  # JSON
    scores_json = Column(Text, nullable=False)  # JSON
    failure_tags_json = Column(Text, nullable=False)  # JSON list
    summary_of_issue = Column(Text, nullable=False)
    what_to_fix = Column(Text, nullable=False)
    rag_citations_json = Column(Text, nullable=True)  # JSON list
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluation = relationship("EvaluationModel", back_populates="judge_output")

    @property
    def gates(self) -> List[Dict]:
        return json_deserializer(self.gates_json) or []

    @gates.setter
    def gates(self, value: List[Dict]):
        self.gates_json = json_serializer(value)

    @property
    def scores(self) -> List[Dict]:
        return json_deserializer(self.scores_json) or []

    @scores.setter
    def scores(self, value: List[Dict]):
        self.scores_json = json_serializer(value)

    @property
    def failure_tags(self) -> List[str]:
        return json_deserializer(self.failure_tags_json) or []

    @failure_tags.setter
    def failure_tags(self, value: List[str]):
        self.failure_tags_json = json_serializer(value)

    @property
    def rag_citations(self) -> List[str]:
        return json_deserializer(self.rag_citations_json) or []

    @rag_citations.setter
    def rag_citations(self, value: List[str]):
        self.rag_citations_json = json_serializer(value)


class HumanQueueModel(Base):
    """Human review queue model."""
    __tablename__ = "human_queue"

    id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey("tickets.id"), nullable=False, index=True)
    evaluation_id = Column(String, ForeignKey("evaluations.id"), nullable=False)
    reason = Column(SQLEnum(HumanQueueReason), nullable=False, index=True)
    priority = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed = Column(Boolean, default=False, index=True)


class HumanReviewModel(Base):
    """Human review model."""
    __tablename__ = "human_reviews"

    id = Column(String, primary_key=True)
    queue_item_id = Column(String, ForeignKey("human_queue.id"), nullable=False)
    evaluation_id = Column(String, ForeignKey("evaluations.id"), nullable=False)
    reviewer_id = Column(String, nullable=True)
    gold_label = Column(String, nullable=True)
    gold_gates_json = Column(Text, nullable=True)  # JSON
    gold_scores_json = Column(Text, nullable=True)  # JSON
    gold_tags_json = Column(Text, nullable=True)  # JSON
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentModel(Base):
    """Experiment database model."""
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    dataset_split = Column(SQLEnum(DatasetSplit), nullable=False)
    docs_version = Column(String, nullable=False)
    config_a_json = Column(Text, nullable=False)  # JSON
    config_b_json = Column(Text, nullable=False)  # JSON
    summary_json = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    results = relationship("ExperimentResultModel", back_populates="experiment")


class ExperimentResultModel(Base):
    """Experiment result database model."""
    __tablename__ = "experiment_results"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    ticket_id = Column(String, ForeignKey("tickets.id"), nullable=False)
    eval_a_id = Column(String, ForeignKey("evaluations.id"), nullable=False)
    eval_b_id = Column(String, ForeignKey("evaluations.id"), nullable=False)
    score_diff_json = Column(Text, nullable=False)  # JSON
    gate_diff_json = Column(Text, nullable=False)  # JSON
    is_ambiguous = Column(Boolean, default=False)
    winner = Column(String, nullable=True)  # "A", "B", or None

    experiment = relationship("ExperimentModel", back_populates="results")


class DocumentModel(Base):
    """Document model for RAG."""
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_url = Column(String, nullable=True)
    version = Column(String, nullable=False, index=True)
    tags_json = Column(Text, nullable=False)  # JSON
    category = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    @property
    def tags(self) -> List[str]:
        return json_deserializer(self.tags_json) or []

    @tags.setter
    def tags(self, value: List[str]):
        self.tags_json = json_serializer(value)


class TraceLogModel(Base):
    """Trace log for Langfuse fallback."""
    __tablename__ = "traces"

    id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False, index=True)
    span_name = Column(String, nullable=False)
    input_json = Column(Text, nullable=True)
    output_json = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
def init_db(database_url: str):
    """Initialize database and create tables."""
    engine = create_engine(database_url, connect_args={"check_same_thread": False} if "sqlite" in database_url else {})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal


def get_db_session(database_url: str):
    """Get database session generator."""
    _, SessionLocal = init_db(database_url)

    def get_session():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    return get_session
