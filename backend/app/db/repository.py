"""Compatibility facade for repository classes.

This module keeps the original import surface (`app.db.repository`) while the
actual implementations live in focused modules under `app.db.repositories`.
"""

from .repositories import (
    BaseRepository,
    DocumentRepository,
    EvalItemRepository,
    EvaluationRepository,
    ExperimentRepository,
    ExperimentResultRepository,
    FailurePatternRepository,
    HumanQueueRepository,
    HumanReviewRepository,
    JudgeOutputRepository,
    MultiComparisonRepository,
    ProposalRepository,
    TraceLogRepository,
    generate_id,
)

__all__ = [
    "BaseRepository",
    "generate_id",
    "EvalItemRepository",
    "EvaluationRepository",
    "JudgeOutputRepository",
    "HumanQueueRepository",
    "HumanReviewRepository",
    "ExperimentRepository",
    "ExperimentResultRepository",
    "DocumentRepository",
    "TraceLogRepository",
    "FailurePatternRepository",
    "MultiComparisonRepository",
    "ProposalRepository",
]
