"""Pydantic schemas for API and data validation."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ============== Enums ==============

class DatasetSplit(str, Enum):
    DEV = "dev"
    TEST = "test"
    AB_EVAL = "ab_eval"


class HumanQueueReason(str, Enum):
    GATE_FAIL = "gate_fail"
    LOW_SCORE = "low_score"
    NOVEL_TAG = "novel_tag"
    AB_AMBIGUOUS = "ab_ambiguous"
    MANUAL = "manual"


class EvaluationKind(str, Enum):
    DATASET = "dataset"
    EXPERIMENT = "experiment"


# ============== EvalItem ==============

class EvalItemBase(BaseModel):
    """Base eval item schema."""
    external_id: Optional[str] = None
    system_prompt: Optional[str] = None
    question: str = Field(..., description="The question or instruction")
    response: str = Field(..., description="The response to evaluate")
    metadata: Optional[Dict[str, Any]] = None
    scenario_id: Optional[str] = None
    candidate_source: Optional[str] = None


class EvalItemCreate(EvalItemBase):
    """Schema for creating an eval item."""
    split: DatasetSplit = DatasetSplit.DEV


class EvalItemInDB(EvalItemBase):
    """Schema for eval item stored in DB."""
    id: str
    split: DatasetSplit
    created_at: datetime
    masked_text: Optional[str] = None

    class Config:
        from_attributes = True


class EvalItemListResponse(BaseModel):
    """Response for eval item list."""
    items: List[EvalItemInDB]
    total: int
    page: int
    page_size: int


class ScenarioItemsResponse(BaseModel):
    """Response for scenario items."""
    scenario_id: str
    items: List[EvalItemInDB]
    count: int


# ============== Classification ==============

class ClassificationResult(BaseModel):
    """Classification output."""
    label: str
    confidence: float = Field(ge=0, le=1)
    required_slots: List[str]
    detected_slots: Dict[str, Any]
    missing_slots: List[str]


# ============== RAG ==============

class RAGDocument(BaseModel):
    """Retrieved document."""
    doc_id: str
    title: str
    content: str
    source_url: Optional[str] = None
    version: str
    tags: List[str]
    relevance_score: float


class RAGResult(BaseModel):
    """RAG retrieval result."""
    query: str
    documents: List[RAGDocument]


# ============== Gates & Scores ==============

class GateResult(BaseModel):
    """Single gate evaluation result."""
    gate_type: str
    passed: bool
    reason: Optional[str] = None
    evidence: Optional[str] = None


class ScoreResult(BaseModel):
    """Single score evaluation result."""
    score_type: str
    score: int = Field(ge=1, le=5)
    justification: str


# ============== Judge Output ==============

class JudgeOutput(BaseModel):
    """Complete judge output."""
    gates: List[GateResult]
    scores: List[ScoreResult]
    failure_tags: List[str]
    summary_of_issue: str
    what_to_fix: str
    rag_citations: List[str]

    @property
    def gate_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def total_score(self) -> int:
        return sum(s.score for s in self.scores)


class JudgeOutputInDB(JudgeOutput):
    """Judge output stored in DB."""
    id: str
    evaluation_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ============== Evaluation ==============

class EvaluationCreate(BaseModel):
    """Schema for creating an evaluation."""
    item_id: str
    prompt_version: str
    model_version: str
    docs_version: str
    evaluation_kind: EvaluationKind = EvaluationKind.DATASET
    evaluated_question: Optional[str] = None
    evaluated_response: Optional[str] = None
    evaluated_system_prompt: Optional[str] = None


class EvaluationInDB(BaseModel):
    """Evaluation stored in DB."""
    id: str
    item_id: str
    prompt_version: str
    model_version: str
    docs_version: str
    evaluation_kind: EvaluationKind = EvaluationKind.DATASET
    evaluated_question: Optional[str] = None
    evaluated_response: Optional[str] = None
    evaluated_system_prompt: Optional[str] = None
    classification: Optional[ClassificationResult] = None
    judge_output: Optional[JudgeOutput] = None
    trace_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============== Human Queue/Review ==============

class HumanQueueItem(BaseModel):
    """Item in human review queue."""
    id: str
    item_id: str
    evaluation_id: str
    reason: HumanQueueReason
    priority: int = 0
    created_at: datetime
    reviewed: bool = False


class HumanReviewCreate(BaseModel):
    """Schema for creating human review."""
    queue_item_id: str
    evaluation_id: str
    reviewer_id: Optional[str] = None
    gold_label: Optional[str] = None
    gold_gates: Optional[Dict[str, bool]] = None
    gold_scores: Optional[Dict[str, int]] = None
    gold_tags: Optional[List[str]] = None
    notes: Optional[str] = None


class HumanReviewInDB(HumanReviewCreate):
    """Human review stored in DB."""
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ============== Experiment ==============

class ExperimentConfig(BaseModel):
    """Configuration for A/B experiment."""
    prompt_version: str
    model_version: str


class ExperimentCreate(BaseModel):
    """Schema for creating experiment."""
    name: str
    dataset_split: DatasetSplit
    docs_version: str
    config_a: ExperimentConfig
    config_b: ExperimentConfig
    sampling_config: Optional[Dict[str, Any]] = None


class ExperimentResult(BaseModel):
    """Single experiment result for an item."""
    item_id: str
    eval_a_id: str
    eval_b_id: str
    score_diff: Dict[str, int]  # score_type -> (a - b)
    gate_diff: Dict[str, bool]  # gate_type -> (same or different)
    is_ambiguous: bool
    winner: Optional[str] = None  # "A", "B", or None


class ExperimentSummary(BaseModel):
    """Experiment summary statistics."""
    experiment_id: str
    total_items: int
    gate_fail_rate_a: float
    gate_fail_rate_b: float
    top_tag_delta: Dict[str, int]  # tag -> (count_a - count_b)
    avg_scores_a: Dict[str, float]
    avg_scores_b: Dict[str, float]
    completeness_distribution_a: Dict[int, int]
    completeness_distribution_b: Dict[int, int]
    human_queue_count: int
    human_queue_rate: float


class ExperimentInDB(BaseModel):
    """Experiment stored in DB."""
    id: str
    name: str
    dataset_split: DatasetSplit
    docs_version: str
    config_a: ExperimentConfig
    config_b: ExperimentConfig
    summary: Optional[ExperimentSummary] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============== API Request/Response ==============

class IngestRequest(BaseModel):
    """Request for batch ingestion."""
    file_path: str = Field(..., min_length=1, description="Server local path (absolute or relative to cwd)")
    split: DatasetSplit = DatasetSplit.DEV


class IngestResponse(BaseModel):
    """Response for batch ingestion."""
    ingested_count: int
    split: DatasetSplit
    errors: List[str] = []


class EvaluateRunRequest(BaseModel):
    """Request for evaluation run."""
    dataset_split: DatasetSplit
    prompt_version: str
    model_version: str
    docs_version: str
    sampling_config: Optional[Dict[str, Any]] = None
    item_ids: Optional[List[str]] = None  # If None, process all in split
    limit: Optional[int] = Field(None, ge=1, le=5000, description="Max items to process")


class EvaluateRunResponse(BaseModel):
    """Response for evaluation run."""
    processed_count: int
    error_count: int = 0
    gate_fail_count: int
    human_queue_count: int
    top_tags: Dict[str, int]
    avg_scores: Dict[str, float]


class ABExperimentRequest(BaseModel):
    """Request for A/B experiment."""
    dataset_split: DatasetSplit
    docs_version: str
    config_a: ExperimentConfig
    config_b: ExperimentConfig
    sampling_config: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    item_ids: Optional[List[str]] = None
    limit: Optional[int] = Field(None, ge=1, le=5000, description="Max items to process per config")


class ABExperimentResponse(BaseModel):
    """Response for A/B experiment."""
    experiment_id: str
    summary: ExperimentSummary


class ReportSummaryResponse(BaseModel):
    """Response for report summary."""
    dataset_split: DatasetSplit
    date_range: Optional[str] = None
    total_evaluations: int
    gate_fail_rate: float
    avg_scores: Dict[str, float]
    tag_distribution: Dict[str, int]
    human_queue_stats: Dict[str, int]


# ============== Document ==============

class DocumentMeta(BaseModel):
    """Document metadata for RAG."""
    doc_id: str
    title: str
    source_url: Optional[str] = None
    version: str
    tags: List[str]
    category: str  # policies, help_center, rubrics


class DocumentInDB(DocumentMeta):
    """Document stored in DB."""
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


# ============== Pattern Analysis ==============

class FailurePattern(BaseModel):
    """A detected failure pattern (co-occurring tags)."""
    id: str
    analysis_run_id: str
    tags: List[str]
    frequency: int
    avg_scores: Dict[str, float] = {}
    taxonomy_labels: Dict[str, int] = {}  # label -> count
    dataset_split: Optional[DatasetSplit] = None
    prompt_version: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime


class PatternAnalysisResult(BaseModel):
    """Result of a pattern analysis run."""
    analysis_run_id: str
    patterns_found: int
    top_patterns: List[FailurePattern]
    total_evaluations_analyzed: int
    dataset_split: Optional[DatasetSplit] = None
    prompt_version: Optional[str] = None
    model_version: Optional[str] = None


class PatternAnalysisRequest(BaseModel):
    """Request to run pattern analysis."""
    dataset_split: Optional[DatasetSplit] = None
    prompt_version: Optional[str] = None
    model_version: Optional[str] = None
    min_frequency: int = Field(default=2, ge=1)
    top_k: int = Field(default=10, ge=1, le=50)


# ============== Prompt Suggestions ==============

class PromptSuggestion(BaseModel):
    """A suggested prompt improvement."""
    id: str
    prompt_name: str
    current_prompt_summary: str
    suggested_prompt: str
    rationale: str
    target_patterns: List[str]  # Pattern IDs this targets
    expected_improvement: str
    coverage: Dict[str, str] = {}
    created_at: datetime


class SuggestionGenerateRequest(BaseModel):
    """Request to generate prompt suggestions."""
    prompt_name: str = "system_prompt"
    dataset_split: Optional[DatasetSplit] = None
    top_k_patterns: int = Field(default=5, ge=1, le=20)
    register_in_langfuse: bool = False


# ============== Multi-Comparison ==============

class MultiExperimentConfig(BaseModel):
    """Single config in a multi-comparison."""
    config_id: str
    prompt_version: str
    model_version: str
    label: Optional[str] = None  # Human-readable label


class MultiComparisonRequest(BaseModel):
    """Request to run N-way comparison."""
    name: str
    dataset_split: DatasetSplit
    docs_version: str
    configs: List[MultiExperimentConfig] = Field(..., min_length=2)
    item_ids: Optional[List[str]] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)


class ConfigRanking(BaseModel):
    """Ranking of one config in a multi-comparison."""
    config_id: str
    label: Optional[str] = None
    rank: int
    total_score: float
    avg_scores: Dict[str, float]
    gate_fail_rate: float
    win_count: int
    win_rate: float


class MultiComparisonSummary(BaseModel):
    """Summary of a multi-comparison experiment."""
    experiment_id: str
    experiment_name: str
    total_items: int
    config_rankings: List[ConfigRanking]
    winner_config_id: str
    created_at: datetime


# ============== Approval Workflow ==============

class ProposalStatus(str, Enum):
    PENDING = "pending"
    TESTING = "testing"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


class PromptProposalCreate(BaseModel):
    """Schema for creating a prompt proposal."""
    prompt_name: str
    prompt_type: str = "system_prompt"  # "system_prompt" | "judge_prompt" — prevents accidental judge modifications
    current_version: Optional[str] = None
    current_prompt: Optional[str] = None
    proposed_prompt: str
    created_by: str = "auto"


class ImprovementCycleRequest(BaseModel):
    """Request to run a self-improvement cycle."""
    dataset_split: DatasetSplit = DatasetSplit.DEV
    limit: Optional[int] = Field(50, ge=1, le=5000)
    prompt_name: str = "system_prompt"
    top_k_patterns: int = Field(default=5, ge=1, le=20)


class ImprovementCycleResponse(BaseModel):
    """Response from a self-improvement cycle."""
    proposal_id: str
    patterns_found: int
    suggestion_rationale: str
    experiment_id: Optional[str] = None
    langfuse_experiment_url: Optional[str] = None
    avg_scores_baseline: Dict[str, float] = {}
    avg_scores_candidate: Dict[str, float] = {}


class PromptProposalInDB(BaseModel):
    """Prompt proposal stored in DB."""
    id: str
    prompt_name: str
    current_version: Optional[str] = None
    current_prompt: Optional[str] = None
    proposed_prompt: str
    proposed_langfuse_version: Optional[str] = None
    status: ProposalStatus
    test_experiment_id: Optional[str] = None
    improvement_metrics: Dict[str, Any] = {}
    created_by: str
    created_at: datetime
    updated_at: datetime
    deployed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
