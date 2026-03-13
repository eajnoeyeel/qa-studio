"""Tests for Phase 2: PatternAnalyzer."""
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.db.repository import EvalItemRepository, EvaluationRepository, JudgeOutputRepository
from app.models.schemas import EvalItemCreate, EvaluationCreate, EvaluationKind, JudgeOutput
from app.services.pattern_analyzer import PatternAnalyzer
from app.models.schemas import GateResult, PatternAnalysisResult, ScoreResult


def _make_eval_model(eval_id, tags, scores, label="reasoning"):
    """Create a minimal mock EvaluationModel."""
    judge = MagicMock()
    judge.failure_tags_json = json.dumps(tags)  # proper JSON
    judge.scores_json = json.dumps([
        {"score_type": "instruction_following", "score": 3},
        {"score_type": "completeness", "score": 2},
    ])

    eval_model = MagicMock()
    eval_model.id = eval_id
    eval_model.prompt_version = "v1"
    eval_model.model_version = "mock"
    eval_model.judge_output = judge
    eval_model.classification_json = f'{{"label": "{label}"}}'
    return eval_model


class TestPatternAnalyzerUnit:
    """Unit tests for PatternAnalyzer internal methods."""

    def setup_method(self):
        self.db = MagicMock()
        self.analyzer = PatternAnalyzer(db_session=self.db)

    def test_compute_patterns_single_tag(self):
        eval_data = [
            {"eval_id": "e1", "tags": ["hallucination"], "scores": {"completeness": 2.0}, "taxonomy_label": "reasoning"},
            {"eval_id": "e2", "tags": ["hallucination"], "scores": {"completeness": 3.0}, "taxonomy_label": "math"},
            {"eval_id": "e3", "tags": ["hallucination"], "scores": {"completeness": 1.0}, "taxonomy_label": "reasoning"},
        ]
        result = self.analyzer._compute_patterns(eval_data, min_frequency=2)
        assert "hallucination" in result
        p = result["hallucination"]
        assert p["frequency"] == 3
        assert p["tags"] == ["hallucination"]
        assert abs(p["avg_scores"]["completeness"] - 2.0) < 0.01
        assert p["taxonomy_labels"]["reasoning"] == 2
        assert p["taxonomy_labels"]["math"] == 1

    def test_compute_patterns_co_occurrence(self):
        eval_data = [
            {"eval_id": "e1", "tags": ["hallucination", "logic_error"], "scores": {}, "taxonomy_label": "coding"},
            {"eval_id": "e2", "tags": ["hallucination", "logic_error"], "scores": {}, "taxonomy_label": "coding"},
            {"eval_id": "e3", "tags": ["hallucination"], "scores": {}, "taxonomy_label": "reasoning"},
        ]
        result = self.analyzer._compute_patterns(eval_data, min_frequency=2)

        # Single tag: hallucination appears in all 3 evals
        assert result["hallucination"]["frequency"] == 3
        # Single tag: logic_error appears in 2 evals
        assert result["logic_error"]["frequency"] == 2
        # Co-occurrence pair
        assert "hallucination|logic_error" in result
        assert result["hallucination|logic_error"]["frequency"] == 2

    def test_compute_patterns_min_frequency_filter(self):
        eval_data = [
            {"eval_id": "e1", "tags": ["rare_tag"], "scores": {}, "taxonomy_label": "math"},
            {"eval_id": "e2", "tags": ["common_tag"], "scores": {}, "taxonomy_label": "math"},
            {"eval_id": "e3", "tags": ["common_tag"], "scores": {}, "taxonomy_label": "math"},
        ]
        result = self.analyzer._compute_patterns(eval_data, min_frequency=2)
        assert "common_tag" in result
        assert "rare_tag" not in result  # frequency=1, filtered out

    def test_compute_patterns_empty(self):
        result = self.analyzer._compute_patterns([], min_frequency=2)
        assert result == {}

    def test_collect_eval_data_skips_no_judge(self):
        eval_model = MagicMock()
        eval_model.judge_output = None
        data = self.analyzer._collect_eval_data([eval_model])
        assert data == []

    def test_collect_eval_data_skips_no_tags(self):
        judge = MagicMock()
        judge.failure_tags_json = "[]"
        judge.scores_json = "[]"
        eval_model = MagicMock()
        eval_model.judge_output = judge
        eval_model.classification_json = '{"label": "reasoning"}'
        data = self.analyzer._collect_eval_data([eval_model])
        assert data == []

    def test_collect_eval_data_with_tags(self):
        eval_model = _make_eval_model("e1", ["hallucination", "logic_error"], {})
        eval_model.id = "e1"
        data = self.analyzer._collect_eval_data([eval_model])
        assert len(data) == 1
        assert "hallucination" in data[0]["tags"]
        assert data[0]["taxonomy_label"] == "reasoning"


class TestPatternAnalyzerAnalyze:
    """Integration-style tests for PatternAnalyzer.analyze()."""

    def setup_method(self):
        self.db = MagicMock()
        self.analyzer = PatternAnalyzer(db_session=self.db)

    @pytest.mark.asyncio
    async def test_analyze_empty_returns_empty_result(self):
        # The new query path uses subquery + join; MagicMock auto-chains
        # .group_by().subquery(), .join(), .filter(), .all()
        mock_query = MagicMock()
        mock_query.all.return_value = []
        self.db.query.return_value = mock_query

        result = await self.analyzer.analyze()
        assert isinstance(result, PatternAnalysisResult)
        assert result.patterns_found == 0
        assert result.total_evaluations_analyzed == 0
        assert result.top_patterns == []

    @pytest.mark.asyncio
    async def test_analyze_with_patterns(self):
        # Create eval models with tags
        evals = [
            _make_eval_model(f"e{i}", ["hallucination", "logic_error"], {})
            for i in range(3)
        ]
        for i, e in enumerate(evals):
            e.item_id = f"item_{i}"
            e.created_at = datetime(2025, 1, 1, i)

        # db.query() is called twice: once for the subquery (max created_at),
        # once for the main join query.  Use side_effect to return different
        # mock chains for each call.
        subq_mock = MagicMock()  # first call: subquery builder
        main_mock = MagicMock()  # second call: main query
        main_mock.all.return_value = evals
        # Any further chaining (.join, .filter) should resolve to main_mock
        main_mock.join.return_value = main_mock
        main_mock.filter.return_value = main_mock

        self.db.query.side_effect = [subq_mock, main_mock]

        # Mock FailurePatternRepository
        with patch("app.services.pattern_analyzer.FailurePatternRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.get_latest.return_value = []
            MockRepo.return_value = mock_repo

            result = await self.analyzer.analyze(min_frequency=2, top_k=5)

        assert isinstance(result, PatternAnalysisResult)
        assert result.total_evaluations_analyzed == 3
        # Patterns persisted (create_batch called)
        assert mock_repo.create_batch.called


class TestPatternAnalyzerPatternKeys:
    """Tests for pattern key generation and deduplication."""

    def setup_method(self):
        self.db = MagicMock()
        self.analyzer = PatternAnalyzer(db_session=self.db)

    def test_tags_sorted_for_stable_key(self):
        eval_data = [
            {"eval_id": "e1", "tags": ["logic_error", "hallucination"], "scores": {}, "taxonomy_label": "math"},
            {"eval_id": "e2", "tags": ["hallucination", "logic_error"], "scores": {}, "taxonomy_label": "math"},
        ]
        result = self.analyzer._compute_patterns(eval_data, min_frequency=2)
        # Order-independent key: hallucination comes before logic_error alphabetically
        assert "hallucination|logic_error" in result
        assert result["hallucination|logic_error"]["frequency"] == 2

    def test_duplicate_tags_deduplicated(self):
        eval_data = [
            {"eval_id": "e1", "tags": ["hallucination", "hallucination"], "scores": {}, "taxonomy_label": "math"},
            {"eval_id": "e2", "tags": ["hallucination"], "scores": {}, "taxonomy_label": "math"},
        ]
        result = self.analyzer._compute_patterns(eval_data, min_frequency=2)
        assert "hallucination" in result
        # No self-pair since deduplicated
        assert "hallucination|hallucination" not in result


@pytest.mark.asyncio
async def test_analyze_filters_dataset_split_and_evaluation_kind(db_session):
    item_repo = EvalItemRepository(db_session)
    eval_repo = EvaluationRepository(db_session)
    judge_repo = JudgeOutputRepository(db_session)

    dev_item = item_repo.create(
        EvalItemCreate(
            external_id="pattern-dev",
            split="dev",
            question="What is recursion?",
            response="A recursive explanation.",
        )
    )
    test_item = item_repo.create(
        EvalItemCreate(
            external_id="pattern-test",
            split="test",
            question="What is gravity?",
            response="Gravity explanation.",
        )
    )

    dataset_eval = eval_repo.create(
        EvaluationCreate(
            item_id=dev_item.id,
            prompt_version="dataset_v1",
            model_version="mock",
            docs_version="v1",
            evaluation_kind=EvaluationKind.DATASET,
        )
    )
    experiment_eval = eval_repo.create(
        EvaluationCreate(
            item_id=dev_item.id,
            prompt_version="candidate_v1",
            model_version="mock",
            docs_version="v1",
            evaluation_kind=EvaluationKind.EXPERIMENT,
        )
    )
    other_split_eval = eval_repo.create(
        EvaluationCreate(
            item_id=test_item.id,
            prompt_version="dataset_v1",
            model_version="mock",
            docs_version="v1",
            evaluation_kind=EvaluationKind.DATASET,
        )
    )

    judge_repo.create(
        dataset_eval.id,
        JudgeOutput(
            gates=[GateResult(gate_type="hallucination", passed=False)],
            scores=[ScoreResult(score_type="completeness", score=2, justification="low")],
            failure_tags=["hallucination"],
            summary_of_issue="hallucination",
            what_to_fix="stay grounded",
            rag_citations=[],
        ),
    )
    judge_repo.create(
        experiment_eval.id,
        JudgeOutput(
            gates=[GateResult(gate_type="hallucination", passed=False)],
            scores=[ScoreResult(score_type="completeness", score=2, justification="low")],
            failure_tags=["logic_error"],
            summary_of_issue="logic_error",
            what_to_fix="fix logic",
            rag_citations=[],
        ),
    )
    judge_repo.create(
        other_split_eval.id,
        JudgeOutput(
            gates=[GateResult(gate_type="hallucination", passed=False)],
            scores=[ScoreResult(score_type="completeness", score=2, justification="low")],
            failure_tags=["off_topic"],
            summary_of_issue="off_topic",
            what_to_fix="stay on topic",
            rag_citations=[],
        ),
    )

    analyzer = PatternAnalyzer(db_session=db_session)
    result = await analyzer.analyze(
        dataset_split="dev",
        evaluation_kind=EvaluationKind.DATASET,
        item_ids=[dev_item.id, test_item.id],
        min_frequency=1,
        top_k=10,
    )

    assert result.total_evaluations_analyzed == 1
    assert result.dataset_split == "dev"
    assert [pattern.tags for pattern in result.top_patterns] == [["hallucination"]]
