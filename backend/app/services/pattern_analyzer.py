"""Phase 2: Pattern Analyzer — failure tag co-occurrence and score correlation analysis."""
import uuid
import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models.database import JudgeOutputModel, EvaluationModel, json_deserializer
from ..models.schemas import FailurePattern, PatternAnalysisResult
from ..db.repository import FailurePatternRepository

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Analyzes failure tag co-occurrence and score correlations across evaluations."""

    def __init__(self, db_session: Session):
        self.db = db_session

    async def analyze(
        self,
        prompt_version: Optional[str] = None,
        model_version: Optional[str] = None,
        min_frequency: int = 2,
        top_k: int = 10,
    ) -> PatternAnalysisResult:
        """
        Run pattern analysis across all stored evaluations.

        Returns a PatternAnalysisResult with co-occurring failure tag patterns,
        score averages per pattern, and taxonomy label distributions.
        """
        analysis_run_id = str(uuid.uuid4())

        # Query evaluations with optional version filters.
        # Use only the latest evaluation per item to avoid distortion
        # from repeated runs inflating pattern counts.
        from sqlalchemy import func
        latest_subq = self.db.query(
            EvaluationModel.item_id,
            func.max(EvaluationModel.created_at).label("max_created"),
        )
        if prompt_version:
            latest_subq = latest_subq.filter(EvaluationModel.prompt_version == prompt_version)
        if model_version:
            latest_subq = latest_subq.filter(EvaluationModel.model_version == model_version)
        latest_subq = latest_subq.group_by(EvaluationModel.item_id).subquery()

        query = self.db.query(EvaluationModel).join(
            latest_subq,
            (EvaluationModel.item_id == latest_subq.c.item_id)
            & (EvaluationModel.created_at == latest_subq.c.max_created),
        )
        if prompt_version:
            query = query.filter(EvaluationModel.prompt_version == prompt_version)
        if model_version:
            query = query.filter(EvaluationModel.model_version == model_version)
        evaluations = query.all()

        if not evaluations:
            return PatternAnalysisResult(
                analysis_run_id=analysis_run_id,
                patterns_found=0,
                top_patterns=[],
                total_evaluations_analyzed=0,
                prompt_version=prompt_version,
                model_version=model_version,
            )

        # Collect per-evaluation data
        eval_data = self._collect_eval_data(evaluations)

        # Build co-occurrence counts
        pattern_data = self._compute_patterns(eval_data, min_frequency)

        # Sort by frequency and take top_k
        sorted_patterns = sorted(pattern_data.values(), key=lambda x: -x["frequency"])[:top_k]

        # Persist to DB
        repo = FailurePatternRepository(self.db)
        if sorted_patterns:
            for p in sorted_patterns:
                p["prompt_version"] = prompt_version
                p["model_version"] = model_version
            repo.create_batch(sorted_patterns, analysis_run_id)

        # Build return value
        top_patterns = repo.get_latest(top_k)

        return PatternAnalysisResult(
            analysis_run_id=analysis_run_id,
            patterns_found=len(sorted_patterns),
            top_patterns=top_patterns,
            total_evaluations_analyzed=len(evaluations),
            prompt_version=prompt_version,
            model_version=model_version,
        )

    def _collect_eval_data(self, evaluations: List[EvaluationModel]) -> List[Dict]:
        """Collect failure tags, scores, and taxonomy label per evaluation."""
        data = []
        for eval_model in evaluations:
            judge = eval_model.judge_output
            if not judge:
                continue

            tags = json_deserializer(judge.failure_tags_json) or []
            if not tags:
                continue

            scores_raw = json_deserializer(judge.scores_json) or []
            scores = {
                s.get("score_type", ""): s.get("score", 0)
                for s in scores_raw
            }

            classification = json_deserializer(eval_model.classification_json) or {}
            label = classification.get("label", "unknown")

            data.append({
                "eval_id": eval_model.id,
                "tags": tags,
                "scores": scores,
                "taxonomy_label": label,
            })

        return data

    def _compute_patterns(
        self, eval_data: List[Dict], min_frequency: int
    ) -> Dict[str, Dict]:
        """Compute co-occurrence patterns from evaluation data."""
        # Track: pattern_key -> {frequency, score_sums, score_counts, taxonomy_counts}
        pattern_counts: Dict[str, Dict] = defaultdict(lambda: {
            "frequency": 0,
            "score_sums": defaultdict(float),
            "score_counts": defaultdict(int),
            "taxonomy_counts": defaultdict(int),
            "tags": [],
        })

        for record in eval_data:
            tags = sorted(set(record["tags"]))  # Deduplicate and sort for stable key
            scores = record["scores"]
            label = record["taxonomy_label"]

            # Include single tags and all tag pairs
            tag_sets = [(t,) for t in tags]
            if len(tags) > 1:
                tag_sets += list(combinations(tags, 2))

            for tag_set in tag_sets:
                key = "|".join(tag_set)
                p = pattern_counts[key]
                p["frequency"] += 1
                p["tags"] = list(tag_set)
                for score_type, score_val in scores.items():
                    p["score_sums"][score_type] += score_val
                    p["score_counts"][score_type] += 1
                p["taxonomy_counts"][label] += 1

        # Convert to output format, filter by min_frequency
        result = {}
        for key, p in pattern_counts.items():
            if p["frequency"] < min_frequency:
                continue
            avg_scores = {
                st: p["score_sums"][st] / p["score_counts"][st]
                for st in p["score_sums"]
                if p["score_counts"][st] > 0
            }
            result[key] = {
                "tags": p["tags"],
                "frequency": p["frequency"],
                "avg_scores": avg_scores,
                "taxonomy_labels": dict(p["taxonomy_counts"]),
            }

        return result
