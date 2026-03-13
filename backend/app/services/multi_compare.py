"""Phase 5: Multi-Comparison — N-way config comparison with ranking."""
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models.schemas import (
    MultiComparisonRequest, MultiComparisonSummary, ConfigRanking,
    EvaluationKind, MultiExperimentConfig, DatasetSplit,
)
from ..providers.base import LLMProvider
from ..rag.retriever import RAGRetriever
from .instrumentation import LangfuseInstrumentation

logger = logging.getLogger(__name__)


class MultiCompareService:
    """
    Runs N-way config comparisons across a dataset split.

    Each config (prompt_version + model_version) is evaluated on the same
    items. Configs are ranked by: gate_pass first, then total_score (sum of
    4 dimensions). Win counts track head-to-head victories.
    """

    def __init__(
        self,
        db_session: Session,
        instrumentation: LangfuseInstrumentation,
    ):
        self.db = db_session
        self.instrumentation = instrumentation

    async def run_comparison(
        self,
        request: MultiComparisonRequest,
        provider_factory,  # Callable[[str, str], LLMProvider]
        retriever: RAGRetriever,
    ) -> MultiComparisonSummary:
        """
        Run N-way comparison across all items in the given dataset split.

        Args:
            request: MultiComparisonRequest with N configs and split info
            provider_factory: callable(prompt_version, model_version) -> LLMProvider
            retriever: RAGRetriever instance

        Returns:
            MultiComparisonSummary with ranked configs
        """
        from ..db.repository import EvalItemRepository, MultiComparisonRepository
        from .pipeline import EvaluationPipeline

        experiment_id = str(uuid.uuid4())
        repo = MultiComparisonRepository(self.db)

        # Fetch items
        item_repo = EvalItemRepository(self.db)
        items = item_repo.get_by_split(request.dataset_split)

        if request.item_ids:
            items = [i for i in items if i.id in set(request.item_ids)]
        if request.limit:
            items = items[:request.limit]

        if not items:
            return self._empty_summary(experiment_id, request)

        # Track per-config aggregate stats
        n = len(request.configs)
        config_stats: Dict[str, Dict[str, Any]] = {
            cfg.config_id: {
                "label": cfg.label or cfg.config_id,
                "gate_fail_count": 0,
                "total_score_sum": 0.0,
                "score_sums": {},
                "score_counts": {},
                "win_count": 0,
                "item_count": 0,
            }
            for cfg in request.configs
        }

        # Evaluate each item across all configs
        for item in items:
            item_results: Dict[str, Dict[str, Any]] = {}

            for cfg in request.configs:
                try:
                    provider = provider_factory(cfg.prompt_version, cfg.model_version)
                    pipeline = EvaluationPipeline(
                        provider=provider,
                        retriever=retriever,
                        instrumentation=self.instrumentation,
                        db_session=self.db,
                    )
                    result = await pipeline.process_item(
                        item,
                        prompt_version=cfg.prompt_version,
                        model_version=cfg.model_version,
                        docs_version=request.docs_version,
                        evaluation_kind=EvaluationKind.EXPERIMENT,
                    )
                    if result.get("error") or not result.get("evaluation_id"):
                        logger.warning(
                            "Config %s produced incomplete evaluation for item %s: %s",
                            cfg.config_id,
                            item.id,
                            result.get("error", "missing evaluation_id"),
                        )
                        result = {
                            "gate_failed": True,
                            "scores": {},
                            "tags": [],
                            "error": result.get("error", "missing evaluation_id"),
                        }
                    item_results[cfg.config_id] = result
                except Exception as e:
                    logger.warning(f"Config {cfg.config_id} failed on item {item.id}: {e}")
                    item_results[cfg.config_id] = {
                        "gate_failed": True,
                        "scores": {},
                        "tags": [],
                        "error": str(e),
                    }

            # Rank configs for this item
            rankings = self._rank_item(request.configs, item_results)
            winner_config_id = rankings[0]["config_id"] if rankings else None

            # Update win counts and aggregate stats
            if winner_config_id:
                config_stats[winner_config_id]["win_count"] += 1

            for cfg in request.configs:
                r = item_results.get(cfg.config_id, {})
                stats = config_stats[cfg.config_id]
                stats["item_count"] += 1

                if r.get("gate_failed"):
                    stats["gate_fail_count"] += 1

                scores = r.get("scores", {})
                total = sum(scores.values())
                stats["total_score_sum"] += total

                for score_type, score_val in scores.items():
                    stats["score_sums"][score_type] = stats["score_sums"].get(score_type, 0) + score_val
                    stats["score_counts"][score_type] = stats["score_counts"].get(score_type, 0) + 1

            # Persist per-item result
            try:
                repo.create_item_result(
                    experiment_id=experiment_id,
                    item_id=item.id,
                    config_results=item_results,
                    rankings=rankings,
                    winner_config_id=winner_config_id,
                )
            except Exception as e:
                logger.warning(f"Failed to persist multi-compare result for item {item.id}: {e}")

        # Build config rankings
        config_rankings = self._build_rankings(request.configs, config_stats, len(items))
        winner = config_rankings[0].config_id if config_rankings else (request.configs[0].config_id if request.configs else "")

        return MultiComparisonSummary(
            experiment_id=experiment_id,
            experiment_name=request.name,
            total_items=len(items),
            config_rankings=config_rankings,
            winner_config_id=winner,
            created_at=__import__("datetime").datetime.utcnow(),
        )

    def _rank_item(
        self,
        configs: List[MultiExperimentConfig],
        item_results: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank configs for a single item: gate_pass first, then total_score."""
        scored = []
        for cfg in configs:
            r = item_results.get(cfg.config_id, {})
            gate_passed = not r.get("gate_failed", True)
            scores = r.get("scores", {})
            total_score = sum(scores.values())
            completeness = scores.get("completeness", 0)
            scored.append({
                "config_id": cfg.config_id,
                "gate_passed": gate_passed,
                "total_score": total_score,
                "completeness": completeness,
            })

        # Sort: gate_pass desc, total_score desc, completeness desc
        scored.sort(key=lambda x: (x["gate_passed"], x["total_score"], x["completeness"]), reverse=True)

        return [
            {"config_id": s["config_id"], "rank": i + 1, "total_score": s["total_score"]}
            for i, s in enumerate(scored)
        ]

    def _build_rankings(
        self,
        configs: List[MultiExperimentConfig],
        config_stats: Dict[str, Dict[str, Any]],
        total_items: int,
    ) -> List[ConfigRanking]:
        """Build final ConfigRanking list sorted by win_rate desc, then avg total_score."""
        rankings = []
        for cfg in configs:
            stats = config_stats[cfg.config_id]
            n = max(stats["item_count"], 1)
            avg_scores = {
                st: stats["score_sums"][st] / stats["score_counts"][st]
                for st in stats["score_sums"]
                if stats["score_counts"].get(st, 0) > 0
            }
            rankings.append({
                "config_id": cfg.config_id,
                "label": cfg.label or cfg.config_id,
                "total_score": stats["total_score_sum"] / n,
                "avg_scores": avg_scores,
                "gate_fail_rate": stats["gate_fail_count"] / n,
                "win_count": stats["win_count"],
                "win_rate": stats["win_count"] / max(total_items, 1),
            })

        # Sort by win_rate desc, then total_score desc
        rankings.sort(key=lambda x: (x["win_rate"], x["total_score"]), reverse=True)

        return [
            ConfigRanking(
                config_id=r["config_id"],
                label=r["label"],
                rank=i + 1,
                total_score=r["total_score"],
                avg_scores=r["avg_scores"],
                gate_fail_rate=r["gate_fail_rate"],
                win_count=r["win_count"],
                win_rate=r["win_rate"],
            )
            for i, r in enumerate(rankings)
        ]

    def _empty_summary(
        self, experiment_id: str, request: MultiComparisonRequest
    ) -> MultiComparisonSummary:
        return MultiComparisonSummary(
            experiment_id=experiment_id,
            experiment_name=request.name,
            total_items=0,
            config_rankings=[
                ConfigRanking(
                    config_id=cfg.config_id,
                    label=cfg.label or cfg.config_id,
                    rank=i + 1,
                    total_score=0.0,
                    avg_scores={},
                    gate_fail_rate=0.0,
                    win_count=0,
                    win_rate=0.0,
                )
                for i, cfg in enumerate(request.configs)
            ],
            winner_config_id=request.configs[0].config_id if request.configs else "",
            created_at=__import__("datetime").datetime.utcnow(),
        )
