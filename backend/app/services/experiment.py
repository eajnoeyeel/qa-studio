"""A/B Experiment service."""
import asyncio
import time
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from ..models.schemas import (
    ExperimentCreate, ExperimentInDB, ExperimentResult,
    ExperimentSummary, ExperimentConfig, HumanQueueReason, DatasetSplit
)
from ..core.rubric import SAMPLING_RULES
from ..db.repository import (
    EvalItemRepository, EvaluationRepository, ExperimentRepository,
    ExperimentResultRepository, HumanQueueRepository
)
from .pipeline import EvaluationPipeline
from .instrumentation import LangfuseInstrumentation

logger = logging.getLogger(__name__)


class ExperimentService:
    """Service for running A/B experiments."""

    def __init__(
        self,
        pipeline_a: EvaluationPipeline,
        pipeline_b: EvaluationPipeline,
        instrumentation: LangfuseInstrumentation,
        db_session
    ):
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
        self.instrumentation = instrumentation
        self.db_session = db_session

    async def run_experiment(
        self,
        name: str,
        dataset_split: DatasetSplit,
        docs_version: str,
        config_a: ExperimentConfig,
        config_b: ExperimentConfig,
        sampling_config: Optional[Dict[str, Any]] = None,
        item_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> ExperimentInDB:
        """Run an A/B experiment."""
        # Create experiment record
        exp_repo = ExperimentRepository(self.db_session)
        experiment = exp_repo.create(ExperimentCreate(
            name=name,
            dataset_split=dataset_split,
            docs_version=docs_version,
            config_a=config_a,
            config_b=config_b,
            sampling_config=sampling_config,
        ))

        # Get items — fetch items in the split, respecting limit
        item_repo = EvalItemRepository(self.db_session)
        if item_ids:
            items = [i for iid in item_ids if (i := item_repo.get(iid))]
        else:
            _, total = item_repo.get_all(split=dataset_split, page=1, page_size=1)
            fetch_size = min(limit, total) if limit else total
            items, _ = item_repo.get_all(split=dataset_split, page=1, page_size=max(fetch_size, 1))

        logger.info(f"Running experiment {experiment.id} on {len(items)} items")

        # Create trace for experiment
        trace = self.instrumentation.create_trace(
            trace_id=f"exp_{experiment.id}",
            name=f"experiment_{name}",
            tags=[
                f"split:{dataset_split.value}",
                f"docs:{docs_version}",
                f"config_a:{config_a.prompt_version}_{config_a.model_version}",
                f"config_b:{config_b.prompt_version}_{config_b.model_version}",
            ],
        )

        # Process items with both configs
        result_repo = ExperimentResultRepository(self.db_session)
        queue_repo = HumanQueueRepository(self.db_session)
        results = []

        semaphore = asyncio.Semaphore(5)

        async def process_one(item):
            async with semaphore:
                try:
                    # Classify once, share across both arms
                    pre_classification = await self.pipeline_a.classify_item(
                        item, trace,
                        prompt_version=config_a.prompt_version,
                        model_version=config_a.model_version,
                    )

                    result_a, result_b = await asyncio.gather(
                        self.pipeline_a.process_item(
                            item,
                            prompt_version=config_a.prompt_version,
                            model_version=config_a.model_version,
                            docs_version=docs_version,
                            sampling_config=sampling_config,
                            pre_classification=pre_classification,
                        ),
                        self.pipeline_b.process_item(
                            item,
                            prompt_version=config_b.prompt_version,
                            model_version=config_b.model_version,
                            docs_version=docs_version,
                            sampling_config=sampling_config,
                            pre_classification=pre_classification,
                        ),
                    )

                    return item.id, result_a, result_b
                except Exception as e:
                    logger.error(f"Error processing item {item.id} in experiment: {e}")
                    return None

        # Process all items concurrently
        raw_results = await asyncio.gather(*[process_one(item) for item in items])

        # Sequential DB writes for experiment results
        for entry in raw_results:
            if entry is None:
                continue
            item_id, result_a, result_b = entry
            eval_a_id = result_a.get("evaluation_id")
            eval_b_id = result_b.get("evaluation_id")
            if result_a.get("error") or result_b.get("error") or not eval_a_id or not eval_b_id:
                logger.warning(
                    "Skipping experiment result for item %s due to incomplete evaluation output (A: %s, B: %s)",
                    item_id,
                    "ok" if eval_a_id and not result_a.get("error") else "failed",
                    "ok" if eval_b_id and not result_b.get("error") else "failed",
                )
                continue
            exp_result = self._compare_results(item_id, result_a, result_b, sampling_config)
            result_repo.create(experiment.id, exp_result)
            results.append(exp_result)

            if exp_result.is_ambiguous:
                queue_repo.create(
                    item_id=item_id,
                    evaluation_id=eval_a_id,
                    reason=HumanQueueReason.AB_AMBIGUOUS,
                    priority=50,
                )

        # Calculate summary
        summary = self._calculate_summary(experiment.id, results)

        # Record in trace
        self.instrumentation.record_span(
            trace,
            "ab_compare",
            input_data={"item_count": len(items)},
            output_data={
                "gate_fail_rate_a": summary.gate_fail_rate_a,
                "gate_fail_rate_b": summary.gate_fail_rate_b,
                "human_queue_rate": summary.human_queue_rate,
            },
        )

        # Update experiment with summary
        experiment = exp_repo.update_summary(experiment.id, summary)

        self.instrumentation.flush()

        return experiment

    def _compare_results(
        self,
        item_id: str,
        result_a: Dict[str, Any],
        result_b: Dict[str, Any],
        sampling_config: Optional[Dict[str, Any]]
    ) -> ExperimentResult:
        """Compare evaluation results from A and B."""
        config = sampling_config or SAMPLING_RULES
        threshold = config.get("ab_ambiguous_threshold", 2.0)

        # Calculate score differences
        scores_a = result_a.get("scores", {})
        scores_b = result_b.get("scores", {})

        score_diff = {}
        total_diff = 0
        for key in set(scores_a.keys()) | set(scores_b.keys()):
            diff = scores_a.get(key, 0) - scores_b.get(key, 0)
            score_diff[key] = diff
            total_diff += abs(diff)

        # Calculate gate differences
        gate_diff = {
            "gate_a_failed": result_a.get("gate_failed", False),
            "gate_b_failed": result_b.get("gate_failed", False),
            "gates_same": result_a.get("gate_failed") == result_b.get("gate_failed"),
        }
        gate_mismatch = gate_diff["gate_a_failed"] != gate_diff["gate_b_failed"]

        # Determine if ambiguous
        is_ambiguous = (not gate_mismatch) and (total_diff <= threshold)

        # Determine winner
        winner = None
        if gate_mismatch:
            # Gate pass always wins over gate fail.
            winner = "A" if not gate_diff["gate_a_failed"] else "B"
        elif not is_ambiguous:
            total_a = sum(scores_a.values())
            total_b = sum(scores_b.values())
            gate_a = not result_a.get("gate_failed", False)
            gate_b = not result_b.get("gate_failed", False)

            if gate_a and not gate_b:
                winner = "A"
            elif gate_b and not gate_a:
                winner = "B"
            elif total_a > total_b:
                winner = "A"
            elif total_b > total_a:
                winner = "B"

        return ExperimentResult(
            item_id=item_id,
            eval_a_id=result_a["evaluation_id"],
            eval_b_id=result_b["evaluation_id"],
            score_diff=score_diff,
            gate_diff=gate_diff,
            is_ambiguous=is_ambiguous,
            winner=winner,
        )

    def _calculate_summary(
        self,
        experiment_id: str,
        results: List[ExperimentResult],
    ) -> ExperimentSummary:
        """Calculate experiment summary statistics."""
        eval_repo = EvaluationRepository(self.db_session)

        total = len(results)
        if total == 0:
            return ExperimentSummary(
                experiment_id=experiment_id,
                total_items=0,
                gate_fail_rate_a=0,
                gate_fail_rate_b=0,
                top_tag_delta={},
                avg_scores_a={},
                avg_scores_b={},
                completeness_distribution_a={},
                completeness_distribution_b={},
                human_queue_count=0,
                human_queue_rate=0,
            )

        # Count gate failures
        gate_fails_a = sum(1 for r in results if r.gate_diff.get("gate_a_failed", False))
        gate_fails_b = sum(1 for r in results if r.gate_diff.get("gate_b_failed", False))

        # Collect scores and tags
        scores_a_sum = defaultdict(float)
        scores_b_sum = defaultdict(float)
        scores_count = defaultdict(int)
        scores_b_count = defaultdict(int)
        tags_a = defaultdict(int)
        tags_b = defaultdict(int)
        completeness_a = defaultdict(int)
        completeness_b = defaultdict(int)

        for result in results:
            # Get evaluations
            eval_a = eval_repo.get(result.eval_a_id)
            eval_b = eval_repo.get(result.eval_b_id)

            if eval_a and eval_a.judge_output:
                for score in eval_a.judge_output.scores:
                    scores_a_sum[score.score_type] += score.score
                    scores_count[score.score_type] += 1
                    if score.score_type == "completeness":
                        completeness_a[score.score] += 1
                for tag in eval_a.judge_output.failure_tags:
                    tags_a[tag] += 1

            if eval_b and eval_b.judge_output:
                for score in eval_b.judge_output.scores:
                    scores_b_sum[score.score_type] += score.score
                    scores_b_count[score.score_type] += 1
                    if score.score_type == "completeness":
                        completeness_b[score.score] += 1
                for tag in eval_b.judge_output.failure_tags:
                    tags_b[tag] += 1

        # Calculate averages
        avg_scores_a = {k: v / scores_count[k] for k, v in scores_a_sum.items() if scores_count[k] > 0}
        avg_scores_b = {k: v / scores_b_count[k] for k, v in scores_b_sum.items() if scores_b_count[k] > 0}

        # Calculate tag deltas
        all_tags = set(tags_a.keys()) | set(tags_b.keys())
        top_tag_delta = {tag: tags_a[tag] - tags_b[tag] for tag in all_tags}

        # Human queue count
        human_queue_count = sum(1 for r in results if r.is_ambiguous)

        return ExperimentSummary(
            experiment_id=experiment_id,
            total_items=total,
            gate_fail_rate_a=gate_fails_a / total,
            gate_fail_rate_b=gate_fails_b / total,
            top_tag_delta=top_tag_delta,
            avg_scores_a=avg_scores_a,
            avg_scores_b=avg_scores_b,
            completeness_distribution_a=dict(completeness_a),
            completeness_distribution_b=dict(completeness_b),
            human_queue_count=human_queue_count,
            human_queue_rate=human_queue_count / total,
        )
