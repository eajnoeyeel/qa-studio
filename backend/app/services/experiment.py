"""A/B Experiment service."""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from ..models.schemas import (
    TicketInDB, ExperimentCreate, ExperimentInDB, ExperimentResult,
    ExperimentSummary, ExperimentConfig, HumanQueueReason, DatasetSplit
)
from ..core.rubric import SAMPLING_RULES
from ..db.repository import (
    TicketRepository, EvaluationRepository, ExperimentRepository,
    ExperimentResultRepository, HumanQueueRepository, JudgeOutputRepository
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
        ticket_ids: Optional[List[str]] = None
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

        # Get tickets
        ticket_repo = TicketRepository(self.db_session)
        if ticket_ids:
            tickets = [ticket_repo.get(tid) for tid in ticket_ids if ticket_repo.get(tid)]
        else:
            tickets, _ = ticket_repo.get_all(split=dataset_split, page=1, page_size=1000)

        logger.info(f"Running experiment {experiment.id} on {len(tickets)} tickets")

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

        # Process tickets with both configs
        result_repo = ExperimentResultRepository(self.db_session)
        queue_repo = HumanQueueRepository(self.db_session)
        results = []

        for ticket in tickets:
            try:
                # Run config A
                result_a = await self.pipeline_a.process_ticket(
                    ticket,
                    prompt_version=config_a.prompt_version,
                    model_version=config_a.model_version,
                    docs_version=docs_version,
                    sampling_config=sampling_config,
                )

                # Run config B
                result_b = await self.pipeline_b.process_ticket(
                    ticket,
                    prompt_version=config_b.prompt_version,
                    model_version=config_b.model_version,
                    docs_version=docs_version,
                    sampling_config=sampling_config,
                )

                # Compare results
                exp_result = self._compare_results(ticket.id, result_a, result_b, sampling_config)
                result_repo.create(experiment.id, exp_result)
                results.append(exp_result)

                # Queue ambiguous cases for human review
                if exp_result.is_ambiguous:
                    queue_repo.create(
                        ticket_id=ticket.id,
                        evaluation_id=result_a.get("evaluation_id", ""),
                        reason=HumanQueueReason.AB_AMBIGUOUS,
                        priority=50,
                    )

            except Exception as e:
                logger.error(f"Error processing ticket {ticket.id} in experiment: {e}")

        # Calculate summary
        summary = self._calculate_summary(experiment.id, results, tickets)

        # Record in trace
        self.instrumentation.record_span(
            trace,
            "ab_compare",
            input_data={"ticket_count": len(tickets)},
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
        ticket_id: str,
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

        # Determine if ambiguous
        is_ambiguous = (
            total_diff <= threshold or
            result_a.get("gate_failed") != result_b.get("gate_failed")
        )

        # Determine winner
        winner = None
        if not is_ambiguous:
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
            ticket_id=ticket_id,
            eval_a_id=result_a.get("evaluation_id", ""),
            eval_b_id=result_b.get("evaluation_id", ""),
            score_diff=score_diff,
            gate_diff=gate_diff,
            is_ambiguous=is_ambiguous,
            winner=winner,
        )

    def _calculate_summary(
        self,
        experiment_id: str,
        results: List[ExperimentResult],
        tickets: List[TicketInDB]
    ) -> ExperimentSummary:
        """Calculate experiment summary statistics."""
        eval_repo = EvaluationRepository(self.db_session)
        judge_repo = JudgeOutputRepository(self.db_session)

        total = len(results)
        if total == 0:
            return ExperimentSummary(
                experiment_id=experiment_id,
                total_tickets=0,
                gate_fail_rate_a=0,
                gate_fail_rate_b=0,
                top_tag_delta={},
                avg_scores_a={},
                avg_scores_b={},
                actionability_distribution_a={},
                actionability_distribution_b={},
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
        actionability_a = defaultdict(int)
        actionability_b = defaultdict(int)

        for result in results:
            # Get evaluations
            eval_a = eval_repo.get(result.eval_a_id)
            eval_b = eval_repo.get(result.eval_b_id)

            if eval_a and eval_a.judge_output:
                for score in eval_a.judge_output.scores:
                    scores_a_sum[score.score_type] += score.score
                    scores_count[score.score_type] += 1
                    if score.score_type == "actionability":
                        actionability_a[score.score] += 1
                for tag in eval_a.judge_output.failure_tags:
                    tags_a[tag] += 1

            if eval_b and eval_b.judge_output:
                for score in eval_b.judge_output.scores:
                    scores_b_sum[score.score_type] += score.score
                    scores_b_count[score.score_type] += 1
                    if score.score_type == "actionability":
                        actionability_b[score.score] += 1
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
            total_tickets=total,
            gate_fail_rate_a=gate_fails_a / total,
            gate_fail_rate_b=gate_fails_b / total,
            top_tag_delta=top_tag_delta,
            avg_scores_a=avg_scores_a,
            avg_scores_b=avg_scores_b,
            actionability_distribution_a=dict(actionability_a),
            actionability_distribution_b=dict(actionability_b),
            human_queue_count=human_queue_count,
            human_queue_rate=human_queue_count / total,
        )
