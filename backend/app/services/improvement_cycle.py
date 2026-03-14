"""Self-improvement cycle orchestrator.

Wires together: dataset evaluation refresh -> pattern analysis -> prompt
suggestion -> response generation A/B -> proposal.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.database import EvalItemModel, EvaluationModel
from ..models.schemas import (
    EvaluationKind,
    ExperimentConfig,
    ExperimentCreate,
    HumanQueueReason,
    ImprovementCycleRequest,
    ImprovementCycleResponse,
    PromptProposalCreate,
    SuggestionGenerateRequest,
)
from ..providers.base import LLMProvider
from .approval_workflow import ApprovalWorkflow
from .experiment import ExperimentService
from .instrumentation import LangfuseInstrumentation
from .pattern_analyzer import PatternAnalyzer
from .pipeline import EvaluationPipeline
from .prompt_suggester import PromptSuggester

logger = logging.getLogger(__name__)

DEFAULT_DOCS_VERSION = "v1"
DATASET_EVAL_FRESHNESS_HOURS = 24
EXPERIMENT_SAMPLING_CONFIG = {
    "gate_fail_to_human": False,
    "low_score_threshold": 0,
    "novel_tag_to_human": False,
    "ab_ambiguous_threshold": 2.0,
}


class ImprovementCycle:
    """Orchestrates one full self-improvement cycle."""

    def __init__(
        self,
        provider: LLMProvider,
        db_session: Session,
        instrumentation: LangfuseInstrumentation,
        session_factory=None,
    ):
        self.provider = provider
        self.db = db_session
        self.instrumentation = instrumentation
        self.session_factory = session_factory

    async def run(self, request: ImprovementCycleRequest) -> ImprovementCycleResponse:
        """Run a full improvement cycle for the system prompt."""
        refreshed_item_ids = await self._ensure_dataset_evaluations(request)

        analyzer = PatternAnalyzer(self.db)
        pattern_result = await analyzer.analyze(
            dataset_split=request.dataset_split,
            evaluation_kind=EvaluationKind.DATASET,
            item_ids=refreshed_item_ids,
            min_frequency=2,
            top_k=request.top_k_patterns,
        )
        if not pattern_result.top_patterns:
            raise ValueError("No failure patterns found in dataset evaluations.")

        logger.info(
            "Improvement cycle: found %d patterns from %d dataset evaluations",
            pattern_result.patterns_found,
            pattern_result.total_evaluations_analyzed,
        )

        current_prompt = self._get_current_system_prompt(request.prompt_name)
        suggester = PromptSuggester(
            provider=self.provider,
            db_session=self.db,
            instrumentation=self.instrumentation,
        )
        suggestions = await suggester.generate_suggestions(
            SuggestionGenerateRequest(
                prompt_name=request.prompt_name,
                dataset_split=request.dataset_split,
                top_k_patterns=request.top_k_patterns,
                register_in_langfuse=True,
            ),
            pattern_result=pattern_result,
        )
        if not suggestions:
            raise ValueError("Failed to generate a changed system prompt suggestion.")

        suggestion = suggestions[0]
        logger.info("Improvement cycle: generated suggestion %s", suggestion.id)

        experiment_result = await self._run_ab_experiment(
            request=request,
            current_prompt=current_prompt,
            candidate_prompt=suggestion.suggested_prompt,
        )

        workflow = ApprovalWorkflow(
            db_session=self.db,
            instrumentation=self.instrumentation,
        )
        proposal = workflow.create_proposal(
            PromptProposalCreate(
                prompt_name=request.prompt_name,
                prompt_type="system_prompt",
                current_version="production",
                current_prompt=current_prompt,
                proposed_prompt=suggestion.suggested_prompt,
                created_by="improvement_cycle",
            )
        )

        if experiment_result["experiment_id"]:
            try:
                workflow.start_test(
                    proposal.id,
                    experiment_id=experiment_result["experiment_id"],
                )
            except ValueError:
                pass

        langfuse_url = None
        if self.instrumentation.enabled:
            langfuse_url = self._seed_langfuse_dataset(experiment_result)

        return ImprovementCycleResponse(
            proposal_id=proposal.id,
            patterns_found=pattern_result.patterns_found,
            suggestion_rationale=suggestion.rationale,
            experiment_id=experiment_result["experiment_id"],
            langfuse_experiment_url=langfuse_url,
            avg_scores_baseline=experiment_result["avg_scores_a"],
            avg_scores_candidate=experiment_result["avg_scores_b"],
        )

    async def _ensure_dataset_evaluations(
        self,
        request: ImprovementCycleRequest,
    ) -> Optional[List[str]]:
        """Refresh dataset evaluations when coverage is missing or stale."""
        from ..api.deps import get_rag_retriever
        from ..db.repository import EvalItemRepository

        item_repo = EvalItemRepository(self.db)
        _, total_items = item_repo.get_all(
            split=request.dataset_split,
            page=1,
            page_size=1,
        )
        if total_items == 0:
            raise ValueError(f"No items found in split '{request.dataset_split.value}'.")

        required = min(request.limit or total_items, total_items)
        items, _ = item_repo.get_all(
            split=request.dataset_split,
            page=1,
            page_size=max(required, 1),
        )
        sample_items = items[:required]
        sample_item_ids = [item.id for item in sample_items]

        coverage_query = self.db.query(func.count(func.distinct(EvaluationModel.item_id))).filter(
            EvaluationModel.evaluation_kind == EvaluationKind.DATASET,
            EvaluationModel.item_id.in_(sample_item_ids),
        )
        latest_query = self.db.query(func.max(EvaluationModel.created_at)).filter(
            EvaluationModel.evaluation_kind == EvaluationKind.DATASET,
            EvaluationModel.item_id.in_(sample_item_ids),
        )
        covered_count = coverage_query.scalar() or 0
        latest_created_at = latest_query.scalar()
        freshness_cutoff = datetime.utcnow() - timedelta(hours=DATASET_EVAL_FRESHNESS_HOURS)

        if covered_count >= len(sample_items) and latest_created_at and latest_created_at >= freshness_cutoff:
            logger.info(
                "Improvement cycle: reusing %d fresh dataset evaluations for split %s",
                covered_count,
                request.dataset_split.value,
            )
            return None

        logger.info(
            "Improvement cycle: refreshing dataset evaluations for %d items in split %s",
            len(sample_items),
            request.dataset_split.value,
        )

        retriever = get_rag_retriever()
        model = getattr(self.provider, "default_model", self.provider.name)
        prompt_version = f"dataset_eval_{int(time.time())}"
        session_id = f"dataset_refresh_{prompt_version}"
        pipeline = EvaluationPipeline(
            provider=self.provider,
            retriever=retriever,
            instrumentation=self.instrumentation,
            db_session=self.db,
            session_factory=self.session_factory,
            session_id=session_id,
        )

        semaphore = asyncio.Semaphore(20)

        async def process_one(item):
            async with semaphore:
                return await pipeline.process_item(
                    item,
                    prompt_version=prompt_version,
                    model_version=model,
                    docs_version=DEFAULT_DOCS_VERSION,
                    evaluation_kind=EvaluationKind.DATASET,
                )

        await asyncio.gather(*[process_one(item) for item in sample_items])
        self.instrumentation.flush()
        return sample_item_ids

    async def _run_ab_experiment(
        self,
        request: ImprovementCycleRequest,
        current_prompt: str,
        candidate_prompt: str,
    ) -> Dict[str, Any]:
        """Generate fresh responses with baseline/candidate prompts, then judge them."""
        from ..api.deps import get_rag_retriever
        from ..db.repository import (
            EvalItemRepository,
            ExperimentRepository,
            ExperimentResultRepository,
            HumanQueueRepository,
        )

        timestamp = int(time.time())
        baseline_label = f"baseline_{timestamp}"
        candidate_label = f"candidate_{timestamp}"
        model = getattr(self.provider, "default_model", self.provider.name)

        exp_repo = ExperimentRepository(self.db)
        experiment = exp_repo.create(
            ExperimentCreate(
                name=f"improvement_cycle_{timestamp}",
                dataset_split=request.dataset_split,
                docs_version=DEFAULT_DOCS_VERSION,
                config_a=ExperimentConfig(
                    prompt_version=baseline_label,
                    model_version=model,
                ),
                config_b=ExperimentConfig(
                    prompt_version=candidate_label,
                    model_version=model,
                ),
                sampling_config=EXPERIMENT_SAMPLING_CONFIG,
            )
        )

        item_repo = EvalItemRepository(self.db)
        _, total_items = item_repo.get_all(split=request.dataset_split, page=1, page_size=1)
        fetch_size = min(request.limit or total_items, total_items)
        items, _ = item_repo.get_all(
            split=request.dataset_split,
            page=1,
            page_size=max(fetch_size, 1),
        )
        if not items:
            return {
                "experiment_id": experiment.id,
                "avg_scores_a": {},
                "avg_scores_b": {},
            }

        retriever = get_rag_retriever()
        session_id = f"improvement_cycle_experiment_{timestamp}"
        pipeline_a = EvaluationPipeline(
            provider=self.provider,
            retriever=retriever,
            instrumentation=self.instrumentation,
            db_session=self.db,
            session_factory=self.session_factory,
            session_id=session_id,
        )
        pipeline_b = EvaluationPipeline(
            provider=self.provider,
            retriever=retriever,
            instrumentation=self.instrumentation,
            db_session=self.db,
            session_factory=self.session_factory,
            session_id=session_id,
        )
        experiment_service = ExperimentService(
            pipeline_a=pipeline_a,
            pipeline_b=pipeline_b,
            instrumentation=self.instrumentation,
            db_session=self.db,
        )
        result_repo = ExperimentResultRepository(self.db)
        queue_repo = HumanQueueRepository(self.db)

        semaphore = asyncio.Semaphore(20)
        results = []

        async def process_one(item):
            async with semaphore:
                classification_trace = self.instrumentation.create_trace(
                    trace_id=f"improve_prepare_{item.id}_{timestamp}",
                    name=f"improve_prepare · {item.id[:8]}",
                    session_id=session_id,
                    user_id=self.provider.name,
                    input={"question": item.question[:500]},
                )
                try:
                    pre_classification = await pipeline_a.classify_item(
                        item,
                        classification_trace,
                        prompt_version="production",
                        model_version=model,
                    )
                finally:
                    if hasattr(classification_trace, "end"):
                        classification_trace.end()

                baseline_response, candidate_response = await asyncio.gather(
                    self.provider.generate(
                        question=item.question,
                        system_prompt=current_prompt,
                        model=model,
                    ),
                    self.provider.generate(
                        question=item.question,
                        system_prompt=candidate_prompt,
                        model=model,
                    ),
                )

                return await asyncio.gather(
                    pipeline_a.process_item(
                        item,
                        prompt_version=baseline_label,
                        model_version=model,
                        docs_version=DEFAULT_DOCS_VERSION,
                        sampling_config=EXPERIMENT_SAMPLING_CONFIG,
                        pre_classification=pre_classification,
                        response_override=baseline_response.content,
                        system_prompt_override=current_prompt,
                        evaluation_kind=EvaluationKind.EXPERIMENT,
                    ),
                    pipeline_b.process_item(
                        item,
                        prompt_version=candidate_label,
                        model_version=model,
                        docs_version=DEFAULT_DOCS_VERSION,
                        sampling_config=EXPERIMENT_SAMPLING_CONFIG,
                        pre_classification=pre_classification,
                        response_override=candidate_response.content,
                        system_prompt_override=candidate_prompt,
                        evaluation_kind=EvaluationKind.EXPERIMENT,
                    ),
                )

        raw_results = await asyncio.gather(*[process_one(item) for item in items])

        for item, (result_a, result_b) in zip(items, raw_results):
            eval_a_id = result_a.get("evaluation_id")
            eval_b_id = result_b.get("evaluation_id")
            if result_a.get("error") or result_b.get("error") or not eval_a_id or not eval_b_id:
                logger.warning(
                    "Skipping improvement-cycle experiment result for item %s due to incomplete output",
                    item.id,
                )
                continue

            exp_result = experiment_service._compare_results(
                item.id,
                result_a,
                result_b,
                EXPERIMENT_SAMPLING_CONFIG,
            )
            result_repo.create(experiment.id, exp_result)
            results.append(exp_result)

            if exp_result.is_ambiguous:
                queue_repo.create(
                    item_id=item.id,
                    evaluation_id=eval_a_id,
                    reason=HumanQueueReason.AB_AMBIGUOUS,
                    priority=50,
                )

        summary = experiment_service._calculate_summary(experiment.id, results)
        experiment = exp_repo.update_summary(experiment.id, summary)
        self.instrumentation.flush()

        return {
            "experiment_id": experiment.id,
            "avg_scores_a": summary.avg_scores_a if summary else {},
            "avg_scores_b": summary.avg_scores_b if summary else {},
        }

    def _seed_langfuse_dataset(
        self,
        experiment_result: Dict[str, Any],
    ) -> Optional[str]:
        """Create a Langfuse dataset with experiment items for visualization."""
        timestamp = int(time.time())
        dataset_name = f"improvement_cycle_{timestamp}"

        self.instrumentation.create_dataset(
            name=dataset_name,
            description=(
                f"Improvement cycle A/B comparison "
                f"(experiment {experiment_result['experiment_id']})"
            ),
        )

        from ..core.config import get_settings

        settings = get_settings()
        base = settings.LANGFUSE_BASE_URL.rstrip("/")
        return f"{base}/datasets/{dataset_name}"

    def _get_current_system_prompt(self, prompt_name: str) -> str:
        """Get the current production system prompt."""
        from .prompt_suggester import SYSTEM_PROMPT_TEMPLATE

        prompt_obj = self.instrumentation.get_prompt(prompt_name, label="production")
        if prompt_obj and hasattr(prompt_obj, "prompt"):
            return prompt_obj.prompt
        return SYSTEM_PROMPT_TEMPLATE
