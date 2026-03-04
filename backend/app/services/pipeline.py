"""Evaluation pipeline service."""
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import logging

from ..models.schemas import (
    EvalItemInDB, EvaluationCreate, EvaluationInDB, ClassificationResult,
    JudgeOutput, GateResult, ScoreResult, HumanQueueReason, RAGResult
)
from ..core.taxonomy import TaxonomyLabel, FailureTag
from ..core.rubric import SAMPLING_RULES
from ..providers.base import LLMProvider
from ..rag.retriever import RAGRetriever
from .instrumentation import LangfuseInstrumentation

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main evaluation pipeline orchestrating all steps."""

    # Process-level set so novel-tag tracking persists across pipeline instances
    # and isn't reset on every request (which would re-trigger novelty routing).
    _global_seen_tags: Set[str] = set()

    def __init__(
        self,
        provider: LLMProvider,
        retriever: RAGRetriever,
        instrumentation: LangfuseInstrumentation,
        db_session,
        session_factory=None,
    ):
        self.provider = provider
        self.retriever = retriever
        self.instrumentation = instrumentation
        self.db_session = db_session
        self.session_factory = session_factory
        self.seen_tags = self._global_seen_tags

    async def process_item(
        self,
        item: EvalItemInDB,
        prompt_version: str,
        model_version: str,
        docs_version: str,
        sampling_config: Optional[Dict[str, Any]] = None,
        pre_classification: Optional[ClassificationResult] = None,
    ) -> Dict[str, Any]:
        """Process a single item through the evaluation pipeline."""
        from ..db.repository import (
            EvalItemRepository, EvaluationRepository, JudgeOutputRepository,
            HumanQueueRepository
        )

        # Per-call session for concurrent safety
        if self.session_factory is not None:
            session = self.session_factory()
            owns_session = True
        else:
            session = self.db_session
            owns_session = False

        # Create trace
        trace_id = f"eval_{item.id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        trace = self.instrumentation.create_trace(
            trace_id=trace_id,
            name=f"evaluate_item_{item.id}",
            tags=[
                f"split:{item.split.value}",
                f"prompt:{prompt_version}",
                f"model:{model_version}",
                f"docs:{docs_version}",
            ],
            metadata={
                "item_id": item.id,
                "prompt_version": prompt_version,
                "model_version": model_version,
                "docs_version": docs_version,
            }
        )

        start_time = time.time()
        result = {
            "item_id": item.id,
            "trace_id": trace_id,
            # Default to failed until a valid judge result is committed.
            # This prevents silent "success" on partial/errored execution.
            "gate_failed": True,
            "human_queued": False,
            "tags": [],
            "scores": {},
        }

        try:
            # --- Compute phase (no DB writes, no locks held) ---

            # Step 1: Prepare text
            prepared_text = await self._prepare(item, trace)

            # Step 2: Mask PII
            masked_text = await self._mask_pii(prepared_text, trace)

            # Step 3: Classify (reuse pre_classification if provided)
            if pre_classification is not None:
                classification = pre_classification
            else:
                classification = await self._classify(
                    masked_text,
                    trace,
                    prompt_version=prompt_version,
                    model_version=model_version,
                )

            # Extract masked question/response for downstream steps
            # so PII-masked text is used for RAG and judge, not raw input
            masked_question = await self._mask_pii(item.question, trace)
            masked_response = await self._mask_pii(item.response, trace)

            # Step 4: RAG Retrieve
            rag_result = await self._rag_retrieve(
                masked_question,
                masked_response,
                classification.label,
                docs_version,
                trace
            )

            # Step 5: Judge (evaluate)
            judge_output = await self._judge(
                masked_question,
                masked_response,
                rag_result,
                trace,
                system_prompt=item.system_prompt,
                prompt_version=prompt_version,
                model_version=model_version,
            )

            # Step 6: Sampling decision
            should_queue, queue_reason = await self._sampling_decision(
                judge_output,
                sampling_config,
                trace
            )

            # --- DB write phase ---
            evaluation, queued_for_human = await self._persist_outputs(
                session=session,
                item=item,
                masked_text=masked_text,
                prompt_version=prompt_version,
                model_version=model_version,
                docs_version=docs_version,
                trace_id=trace_id,
                classification=classification,
                judge_output=judge_output,
                should_queue=should_queue,
                queue_reason=queue_reason,
            )

            # Record scores in Langfuse (after commit, non-blocking)
            for score in judge_output.scores:
                self.instrumentation.record_score(
                    trace,
                    f"score_{score.score_type}",
                    score.score,
                    score.justification
                )

            for gate in judge_output.gates:
                self.instrumentation.record_score(
                    trace,
                    f"gate_{gate.gate_type}",
                    1.0 if gate.passed else 0.0,
                    gate.reason
                )

            # Compile result
            result["gate_failed"] = not judge_output.gate_passed
            result["tags"] = judge_output.failure_tags
            result["scores"] = {s.score_type: s.score for s in judge_output.scores}
            result["evaluation_id"] = evaluation.id
            result["classification"] = classification.label
            result["human_queued"] = queued_for_human

            # Update trace with input/output so Langfuse UI shows meaningful data
            if hasattr(trace, 'update'):
                trace.update(
                    input={"question": item.question[:500], "response": item.response[:500]},
                    output={
                        "gate_passed": judge_output.gate_passed,
                        "scores": result["scores"],
                        "failure_tags": judge_output.failure_tags,
                        "classification": classification.label,
                        "human_queued": queued_for_human,
                    },
                )

        except Exception as e:
            session.rollback()
            logger.error(f"Error processing item {item.id}: {e}")
            self.instrumentation.record_span(
                trace,
                "pipeline_error",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
            result["gate_failed"] = True
            result["error"] = str(e)
            if hasattr(trace, 'update'):
                trace.update(
                    input={"question": item.question[:500]},
                    output={"error": str(e)},
                )

        finally:
            self.instrumentation.flush()
            if owns_session:
                session.close()

        return result

    async def _persist_outputs(
        self,
        session,
        item: EvalItemInDB,
        masked_text: str,
        prompt_version: str,
        model_version: str,
        docs_version: str,
        trace_id: str,
        classification: ClassificationResult,
        judge_output: JudgeOutput,
        should_queue: bool,
        queue_reason: Optional[HumanQueueReason],
    ):
        """Persist all pipeline outputs in a single transaction."""
        from ..db.repository import (
            EvalItemRepository, EvaluationRepository, JudgeOutputRepository, HumanQueueRepository
        )

        queued_for_human = False

        item_repo = EvalItemRepository(session)
        item_repo.update_masked(item.id, masked_text, commit=False)

        eval_repo = EvaluationRepository(session)
        evaluation = eval_repo.create(
            EvaluationCreate(
                item_id=item.id,
                prompt_version=prompt_version,
                model_version=model_version,
                docs_version=docs_version,
            ),
            trace_id=trace_id,
            commit=False,
        )

        eval_repo.update_classification(evaluation.id, classification, commit=False)

        judge_repo = JudgeOutputRepository(session)
        judge_repo.create(evaluation.id, judge_output, commit=False)

        if should_queue and queue_reason is not None:
            queue_repo = HumanQueueRepository(session)
            queue_repo.create(
                item_id=item.id,
                evaluation_id=evaluation.id,
                reason=queue_reason,
                priority=self._calculate_priority(queue_reason, judge_output),
                commit=False,
            )
            queued_for_human = True

        session.commit()
        return evaluation, queued_for_human

    async def classify_item(self, item, trace, prompt_version, model_version):
        """Classify an item (public, for sharing across A/B arms)."""
        prepared = await self._prepare(item, trace)
        masked = await self._mask_pii(prepared, trace)
        return await self._classify(masked, trace, prompt_version, model_version)

    async def _prepare(self, item: EvalItemInDB, trace) -> str:
        """Prepare text from item fields."""
        start = time.time()

        parts = []
        if item.system_prompt:
            parts.append(f"[SYSTEM]: {item.system_prompt}")
        parts.append(f"[QUESTION]: {item.question}")
        parts.append(f"[RESPONSE]: {item.response}")
        prepared = "\n\n".join(parts)

        self.instrumentation.record_span(
            trace,
            "prepare",
            input_data={"has_system_prompt": bool(item.system_prompt)},
            output_data={"text_length": len(prepared)},
            latency_ms=(time.time() - start) * 1000,
        )

        return prepared

    async def _mask_pii(self, text: str, trace) -> str:
        """Mask PII in text."""
        start = time.time()
        masked = text

        # Email pattern
        masked = re.sub(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            '[EMAIL]',
            masked
        )

        # Phone pattern (various formats)
        masked = re.sub(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            '[PHONE]',
            masked
        )

        # Credit card pattern
        masked = re.sub(
            r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
            '[CARD]',
            masked
        )

        # SSN pattern
        masked = re.sub(
            r'\d{3}[-\s]?\d{2}[-\s]?\d{4}',
            '[SSN]',
            masked
        )

        pii_found = masked != text

        self.instrumentation.record_span(
            trace,
            "mask_pii",
            input_data={"text_length": len(text)},
            output_data={"pii_found": pii_found},
            latency_ms=(time.time() - start) * 1000,
        )

        return masked

    async def _classify(
        self,
        text: str,
        trace,
        prompt_version: str,
        model_version: str,
    ) -> ClassificationResult:
        """Classify the item."""
        start = time.time()

        labels = [label.value for label in TaxonomyLabel]
        result = await self.provider.classify(
            text,
            labels,
            prompt_label=prompt_version,
            model=model_version,
        )

        classification = ClassificationResult(
            label=result["label"],
            confidence=result.get("confidence", 0.5),
            required_slots=result.get("required_slots", []),
            detected_slots=result.get("detected_slots", {}),
            missing_slots=result.get("missing_slots", []),
        )

        self.instrumentation.record_span(
            trace,
            "classify",
            input_data={"text_length": len(text)},
            output_data={
                "label": classification.label,
                "confidence": classification.confidence,
                "missing_slots": classification.missing_slots,
            },
            latency_ms=(time.time() - start) * 1000,
        )

        return classification

    async def _rag_retrieve(
        self,
        question: str,
        response: str,
        taxonomy_label: str,
        docs_version: str,
        trace
    ) -> RAGResult:
        """Retrieve relevant documents."""
        start = time.time()

        result = self.retriever.get_context_for_evaluation(
            question=question,
            response=response,
            taxonomy_label=taxonomy_label,
            docs_version=docs_version,
            top_k=5,
        )

        self.instrumentation.record_span(
            trace,
            "rag_retrieve",
            input_data={"query_length": len(question), "taxonomy": taxonomy_label},
            output_data={
                "doc_count": len(result.documents),
                "doc_ids": [d.doc_id for d in result.documents],
            },
            latency_ms=(time.time() - start) * 1000,
        )

        return result

    async def _judge(
        self,
        question: str,
        response: str,
        rag_result: RAGResult,
        trace,
        system_prompt: Optional[str] = None,
        prompt_version: str = "production",
        model_version: str = "mock",
    ) -> JudgeOutput:
        """Evaluate the response."""
        start = time.time()

        # Build context from RAG documents
        context = "\n\n".join([
            f"[{doc.title}]\n{doc.content}"
            for doc in rag_result.documents
        ]) if rag_result.documents else None

        result = await self.provider.evaluate(
            question=question,
            response=response,
            rubric={},  # Rubric is built into provider
            context=context,
            system_prompt=system_prompt,
            prompt_label=prompt_version,
            model=model_version,
        )

        # Convert to JudgeOutput
        gates = [GateResult(**g) for g in result["gates"]]
        scores = [ScoreResult(**s) for s in result["scores"]]

        judge_output = JudgeOutput(
            gates=gates,
            scores=scores,
            failure_tags=result.get("failure_tags", []),
            summary_of_issue=result.get("summary_of_issue", ""),
            what_to_fix=result.get("what_to_fix", ""),
            rag_citations=result.get("rag_citations", [d.doc_id for d in rag_result.documents[:3]]),
        )

        self.instrumentation.record_span(
            trace,
            "judge",
            input_data={
                "question_length": len(question),
                "response_length": len(response),
                "context_docs": len(rag_result.documents),
            },
            output_data={
                "gate_passed": judge_output.gate_passed,
                "total_score": judge_output.total_score,
                "failure_tags": judge_output.failure_tags,
            },
            latency_ms=(time.time() - start) * 1000,
        )

        return judge_output

    async def _sampling_decision(
        self,
        judge_output: JudgeOutput,
        sampling_config: Optional[Dict[str, Any]],
        trace
    ) -> tuple[bool, Optional[HumanQueueReason]]:
        """Decide if this evaluation should go to human queue."""
        start = time.time()
        config = sampling_config or SAMPLING_RULES

        should_queue = False
        reason = None

        # Rule 1: Gate failure
        if not judge_output.gate_passed and config.get("gate_fail_to_human", True):
            should_queue = True
            reason = HumanQueueReason.GATE_FAIL

        # Rule 2: Low scores
        if not should_queue:
            threshold = config.get("low_score_threshold", 2)
            for score in judge_output.scores:
                if score.score_type in ["instruction_following", "completeness"]:
                    if score.score <= threshold:
                        should_queue = True
                        reason = HumanQueueReason.LOW_SCORE
                        break

        # Rule 3: Novel tags
        if not should_queue and config.get("novel_tag_to_human", True):
            for tag in judge_output.failure_tags:
                if tag not in self.seen_tags:
                    should_queue = True
                    reason = HumanQueueReason.NOVEL_TAG
                    break

        # Update seen tags
        self.seen_tags.update(judge_output.failure_tags)

        self.instrumentation.record_span(
            trace,
            "sampling_decision",
            input_data={
                "gate_passed": judge_output.gate_passed,
                "scores": {s.score_type: s.score for s in judge_output.scores},
            },
            output_data={
                "should_queue": should_queue,
                "reason": reason.value if reason else None,
            },
            latency_ms=(time.time() - start) * 1000,
        )

        return should_queue, reason

    def _calculate_priority(
        self,
        reason: HumanQueueReason,
        judge_output: JudgeOutput
    ) -> int:
        """Calculate priority for human queue (higher = more urgent)."""
        priority = 0

        # Gate failures are highest priority
        if reason == HumanQueueReason.GATE_FAIL:
            priority = 100

        # Novel tags are interesting
        elif reason == HumanQueueReason.NOVEL_TAG:
            priority = 50

        # Low scores
        elif reason == HumanQueueReason.LOW_SCORE:
            # Lower scores = higher priority
            min_score = min(s.score for s in judge_output.scores)
            priority = 30 + (5 - min_score) * 5

        return priority
