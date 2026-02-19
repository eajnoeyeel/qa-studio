"""Evaluation pipeline service."""
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import logging

from ..models.schemas import (
    TicketInDB, EvaluationCreate, EvaluationInDB, ClassificationResult,
    JudgeOutput, GateResult, ScoreResult, HumanQueueReason, RAGResult, Message
)
from ..core.taxonomy import TaxonomyLabel, FailureTag
from ..core.rubric import SAMPLING_RULES
from ..providers.base import LLMProvider
from ..rag.retriever import RAGRetriever
from .instrumentation import LangfuseInstrumentation

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main evaluation pipeline orchestrating all steps."""

    def __init__(
        self,
        provider: LLMProvider,
        retriever: RAGRetriever,
        instrumentation: LangfuseInstrumentation,
        db_session
    ):
        self.provider = provider
        self.retriever = retriever
        self.instrumentation = instrumentation
        self.db_session = db_session
        self.seen_tags: Set[str] = set()

    async def process_ticket(
        self,
        ticket: TicketInDB,
        prompt_version: str,
        model_version: str,
        docs_version: str,
        sampling_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a single ticket through the evaluation pipeline."""
        from ..db.repository import (
            TicketRepository, EvaluationRepository, JudgeOutputRepository,
            HumanQueueRepository
        )

        # Create trace
        trace_id = f"eval_{ticket.id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        trace = self.instrumentation.create_trace(
            trace_id=trace_id,
            name=f"evaluate_ticket_{ticket.id}",
            tags=[
                f"split:{ticket.split.value}",
                f"prompt:{prompt_version}",
                f"model:{model_version}",
                f"docs:{docs_version}",
            ],
            metadata={
                "ticket_id": ticket.id,
                "prompt_version": prompt_version,
                "model_version": model_version,
                "docs_version": docs_version,
            }
        )

        start_time = time.time()
        result = {
            "ticket_id": ticket.id,
            "trace_id": trace_id,
            "gate_failed": False,
            "human_queued": False,
            "tags": [],
            "scores": {},
        }

        try:
            # Step 1: Normalize
            normalized_text = await self._normalize(ticket, trace)

            # Step 2: Mask PII
            masked_text = await self._mask_pii(normalized_text, trace)

            # Update ticket with normalized/masked text
            ticket_repo = TicketRepository(self.db_session)
            ticket_repo.update_normalized(ticket.id, normalized_text, masked_text, commit=False)

            # Step 3: Classify
            classification = await self._classify(masked_text, trace)

            # Step 4: RAG Retrieve
            rag_result = await self._rag_retrieve(
                masked_text,
                ticket.candidate_response,
                classification.label,
                docs_version,
                trace
            )

            # Step 5: Judge (evaluate)
            judge_output = await self._judge(
                masked_text,
                ticket.candidate_response,
                rag_result,
                trace
            )

            # Persist evaluation, classification, and judge output atomically
            eval_repo = EvaluationRepository(self.db_session)
            evaluation = eval_repo.create(
                EvaluationCreate(
                    ticket_id=ticket.id,
                    prompt_version=prompt_version,
                    model_version=model_version,
                    docs_version=docs_version,
                ),
                trace_id=trace_id,
                commit=False,
            )

            eval_repo.update_classification(evaluation.id, classification, commit=False)

            judge_repo = JudgeOutputRepository(self.db_session)
            judge_repo.create(evaluation.id, judge_output, commit=False)

            # Record scores in Langfuse
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

            # Step 6: Sampling decision
            should_queue, queue_reason = await self._sampling_decision(
                judge_output,
                sampling_config,
                trace
            )

            if should_queue:
                queue_repo = HumanQueueRepository(self.db_session)
                queue_repo.create(
                    ticket_id=ticket.id,
                    evaluation_id=evaluation.id,
                    reason=queue_reason,
                    priority=self._calculate_priority(queue_reason, judge_output),
                    commit=False,
                )
                result["human_queued"] = True

            # Commit all DB writes atomically
            self.db_session.commit()

            # Compile result
            result["gate_failed"] = not judge_output.gate_passed
            result["tags"] = judge_output.failure_tags
            result["scores"] = {s.score_type: s.score for s in judge_output.scores}
            result["evaluation_id"] = evaluation.id
            result["classification"] = classification.label

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error processing ticket {ticket.id}: {e}")
            self.instrumentation.record_span(
                trace,
                "pipeline_error",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
            result["error"] = str(e)

        finally:
            self.instrumentation.flush()

        return result

    async def _normalize(self, ticket: TicketInDB, trace) -> str:
        """Normalize conversation to text."""
        start = time.time()

        # Convert conversation to text
        lines = []
        for msg in ticket.conversation:
            role = msg.role.upper()
            lines.append(f"[{role}]: {msg.content}")
        normalized = "\n\n".join(lines)

        self.instrumentation.record_span(
            trace,
            "normalize",
            input_data={"message_count": len(ticket.conversation)},
            output_data={"text_length": len(normalized)},
            latency_ms=(time.time() - start) * 1000,
        )

        return normalized

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

    async def _classify(self, text: str, trace) -> ClassificationResult:
        """Classify the ticket."""
        start = time.time()

        labels = [label.value for label in TaxonomyLabel]
        result = await self.provider.classify(text, labels)

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
        conversation: str,
        candidate_response: str,
        taxonomy_label: str,
        docs_version: str,
        trace
    ) -> RAGResult:
        """Retrieve relevant documents."""
        start = time.time()

        result = self.retriever.get_context_for_evaluation(
            conversation=conversation,
            candidate_response=candidate_response,
            taxonomy_label=taxonomy_label,
            top_k=5,
        )

        self.instrumentation.record_span(
            trace,
            "rag_retrieve",
            input_data={"query_length": len(conversation), "taxonomy": taxonomy_label},
            output_data={
                "doc_count": len(result.documents),
                "doc_ids": [d.doc_id for d in result.documents],
            },
            latency_ms=(time.time() - start) * 1000,
        )

        return result

    async def _judge(
        self,
        conversation: str,
        candidate_response: str,
        rag_result: RAGResult,
        trace
    ) -> JudgeOutput:
        """Evaluate the candidate response."""
        start = time.time()

        # Build context from RAG documents
        context = "\n\n".join([
            f"[{doc.title}]\n{doc.content}"
            for doc in rag_result.documents
        ]) if rag_result.documents else None

        result = await self.provider.evaluate(
            conversation=conversation,
            candidate_response=candidate_response,
            rubric={},  # Rubric is built into provider
            context=context,
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
                "conversation_length": len(conversation),
                "response_length": len(candidate_response),
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
                if score.score_type in ["understanding", "actionability"]:
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
