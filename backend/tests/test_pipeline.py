"""Tests for evaluation pipeline experiment safeguards."""
import pytest

from app.db.repository import EvalItemRepository, EvaluationRepository
from app.models.schemas import EvalItemCreate, EvaluationKind, RAGResult
from app.providers.mock import MockProvider
from app.services.instrumentation import LangfuseInstrumentation
from app.services.pipeline import EvaluationPipeline


class DummyRetriever:
    def get_context_for_evaluation(self, question, response, taxonomy_label, docs_version, top_k=5):
        return RAGResult(query=question, documents=[])


class RecordingProvider(MockProvider):
    def __init__(self):
        super().__init__(seed=42)
        self.classify_labels = []
        self.evaluate_labels = []

    async def classify(self, text, labels, label_descriptions=None, prompt_label="production", model=None):
        self.classify_labels.append(prompt_label)
        return await super().classify(text, labels, label_descriptions, prompt_label, model)

    async def evaluate(
        self,
        question,
        response,
        rubric,
        context=None,
        system_prompt=None,
        prompt_label="production",
        model=None,
    ):
        self.evaluate_labels.append(prompt_label)
        return await super().evaluate(question, response, rubric, context, system_prompt, prompt_label, model)


@pytest.mark.asyncio
async def test_experiment_evaluations_use_fixed_judge_prompt_and_preserve_source_item(db_session):
    item_repo = EvalItemRepository(db_session)
    eval_repo = EvaluationRepository(db_session)
    item = item_repo.create(
        EvalItemCreate(
            external_id="pipeline-exp-item",
            split="dev",
            question="Explain recursion.",
            response="Original dataset response.",
            system_prompt="Original system prompt.",
        )
    )

    provider = RecordingProvider()
    instrumentation = LangfuseInstrumentation(
        public_key="",
        secret_key="",
        host="http://localhost",
        db_session=db_session,
    )
    pipeline = EvaluationPipeline(
        provider=provider,
        retriever=DummyRetriever(),
        instrumentation=instrumentation,
        db_session=db_session,
    )

    generated_response = (
        "Step 1: Recursion solves a problem by reducing it to a smaller version of itself. "
        "Step 2: Each step keeps moving toward a base case."
    )
    result = await pipeline.process_item(
        item,
        prompt_version="candidate_v1",
        model_version="mock",
        docs_version="v1",
        response_override=generated_response,
        system_prompt_override="Be clear. Answer only from supported information.",
        evaluation_kind=EvaluationKind.EXPERIMENT,
    )

    stored_item = item_repo.get(item.id)
    stored_eval = eval_repo.get(result["evaluation_id"])

    assert provider.classify_labels == ["production"]
    assert provider.evaluate_labels == ["production"]
    assert stored_item.masked_text is None
    assert stored_eval is not None
    assert stored_eval.evaluation_kind == EvaluationKind.EXPERIMENT
    assert stored_eval.evaluated_response == generated_response
    assert stored_eval.evaluated_system_prompt == "Be clear. Answer only from supported information."
