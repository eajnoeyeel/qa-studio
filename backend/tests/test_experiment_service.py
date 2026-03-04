"""Regression tests for ExperimentService winner/ambiguity logic."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from app.services.experiment import ExperimentService
from app.models.schemas import ExperimentConfig, DatasetSplit


def test_gate_mismatch_not_ambiguous_and_passing_gate_wins():
    """Gate-pass vs gate-fail should produce a deterministic winner."""
    service = ExperimentService(
        pipeline_a=MagicMock(),
        pipeline_b=MagicMock(),
        instrumentation=MagicMock(),
        db_session=MagicMock(),
    )

    result = service._compare_results(
        item_id="item-1",
        result_a={
            "evaluation_id": "eval-a",
            "gate_failed": False,
            "scores": {"instruction_following": 3, "completeness": 3},
        },
        result_b={
            "evaluation_id": "eval-b",
            "gate_failed": True,
            "scores": {"instruction_following": 4, "completeness": 4},
        },
        sampling_config={"ab_ambiguous_threshold": 2.0},
    )

    assert result.is_ambiguous is False
    assert result.winner == "A"


@pytest.mark.asyncio
async def test_run_experiment_skips_items_with_incomplete_arm_results(monkeypatch):
    """If either arm fails/misses evaluation_id, result rows should not be persisted."""
    pipeline_a = MagicMock()
    pipeline_b = MagicMock()
    instrumentation = MagicMock()
    instrumentation.create_trace.return_value = MagicMock()

    item = SimpleNamespace(id="item-1")
    item_repo = MagicMock()
    item_repo.get_all.side_effect = [([], 1), ([item], 1)]

    exp_repo = MagicMock()
    exp_repo.create.return_value = SimpleNamespace(id="exp-1")
    exp_repo.update_summary.return_value = SimpleNamespace(id="exp-1", summary=None)

    result_repo = MagicMock()
    queue_repo = MagicMock()
    eval_repo = MagicMock()

    monkeypatch.setattr("app.services.experiment.EvalItemRepository", lambda db: item_repo)
    monkeypatch.setattr("app.services.experiment.ExperimentRepository", lambda db: exp_repo)
    monkeypatch.setattr("app.services.experiment.ExperimentResultRepository", lambda db: result_repo)
    monkeypatch.setattr("app.services.experiment.HumanQueueRepository", lambda db: queue_repo)
    monkeypatch.setattr("app.services.experiment.EvaluationRepository", lambda db: eval_repo)

    async def _classify_item(*args, **kwargs):
        return {"label": "open_qa"}

    async def _process_a(*args, **kwargs):
        return {"error": "arm-a failed", "gate_failed": True}

    async def _process_b(*args, **kwargs):
        return {"evaluation_id": "eval-b", "gate_failed": False, "scores": {"clarity": 4}}

    pipeline_a.classify_item.side_effect = _classify_item
    pipeline_a.process_item.side_effect = _process_a
    pipeline_b.process_item.side_effect = _process_b

    service = ExperimentService(
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        instrumentation=instrumentation,
        db_session=MagicMock(),
    )

    await service.run_experiment(
        name="exp",
        dataset_split=DatasetSplit.DEV,
        docs_version="v1",
        config_a=ExperimentConfig(prompt_version="v1", model_version="mock"),
        config_b=ExperimentConfig(prompt_version="v2", model_version="mock"),
        limit=1,
    )

    result_repo.create.assert_not_called()
    queue_repo.create.assert_not_called()
