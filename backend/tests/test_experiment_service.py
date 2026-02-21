"""Regression tests for ExperimentService winner/ambiguity logic."""
from unittest.mock import MagicMock

from app.services.experiment import ExperimentService


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
