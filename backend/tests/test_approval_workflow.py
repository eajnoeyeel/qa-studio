"""Tests for Phase 6: ApprovalWorkflow state machine."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.services.approval_workflow import ApprovalWorkflow, TRANSITIONS
from app.models.schemas import ProposalStatus, PromptProposalCreate, PromptProposalInDB


def _make_proposal(status: ProposalStatus = ProposalStatus.PENDING) -> PromptProposalInDB:
    """Create a minimal PromptProposalInDB."""
    return PromptProposalInDB(
        id="test-proposal-id",
        prompt_name="judge_evaluate",
        current_version="v1",
        proposed_prompt="Improved prompt text here.",
        proposed_langfuse_version=None,
        status=status,
        test_experiment_id=None,
        improvement_metrics={},
        created_by="auto",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        deployed_at=None,
    )


class TestTransitionTable:
    """Test the TRANSITIONS table is correctly defined."""

    def test_pending_can_transition_to_testing_or_rejected(self):
        allowed = TRANSITIONS[ProposalStatus.PENDING]
        assert ProposalStatus.TESTING in allowed
        assert ProposalStatus.REJECTED in allowed
        assert ProposalStatus.APPROVED not in allowed
        assert ProposalStatus.DEPLOYED not in allowed

    def test_testing_can_transition_to_approved_or_rejected(self):
        allowed = TRANSITIONS[ProposalStatus.TESTING]
        assert ProposalStatus.APPROVED in allowed
        assert ProposalStatus.REJECTED in allowed
        assert ProposalStatus.PENDING not in allowed

    def test_approved_can_transition_to_deployed_or_rejected(self):
        allowed = TRANSITIONS[ProposalStatus.APPROVED]
        assert ProposalStatus.DEPLOYED in allowed
        assert ProposalStatus.REJECTED in allowed

    def test_rejected_is_terminal(self):
        assert len(TRANSITIONS[ProposalStatus.REJECTED]) == 0

    def test_deployed_is_terminal(self):
        assert len(TRANSITIONS[ProposalStatus.DEPLOYED]) == 0


class TestApprovalWorkflow:
    """Unit tests for ApprovalWorkflow state machine."""

    def setup_method(self):
        self.db = MagicMock()
        self.workflow = ApprovalWorkflow(db_session=self.db)

    def test_create_proposal(self):
        mock_data = PromptProposalCreate(
            prompt_name="judge_evaluate",
            proposed_prompt="New improved prompt.",
        )
        expected = _make_proposal()
        # Patch at the db.repository module level (where ProposalRepository lives)
        with patch("app.db.repository.ProposalRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.create.return_value = expected
            MockRepo.return_value = mock_repo

            result = self.workflow.create_proposal(mock_data)

        assert result.status == ProposalStatus.PENDING
        assert result.prompt_name == "judge_evaluate"

    def test_start_test_transitions_pending_to_testing(self):
        proposal = _make_proposal(ProposalStatus.PENDING)
        testing_proposal = _make_proposal(ProposalStatus.TESTING)

        with patch.object(self.workflow, "get_proposal", return_value=proposal):
            with patch.object(self.workflow, "_transition", return_value=testing_proposal) as mock_t:
                result = self.workflow.start_test("test-proposal-id", experiment_id="exp-123")

        assert result.status == ProposalStatus.TESTING
        mock_t.assert_called_once_with(
            "test-proposal-id",
            ProposalStatus.TESTING,
            updates={"test_experiment_id": "exp-123"},
        )

    def test_approve_transitions_testing_to_approved(self):
        proposal = _make_proposal(ProposalStatus.TESTING)
        approved_proposal = _make_proposal(ProposalStatus.APPROVED)

        with patch.object(self.workflow, "get_proposal", return_value=proposal):
            with patch.object(self.workflow, "_transition", return_value=approved_proposal) as mock_t:
                result = self.workflow.approve("test-proposal-id", improvement_metrics={"delta": -0.1})

        assert result.status == ProposalStatus.APPROVED

    def test_reject_from_pending(self):
        proposal = _make_proposal(ProposalStatus.PENDING)
        rejected_proposal = _make_proposal(ProposalStatus.REJECTED)

        with patch.object(self.workflow, "get_proposal", return_value=proposal):
            with patch.object(self.workflow, "_transition", return_value=rejected_proposal):
                result = self.workflow.reject("test-proposal-id")

        assert result.status == ProposalStatus.REJECTED

    def test_reject_from_approved(self):
        proposal = _make_proposal(ProposalStatus.APPROVED)
        rejected_proposal = _make_proposal(ProposalStatus.REJECTED)

        with patch.object(self.workflow, "get_proposal", return_value=proposal):
            with patch.object(self.workflow, "_transition", return_value=rejected_proposal):
                result = self.workflow.reject("test-proposal-id")

        assert result.status == ProposalStatus.REJECTED

    def test_invalid_transition_raises_value_error(self):
        # Cannot go from PENDING → APPROVED (skipping TESTING)
        proposal = _make_proposal(ProposalStatus.PENDING)

        with patch.object(self.workflow, "get_proposal", return_value=proposal):
            with pytest.raises(ValueError, match="Invalid transition"):
                # Directly call _transition to test the guard
                self.workflow._transition("test-proposal-id", ProposalStatus.APPROVED)

    def test_transition_from_terminal_state_raises(self):
        # Cannot transition from DEPLOYED
        proposal = _make_proposal(ProposalStatus.DEPLOYED)

        with patch.object(self.workflow, "get_proposal", return_value=proposal):
            with pytest.raises(ValueError, match="Invalid transition"):
                self.workflow._transition("test-proposal-id", ProposalStatus.REJECTED)

    def test_proposal_not_found_raises(self):
        with patch.object(self.workflow, "get_proposal", return_value=None):
            with pytest.raises(ValueError, match="Proposal not found"):
                self.workflow.start_test("nonexistent-id", experiment_id="exp-123")

    def test_deploy_transitions_approved_to_deployed(self):
        proposal = _make_proposal(ProposalStatus.APPROVED)
        deployed_proposal = _make_proposal(ProposalStatus.DEPLOYED)
        mock_instrumentation = MagicMock()
        mock_instrumentation.enabled = True

        workflow = ApprovalWorkflow(db_session=self.db, instrumentation=mock_instrumentation)

        with patch.object(workflow, "get_proposal", return_value=proposal):
            with patch.object(workflow, "_deploy_to_langfuse", return_value=3):
                with patch.object(workflow, "_transition", return_value=deployed_proposal):
                    result = workflow.deploy("test-proposal-id")

        assert result.status == ProposalStatus.DEPLOYED

    def test_deploy_calls_langfuse_when_instrumentation_available(self):
        proposal = _make_proposal(ProposalStatus.APPROVED)
        deployed_proposal = _make_proposal(ProposalStatus.DEPLOYED)
        mock_instrumentation = MagicMock()
        mock_instrumentation.enabled = True
        mock_instrumentation.create_prompt.return_value = {"version": 3}

        workflow = ApprovalWorkflow(db_session=self.db, instrumentation=mock_instrumentation)

        with patch.object(workflow, "get_proposal", return_value=proposal):
            with patch.object(workflow, "_transition", return_value=deployed_proposal):
                workflow.deploy("test-proposal-id")

        mock_instrumentation.create_prompt.assert_called_once_with(
            name="judge_evaluate",
            prompt="Improved prompt text here.",
            labels=["production"],
        )

    def test_deploy_requires_langfuse_configuration(self):
        proposal = _make_proposal(ProposalStatus.APPROVED)
        workflow = ApprovalWorkflow(db_session=self.db, instrumentation=None)

        with patch.object(workflow, "get_proposal", return_value=proposal):
            with pytest.raises(ValueError, match="Langfuse is not configured"):
                workflow.deploy("test-proposal-id")


class TestApprovalWorkflowFullCycle:
    """Integration-style tests for the complete proposal lifecycle."""

    def test_full_lifecycle_pending_to_deployed(self):
        """Verify the happy path: PENDING → TESTING → APPROVED → DEPLOYED."""
        db = MagicMock()

        statuses = [
            ProposalStatus.PENDING,   # initial: start_test reads PENDING
            ProposalStatus.TESTING,   # after start_test: approve reads TESTING
            ProposalStatus.APPROVED,  # after approve: deploy reads APPROVED
        ]

        call_count = [0]

        def mock_get(proposal_id):
            idx = min(call_count[0], len(statuses) - 1)
            return _make_proposal(statuses[idx])

        def mock_update(proposal_id, status, extra=None):
            call_count[0] += 1
            return _make_proposal(status)

        with patch("app.db.repository.ProposalRepository") as MockRepo:
            mock_repo = MagicMock()
            mock_repo.get.side_effect = mock_get
            mock_repo.update_status.side_effect = mock_update
            MockRepo.return_value = mock_repo

            instrumentation = MagicMock()
            instrumentation.enabled = True
            workflow = ApprovalWorkflow(db_session=db, instrumentation=instrumentation)

            r1 = workflow.start_test("pid", experiment_id="exp-1")
            assert r1.status == ProposalStatus.TESTING

            r2 = workflow.approve("pid")
            assert r2.status == ProposalStatus.APPROVED

            with patch.object(workflow, "_deploy_to_langfuse", return_value=5):
                r3 = workflow.deploy("pid")
            assert r3.status == ProposalStatus.DEPLOYED
