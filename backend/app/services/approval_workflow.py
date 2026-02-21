"""Phase 6: Approval Workflow — state machine for prompt proposals."""
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.schemas import (
    ProposalStatus, PromptProposalCreate, PromptProposalInDB,
)

logger = logging.getLogger(__name__)

# Valid state transitions
TRANSITIONS = {
    ProposalStatus.PENDING: {ProposalStatus.TESTING, ProposalStatus.REJECTED},
    ProposalStatus.TESTING: {ProposalStatus.APPROVED, ProposalStatus.REJECTED},
    ProposalStatus.APPROVED: {ProposalStatus.DEPLOYED, ProposalStatus.REJECTED},
    ProposalStatus.REJECTED: set(),   # Terminal state
    ProposalStatus.DEPLOYED: set(),   # Terminal state
}


class ApprovalWorkflow:
    """
    State machine for prompt proposal lifecycle.

    Transitions:
        pending ──test──→ testing ──approve──→ approved ──deploy──→ deployed
            │                 │                     │
            └──reject──→ rejected ←──reject────────┘
    """

    def __init__(self, db_session: Session, instrumentation=None):
        self.db = db_session
        self.instrumentation = instrumentation

    # ─── CRUD ────────────────────────────────────────────────────────────────

    def create_proposal(self, data: PromptProposalCreate) -> PromptProposalInDB:
        """Create a new proposal in PENDING state."""
        from ..db.repository import ProposalRepository
        repo = ProposalRepository(self.db)
        return repo.create(data)

    def get_proposal(self, proposal_id: str) -> Optional[PromptProposalInDB]:
        """Fetch a proposal by ID."""
        from ..db.repository import ProposalRepository
        repo = ProposalRepository(self.db)
        return repo.get(proposal_id)

    def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
        limit: int = 50,
    ) -> List[PromptProposalInDB]:
        """List proposals, optionally filtered by status."""
        from ..db.repository import ProposalRepository
        repo = ProposalRepository(self.db)
        return repo.get_all(status=status, limit=limit)

    # ─── State transitions ────────────────────────────────────────────────────

    def start_test(
        self,
        proposal_id: str,
        experiment_id: str,
    ) -> PromptProposalInDB:
        """
        Move proposal from PENDING → TESTING.

        Links the proposal to an A/B or multi-comparison experiment.
        """
        return self._transition(
            proposal_id,
            ProposalStatus.TESTING,
            updates={"test_experiment_id": experiment_id},
        )

    def approve(
        self,
        proposal_id: str,
        improvement_metrics: Optional[Dict[str, Any]] = None,
    ) -> PromptProposalInDB:
        """Move proposal from TESTING → APPROVED."""
        return self._transition(
            proposal_id,
            ProposalStatus.APPROVED,
            updates={"improvement_metrics": improvement_metrics or {}},
        )

    def reject(self, proposal_id: str) -> PromptProposalInDB:
        """Move proposal from PENDING/TESTING/APPROVED → REJECTED."""
        return self._transition(proposal_id, ProposalStatus.REJECTED)

    def deploy(self, proposal_id: str) -> PromptProposalInDB:
        """
        Move proposal from APPROVED → DEPLOYED.

        If Langfuse instrumentation is available, promotes the proposed prompt
        to the 'production' label and archives the previous production version.
        """
        proposal = self._get_or_raise(proposal_id)

        if not self.instrumentation or not getattr(self.instrumentation, "enabled", False):
            raise ValueError("Langfuse is not configured; cannot deploy proposal")

        # Register & promote in Langfuse
        langfuse_version = self._deploy_to_langfuse(proposal)
        if langfuse_version is None:
            raise ValueError("Failed to deploy proposal to Langfuse")

        updates: Dict[str, Any] = {
            "deployed_at": datetime.utcnow(),
            "proposed_langfuse_version": str(langfuse_version),
        }

        return self._transition(
            proposal_id,
            ProposalStatus.DEPLOYED,
            updates=updates,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _transition(
        self,
        proposal_id: str,
        target_status: ProposalStatus,
        updates: Optional[Dict[str, Any]] = None,
    ) -> PromptProposalInDB:
        """Validate and apply a state transition."""
        proposal = self._get_or_raise(proposal_id)
        current = ProposalStatus(proposal.status)

        allowed = TRANSITIONS.get(current, set())
        if target_status not in allowed:
            raise ValueError(
                f"Invalid transition: {current.value} → {target_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        from ..db.repository import ProposalRepository
        repo = ProposalRepository(self.db)
        return repo.update_status(proposal_id, target_status, extra=updates or {})

    def _get_or_raise(self, proposal_id: str) -> PromptProposalInDB:
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")
        return proposal

    def _deploy_to_langfuse(self, proposal: PromptProposalInDB) -> Optional[int]:
        """Register the proposed prompt in Langfuse and label it 'production'."""
        try:
            result = self.instrumentation.create_prompt(
                name=proposal.prompt_name,
                prompt=proposal.proposed_prompt,
                labels=["production"],
            )
            if result:
                version = result.get("version")
                logger.info(
                    f"Deployed '{proposal.prompt_name}' v{version} to Langfuse production"
                )
                return version
        except Exception as e:
            logger.error(f"Langfuse deploy failed for proposal {proposal.id}: {e}")
        return None
