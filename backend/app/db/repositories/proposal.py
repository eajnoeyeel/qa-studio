"""Repository for prompt proposal workflow state."""
from datetime import datetime as dt
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ...models.database import PromptProposalModel, json_deserializer, json_serializer
from ...models.schemas import PromptProposalCreate, PromptProposalInDB, ProposalStatus
from .common import generate_id


class ProposalRepository:
    """Repository for prompt proposal operations."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, proposal: PromptProposalCreate) -> PromptProposalInDB:
        model = PromptProposalModel(
            id=generate_id(),
            prompt_name=proposal.prompt_name,
            prompt_type=proposal.prompt_type,
            current_version=proposal.current_version,
            proposed_prompt=proposal.proposed_prompt,
            status="pending",
            created_by=proposal.created_by,
            created_at=dt.utcnow(),
            updated_at=dt.utcnow(),
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def get(self, proposal_id: str) -> Optional[PromptProposalInDB]:
        model = self.db.query(PromptProposalModel).filter(
            PromptProposalModel.id == proposal_id
        ).first()
        return self._to_schema(model) if model else None

    def get_all(self, status=None, limit: int = 50) -> List[PromptProposalInDB]:
        """List proposals, optionally filtered by status (str or ProposalStatus enum)."""
        query = self.db.query(PromptProposalModel)
        if status is not None:
            status_val = status.value if hasattr(status, "value") else status
            query = query.filter(PromptProposalModel.status == status_val)
        models = query.order_by(PromptProposalModel.created_at.desc()).limit(limit).all()
        return [self._to_schema(m) for m in models]

    def update_status(
        self,
        proposal_id: str,
        status,  # str or ProposalStatus enum
        extra: Optional[Dict] = None,
    ) -> Optional[PromptProposalInDB]:
        """Update proposal status and apply any extra field updates."""
        model = self.db.query(PromptProposalModel).filter(
            PromptProposalModel.id == proposal_id
        ).first()
        if not model:
            return None
        model.status = status.value if hasattr(status, "value") else status
        model.updated_at = dt.utcnow()
        for key, value in (extra or {}).items():
            if key == "test_experiment_id":
                model.test_experiment_id = value
            elif key == "proposed_langfuse_version":
                model.proposed_langfuse_version = value
            elif key == "improvement_metrics":
                model.improvement_metrics_json = json_serializer(value) if value is not None else None
            elif key == "deployed_at":
                model.deployed_at = value
        self.db.commit()
        self.db.refresh(model)
        return self._to_schema(model)

    def _to_schema(self, model: PromptProposalModel) -> PromptProposalInDB:
        return PromptProposalInDB(
            id=model.id,
            prompt_name=model.prompt_name,
            current_version=model.current_version,
            proposed_prompt=model.proposed_prompt,
            proposed_langfuse_version=model.proposed_langfuse_version,
            status=ProposalStatus(model.status),
            test_experiment_id=model.test_experiment_id,
            improvement_metrics=json_deserializer(model.improvement_metrics_json) or {},
            created_by=model.created_by or "auto",
            created_at=model.created_at,
            updated_at=model.updated_at,
            deployed_at=model.deployed_at,
        )
