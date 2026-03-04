"""Approval workflow proposal endpoints."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...models.schemas import PromptProposalCreate, PromptProposalInDB, ProposalStatus
from ...services.approval_workflow import ApprovalWorkflow
from ...services.instrumentation import LangfuseInstrumentation
from ..deps import get_db, get_instrumentation

router = APIRouter()


@router.get("/proposals", response_model=List[PromptProposalInDB])
async def list_proposals(
    status: Optional[ProposalStatus] = None,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List prompt proposals, optionally filtered by status."""
    workflow = ApprovalWorkflow(db_session=db)
    return workflow.list_proposals(status=status, limit=limit)


@router.post("/proposals", response_model=PromptProposalInDB)
async def create_proposal(
    data: PromptProposalCreate,
    db: Session = Depends(get_db),
):
    """Create a new prompt proposal (starts in PENDING state)."""
    workflow = ApprovalWorkflow(db_session=db)
    return workflow.create_proposal(data)


@router.get("/proposals/{proposal_id}", response_model=PromptProposalInDB)
async def get_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
):
    """Get a proposal by ID."""
    workflow = ApprovalWorkflow(db_session=db)
    proposal = workflow.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return proposal


@router.post("/proposals/{proposal_id}/test", response_model=PromptProposalInDB)
async def start_proposal_test(
    proposal_id: str,
    experiment_id: str = Body(..., embed=True),
    db: Session = Depends(get_db),
):
    """Start A/B test for a proposal (PENDING → TESTING)."""
    workflow = ApprovalWorkflow(db_session=db)
    try:
        return workflow.start_test(proposal_id, experiment_id=experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/proposals/{proposal_id}/approve", response_model=PromptProposalInDB)
async def approve_proposal(
    proposal_id: str,
    improvement_metrics: Optional[Dict[str, Any]] = Body(None),
    db: Session = Depends(get_db),
):
    """Approve a proposal (TESTING → APPROVED)."""
    workflow = ApprovalWorkflow(db_session=db)
    try:
        return workflow.approve(proposal_id, improvement_metrics=improvement_metrics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/proposals/{proposal_id}/reject", response_model=PromptProposalInDB)
async def reject_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
):
    """Reject a proposal (any state → REJECTED)."""
    workflow = ApprovalWorkflow(db_session=db)
    try:
        return workflow.reject(proposal_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/proposals/{proposal_id}/deploy", response_model=PromptProposalInDB)
async def deploy_proposal(
    proposal_id: str,
    db: Session = Depends(get_db),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Deploy an approved proposal to Langfuse production label (APPROVED → DEPLOYED)."""
    workflow = ApprovalWorkflow(db_session=db, instrumentation=instrumentation)
    try:
        return workflow.deploy(proposal_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
