"""Prompt registry endpoints."""
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from ...services.instrumentation import LangfuseInstrumentation
from ..deps import get_instrumentation

router = APIRouter()


@router.get("/prompts")
async def list_prompts(
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """List all prompts from Langfuse registry."""
    prompts = instrumentation.list_prompts()
    return {"prompts": prompts}


@router.post("/prompts")
async def create_prompt(
    name: str = Body(...),
    prompt: str = Body(...),
    labels: Optional[List[str]] = Body(None),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Register a prompt in Langfuse."""
    result = instrumentation.create_prompt(name=name, prompt=prompt, labels=labels)
    if result is None:
        raise HTTPException(status_code=503, detail="Langfuse not configured or unavailable")
    return result


@router.post("/prompts/{name}/label")
async def update_prompt_label(
    name: str,
    version: int = Body(...),
    new_label: str = Body(...),
    instrumentation: LangfuseInstrumentation = Depends(get_instrumentation),
):
    """Update a prompt version label (e.g., promote to 'production')."""
    success = instrumentation.update_prompt_label(name=name, version=version, new_label=new_label)
    if not success:
        raise HTTPException(status_code=503, detail="Langfuse not configured or update failed")
    return {"success": True, "name": name, "version": version, "label": new_label}
