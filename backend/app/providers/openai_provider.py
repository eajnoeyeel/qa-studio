"""OpenAI LLM provider."""
import json
from typing import Any, Dict, List, Optional

from .base import LLMProvider, LLMMessage, LLMResponse
from ..core.taxonomy import TaxonomyLabel, REQUIRED_SLOTS, LABEL_DESCRIPTIONS
from ..core.rubric import GateType, ScoreType, GATE_DESCRIPTIONS, SCORE_RUBRICS


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.default_model = default_model
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Complete a chat conversation."""
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw=response,
        )

    async def classify(
        self,
        text: str,
        labels: List[str],
        label_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Classify text using OpenAI."""
        descriptions = label_descriptions or LABEL_DESCRIPTIONS

        prompt = f"""Classify the following customer support message into one of these categories.

Categories:
{chr(10).join(f'- {label}: {descriptions.get(label, descriptions.get(TaxonomyLabel(label), ""))}' for label in labels)}

Message:
{text}

Respond with a JSON object containing:
- "label": the chosen category
- "confidence": a number between 0 and 1
- "required_slots": list of information needed for this category
- "detected_slots": object with detected slot values
- "missing_slots": list of slots not found in the message

JSON Response:"""

        response = await self.complete([LLMMessage(role="user", content=prompt)])

        try:
            result = json.loads(response.content)
            # Add required slots if not present
            if "required_slots" not in result:
                try:
                    taxonomy_label = TaxonomyLabel(result["label"])
                    result["required_slots"] = REQUIRED_SLOTS.get(taxonomy_label, [])
                except ValueError:
                    result["required_slots"] = []
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "label": labels[0],
                "confidence": 0.5,
                "required_slots": [],
                "detected_slots": {},
                "missing_slots": [],
            }

    async def evaluate(
        self,
        conversation: str,
        candidate_response: str,
        rubric: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a candidate response using OpenAI."""
        context_section = ""
        if context:
            context_section = f"""
Reference Documentation (use to verify claims):
{context}
"""

        prompt = f"""You are a QA evaluator for customer support responses.

{context_section}

Customer Conversation:
{conversation}

Candidate Response to Evaluate:
{candidate_response}

Evaluation Rubric:

GATES (pass/fail):
1. policy_safety: {GATE_DESCRIPTIONS[GateType.POLICY_SAFETY]}
2. overclaim: {GATE_DESCRIPTIONS[GateType.OVERCLAIM]}

SCORES (1-5 scale):
{chr(10).join(f'{st.value}: ' + chr(10).join(f'  {score}: {desc}' for score, desc in SCORE_RUBRICS[st].items()) for st in ScoreType)}

FAILURE TAGS to consider:
intent_miss, missing_slot, no_next_step, policy_pii, overclaim, escalation_needed, tool_needed, tone_issue, contradiction, sso_admin_required, permission_model_mismatch, billing_context_missing

Respond with a JSON object:
{{
  "gates": [
    {{"gate_type": "policy_safety", "passed": true/false, "reason": "...", "evidence": "..."}},
    {{"gate_type": "overclaim", "passed": true/false, "reason": "...", "evidence": "..."}}
  ],
  "scores": [
    {{"score_type": "understanding", "score": 1-5, "justification": "..."}},
    {{"score_type": "info_strategy", "score": 1-5, "justification": "..."}},
    {{"score_type": "actionability", "score": 1-5, "justification": "..."}},
    {{"score_type": "communication", "score": 1-5, "justification": "..."}}
  ],
  "failure_tags": ["tag1", "tag2"],
  "summary_of_issue": "One sentence summary",
  "what_to_fix": "Specific improvement suggestions",
  "rag_citations": ["doc_id1", "doc_id2"]
}}

JSON Response:"""

        response = await self.complete([LLMMessage(role="user", content=prompt)])

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Return empty evaluation on parse failure
            return {
                "gates": [
                    {"gate_type": "policy_safety", "passed": True, "reason": None, "evidence": None},
                    {"gate_type": "overclaim", "passed": True, "reason": None, "evidence": None},
                ],
                "scores": [
                    {"score_type": "understanding", "score": 3, "justification": "Parse error"},
                    {"score_type": "info_strategy", "score": 3, "justification": "Parse error"},
                    {"score_type": "actionability", "score": 3, "justification": "Parse error"},
                    {"score_type": "communication", "score": 3, "justification": "Parse error"},
                ],
                "failure_tags": [],
                "summary_of_issue": "Evaluation parse error",
                "what_to_fix": "Re-run evaluation",
                "rag_citations": [],
            }
