"""OpenAI LLM provider."""
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import LLMProvider, LLMMessage, LLMResponse
from ..core.taxonomy import TaxonomyLabel, REQUIRED_SLOTS, LABEL_DESCRIPTIONS
from ..core.rubric import GateType, ScoreType, GATE_DESCRIPTIONS, SCORE_RUBRICS


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini", instrumentation=None):
        self.api_key = api_key
        self.default_model = default_model
        self._client = None
        self.instrumentation = instrumentation  # LangfuseInstrumentation for prompt registry
        # Avoid repeated prompt-refresh misses from hammering Langfuse and logs.
        self._missing_prompt_until: Dict[Tuple[str, str], float] = {}
        self._missing_prompt_ttl_seconds = 60.0

    def _get_prompt_text(self, name: str, label: str = "production") -> Optional[str]:
        """Fetch compiled prompt text from Langfuse, returns None on failure."""
        key = (name, label)
        if self._missing_prompt_until.get(key, 0.0) > time.monotonic():
            return None

        if self.instrumentation:
            prompt_obj = self.instrumentation.get_prompt(name, label=label)
            if prompt_obj and hasattr(prompt_obj, "prompt"):
                self._missing_prompt_until.pop(key, None)
                return prompt_obj.prompt
            self._missing_prompt_until[key] = time.monotonic() + self._missing_prompt_ttl_seconds
        return None

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
        label_descriptions: Optional[Dict[str, str]] = None,
        prompt_label: str = "production",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify text using OpenAI. Loads prompt from Langfuse if available."""
        descriptions = label_descriptions or LABEL_DESCRIPTIONS

        # Try loading from Langfuse prompt registry first
        langfuse_prompt = self._get_prompt_text("classify", label=prompt_label)
        if langfuse_prompt:
            labels_str = "\n".join(
                f"- {lbl}: {descriptions.get(lbl, descriptions.get(TaxonomyLabel(lbl), ''))}"
                for lbl in labels
            )
            prompt = langfuse_prompt.replace("{{labels}}", labels_str).replace("{{text}}", text)
            if "json" not in prompt.lower():
                prompt += "\n\nRespond with a JSON object."
        else:
            # Hardcoded fallback
            prompt = f"""Classify the following text/question into one of these task categories.

Categories:
{chr(10).join(f'- {label}: {descriptions.get(label, descriptions.get(TaxonomyLabel(label), ""))}' for label in labels)}

Text:
{text}

Respond with a JSON object containing:
- "label": the chosen category
- "confidence": a number between 0 and 1
- "required_slots": list of information needed for this category
- "detected_slots": object with detected slot values
- "missing_slots": list of slots not found in the text

JSON Response:"""

        response = await self.complete(
            [LLMMessage(role="user", content=prompt)],
            model=model,
            response_format={"type": "json_object"},
        )

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
        question: str,
        response: str,
        rubric: Dict[str, Any],
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_label: str = "production",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a response using OpenAI. Loads prompt from Langfuse if available."""
        context_section = ""
        if context:
            context_section = f"Reference Documentation (use to verify claims):\n{context}\n"

        system_prompt_section = ""
        if system_prompt:
            system_prompt_section = f"System Prompt (instructions given to the model):\n{system_prompt}\n"

        # Try loading judge prompt from Langfuse
        langfuse_prompt = self._get_prompt_text("judge_evaluate", label=prompt_label)
        if langfuse_prompt:
            prompt = (
                langfuse_prompt
                .replace("{{context_section}}", context_section)
                .replace("{{system_prompt_section}}", system_prompt_section)
                .replace("{{question}}", question)
                .replace("{{response}}", response)
            )
            # OpenAI json_object mode requires the word "json" in the prompt
            if "json" not in prompt.lower():
                prompt += "\n\nRespond with a JSON object."
        else:
            # Hardcoded fallback
            prompt = f"""You are a QA evaluator for AI-generated responses.

{context_section}
{system_prompt_section}

Question/Instruction:
{question}

Response to Evaluate:
{response}

Evaluation Rubric:

GATES (pass/fail):
1. factual_safety: {GATE_DESCRIPTIONS[GateType.FACTUAL_SAFETY]}
2. hallucination: {GATE_DESCRIPTIONS[GateType.HALLUCINATION]}

SCORES (1-5 scale):
{chr(10).join(f'{st.value}: ' + chr(10).join(f'  {score}: {desc}' for score, desc in SCORE_RUBRICS[st].items()) for st in ScoreType)}

FAILURE TAGS to consider:
instruction_miss, incomplete_answer, hallucination, logic_error, format_violation, over_verbose, under_verbose, wrong_language, unsafe_content, citation_missing, off_topic, partial_answer

Respond with a JSON object:
{{
  "gates": [
    {{"gate_type": "factual_safety", "passed": true/false, "reason": "...", "evidence": "..."}},
    {{"gate_type": "hallucination", "passed": true/false, "reason": "...", "evidence": "..."}}
  ],
  "scores": [
    {{"score_type": "instruction_following", "score": 1-5, "justification": "..."}},
    {{"score_type": "reasoning_quality", "score": 1-5, "justification": "..."}},
    {{"score_type": "completeness", "score": 1-5, "justification": "..."}},
    {{"score_type": "clarity", "score": 1-5, "justification": "..."}}
  ],
  "failure_tags": ["tag1", "tag2"],
  "summary_of_issue": "One sentence summary",
  "what_to_fix": "Specific improvement suggestions",
  "rag_citations": ["doc_id1", "doc_id2"]
}}

JSON Response:"""

        llm_response = await self.complete(
            [LLMMessage(role="user", content=prompt)],
            model=model,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(llm_response.content)
        except json.JSONDecodeError:
            # Return empty evaluation on parse failure
            return {
                "gates": [
                    {"gate_type": "factual_safety", "passed": True, "reason": None, "evidence": None},
                    {"gate_type": "hallucination", "passed": True, "reason": None, "evidence": None},
                ],
                "scores": [
                    {"score_type": "instruction_following", "score": 3, "justification": "Parse error"},
                    {"score_type": "reasoning_quality", "score": 3, "justification": "Parse error"},
                    {"score_type": "completeness", "score": 3, "justification": "Parse error"},
                    {"score_type": "clarity", "score": 3, "justification": "Parse error"},
                ],
                "failure_tags": [],
                "summary_of_issue": "Evaluation parse error",
                "what_to_fix": "Re-run evaluation",
                "rag_citations": [],
            }
