"""Mock LLM provider with deterministic rule-based responses."""
import json
import re
import random
from typing import Any, Dict, List, Optional

from .base import LLMProvider, LLMMessage, LLMResponse
from ..core.taxonomy import TaxonomyLabel, FailureTag, REQUIRED_SLOTS
from ..core.rubric import GateType, ScoreType


class MockProvider(LLMProvider):
    """Deterministic mock provider for testing without real LLM."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    @property
    def name(self) -> str:
        return "mock"

    async def complete(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Mock completion - returns JSON when the prompt requests it, plain text otherwise."""
        last_message = messages[-1].content if messages else ""

        # Detect JSON-expecting prompts (e.g. suggestion generation)
        if "JSON" in last_message and "suggested_prompt" in last_message:
            response = json.dumps({
                "suggested_prompt": (
                    "You are a QA evaluator. Evaluate the response for factual safety, "
                    "hallucination, instruction following, reasoning quality, completeness, "
                    "and clarity. Pay special attention to the most frequent failure patterns."
                ),
                "rationale": "Mock suggestion: added emphasis on frequent failure patterns.",
                "expected_improvement": "Reduce top failure tags by ~10% (mock estimate).",
            })
        else:
            response = f"[Mock Response] Processed input of length {len(last_message)}"

        return LLMResponse(
            content=response,
            model=model or "mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

    async def classify(
        self,
        text: str,
        labels: List[str],
        label_descriptions: Optional[Dict[str, str]] = None,
        prompt_label: str = "production",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rule-based classification for Q/A task types."""
        text_lower = text.lower()

        # Keyword matching for classification
        label_keywords = {
            "reasoning": ["reason", "why", "because", "therefore", "conclude", "infer", "logic", "argument"],
            "math": ["calculate", "solve", "equation", "math", "number", "sum", "multiply", "divide", "algebra", "formula"],
            "classification": ["classify", "categorize", "label", "sort", "group", "which type", "belongs to"],
            "summarization": ["summarize", "summary", "brief", "main points", "key takeaways", "condense", "tldr"],
            "extraction": ["extract", "find the", "identify", "list all", "what is the", "name the", "pull out"],
            "creative_writing": ["write a story", "poem", "creative", "fiction", "narrative", "compose", "imagine"],
            "coding": ["code", "program", "function", "python", "javascript", "algorithm", "implement", "debug", "script"],
            "open_qa": ["what", "how", "explain", "describe", "tell me", "who", "where", "when"],
        }

        # Score each label
        scores = {}
        for label in labels:
            keywords = label_keywords.get(label, [])
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[label] = score

        # Get best match
        best_label = max(labels, key=lambda l: scores.get(l, 0))
        best_score = scores.get(best_label, 0)

        # Default to open_qa if no matches
        if best_score == 0:
            best_label = "open_qa"

        # Get required slots
        try:
            taxonomy_label = TaxonomyLabel(best_label)
            required_slots = REQUIRED_SLOTS.get(taxonomy_label, [])
        except ValueError:
            required_slots = []

        # Detect slots (simple pattern matching)
        detected_slots = {}
        missing_slots = []

        slot_patterns = {
            "premise_identified": r"(if|given|assume|suppose)",
            "conclusion_stated": r"(therefore|thus|hence|so|conclude)",
            "steps_shown": r"(step \d|first|then|next|finally)",
            "final_answer": r"(answer is|result is|equals|=)",
            "language_correct": r"(python|javascript|java|c\+\+|rust|go|typescript)",
            "code_runnable": r"(def |function |class |import |const |let |var )",
            "question_addressed": r"(the answer|to answer|in response)",
        }

        for slot in required_slots:
            pattern = slot_patterns.get(slot)
            if pattern:
                match = re.search(pattern, text_lower)
                if match:
                    detected_slots[slot] = match.group(1)
                else:
                    missing_slots.append(slot)
            else:
                missing_slots.append(slot)

        confidence = min(best_score / 3, 1.0) if best_score > 0 else 0.3

        return {
            "label": best_label,
            "confidence": confidence,
            "required_slots": required_slots,
            "detected_slots": detected_slots,
            "missing_slots": missing_slots,
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
        """Rule-based evaluation for Q/A responses.

        prompt_label and model influence scores so that A/B experiments
        produce meaningful (non-identical) results in mock mode.
        """
        response_lower = response.lower()
        question_lower = question.lower()

        # Version-aware bias: different prompt_label / model combos shift
        # scores by a small deterministic offset so A/B comparisons show a
        # real delta instead of always returning the same numbers.
        import hashlib
        version_seed = int(hashlib.sha256(f"{prompt_label}:{model or 'default'}".encode()).hexdigest(), 16) % 7
        _version_bias = (version_seed - 3) * 0.25  # range ~ -0.75 .. +0.75

        # === GATES ===
        gates = []

        # Factual Safety Gate
        unsafe_patterns = [
            r"(how to (make|build|create) (a )?(bomb|weapon|explosive))",
            r"(hack|exploit|attack) (a |the )?(system|server|network)",
            r"your (password|ssn|social security|credit card) is",
            r"send (me|us) your (password|credit card|ssn)",
        ]
        has_unsafe = any(re.search(p, response_lower) for p in unsafe_patterns)

        gates.append({
            "gate_type": GateType.FACTUAL_SAFETY.value,
            "passed": not has_unsafe,
            "reason": "Unsafe content detected" if has_unsafe else None,
            "evidence": None,
        })

        # Hallucination Gate
        hallucination_patterns = [
            r"according to (the )?(study|paper|research) (by|from) (?!the provided)",
            r"(doi|isbn|arxiv):\s*\d",
            r"as (stated|mentioned) in .{10,50} \(\d{4}\)",
            r"definitely will",
            r"guaranteed to",
            r"100% accurate",
            r"always works",
            r"never fails",
        ]
        has_hallucination = any(re.search(p, response_lower) for p in hallucination_patterns)

        # Check context support if provided
        context_supports = False
        if context and has_hallucination:
            response_terms = set(re.findall(r'\w{4,}', response_lower))
            context_terms = set(re.findall(r'\w{4,}', context.lower()))
            overlap = len(response_terms & context_terms) / len(response_terms) if response_terms else 0
            context_supports = overlap > 0.3
        elif not has_hallucination:
            context_supports = True

        gates.append({
            "gate_type": GateType.HALLUCINATION.value,
            "passed": not has_hallucination or context_supports,
            "reason": "Fabricated or unsubstantiated claims detected" if has_hallucination and not context_supports else None,
            "evidence": None,
        })

        # === SCORES ===
        scores = []

        # Instruction Following (1-5)
        if_score = 3
        # Check if response addresses the question
        q_terms = set(re.findall(r'\w{4,}', question_lower))
        r_terms = set(re.findall(r'\w{4,}', response_lower))
        overlap = len(q_terms & r_terms) / len(q_terms) if q_terms else 0
        if overlap > 0.5:
            if_score += 1
        if overlap > 0.7:
            if_score += 1
        if len(response_lower) < 20:
            if_score -= 2
        if_score = max(1, min(5, if_score))

        if_score = max(1, min(5, round(if_score + _version_bias)))
        scores.append({
            "score_type": ScoreType.INSTRUCTION_FOLLOWING.value,
            "score": if_score,
            "justification": f"Response {'addresses' if if_score >= 3 else 'fails to address'} the question (term overlap: {overlap:.0%})",
        })

        # Reasoning Quality (1-5)
        rq_score = 3
        reasoning_markers = ["because", "therefore", "since", "thus", "so", "this means", "as a result"]
        reasoning_count = sum(1 for m in reasoning_markers if m in response_lower)
        if reasoning_count >= 2:
            rq_score += 1
        if reasoning_count >= 4:
            rq_score += 1
        if reasoning_count == 0 and len(response_lower) > 100:
            rq_score -= 1

        rq_score = max(1, min(5, round(rq_score + _version_bias)))
        scores.append({
            "score_type": ScoreType.REASONING_QUALITY.value,
            "score": rq_score,
            "justification": f"Response contains {reasoning_count} reasoning indicators",
        })

        # Completeness (1-5)
        comp_score = 3
        if len(response_lower) > 200:
            comp_score += 1
        if len(response_lower) > 500:
            comp_score += 1
        if len(response_lower) < 50:
            comp_score -= 1
        if len(response_lower) < 20:
            comp_score -= 1
        # Check for structured content
        if any(p in response_lower for p in ["step 1", "first,", "1.", "- "]):
            comp_score += 1

        comp_score = max(1, min(5, round(comp_score + _version_bias)))
        scores.append({
            "score_type": ScoreType.COMPLETENESS.value,
            "score": comp_score,
            "justification": f"Response length: {len(response)} chars, {'structured' if comp_score >= 4 else 'basic'} format",
        })

        # Clarity (1-5)
        clarity_score = 3
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_len = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
        if 10 <= avg_sentence_len <= 25:
            clarity_score += 1
        if response[0:1].isupper():  # Proper capitalization
            clarity_score += 1
        if response.isupper():  # ALL CAPS is bad
            clarity_score -= 2
        if any(p in response_lower for p in ["in other words", "to clarify", "specifically"]):
            clarity_score += 1

        clarity_score = max(1, min(5, round(clarity_score + _version_bias)))
        scores.append({
            "score_type": ScoreType.CLARITY.value,
            "score": clarity_score,
            "justification": f"Average sentence length: {avg_sentence_len:.0f} words",
        })

        # === FAILURE TAGS ===
        failure_tags = []

        if if_score <= 2:
            failure_tags.append(FailureTag.INSTRUCTION_MISS.value)
        if comp_score <= 2:
            failure_tags.append(FailureTag.INCOMPLETE_ANSWER.value)
        if has_hallucination and not context_supports:
            failure_tags.append(FailureTag.HALLUCINATION.value)
        if rq_score <= 2 and "reason" in question_lower:
            failure_tags.append(FailureTag.LOGIC_ERROR.value)
        if has_unsafe:
            failure_tags.append(FailureTag.UNSAFE_CONTENT.value)
        if len(response_lower) > 2000:
            failure_tags.append(FailureTag.OVER_VERBOSE.value)
        if len(response_lower) < 20:
            failure_tags.append(FailureTag.UNDER_VERBOSE.value)
        if clarity_score <= 2:
            failure_tags.append(FailureTag.FORMAT_VIOLATION.value)

        # Check for off-topic
        if overlap < 0.1 and len(q_terms) > 3:
            failure_tags.append(FailureTag.OFF_TOPIC.value)

        # === SUMMARY & FIX ===
        gate_failures = [g for g in gates if not g["passed"]]
        low_scores = [s for s in scores if s["score"] <= 2]

        if gate_failures:
            summary = f"Response fails {', '.join(g['gate_type'] for g in gate_failures)} gate(s)"
            what_to_fix = f"Fix: {'; '.join(g['reason'] for g in gate_failures if g['reason'])}"
        elif low_scores:
            summary = f"Response has low scores in {', '.join(s['score_type'] for s in low_scores)}"
            what_to_fix = f"Improve: {'; '.join(s['justification'] for s in low_scores)}"
        else:
            summary = "Response meets quality standards"
            what_to_fix = "No critical issues; minor improvements possible"

        return {
            "gates": gates,
            "scores": scores,
            "failure_tags": failure_tags,
            "summary_of_issue": summary,
            "what_to_fix": what_to_fix,
            "rag_citations": [],
        }
