"""Mock LLM provider with deterministic rule-based responses."""
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
        """Mock completion - returns templated response."""
        last_message = messages[-1].content if messages else ""

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
        label_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Rule-based classification."""
        text_lower = text.lower()

        # Keyword matching for classification
        label_keywords = {
            "billing_seats": ["seat", "seats", "user limit", "add user", "remove user", "license"],
            "billing_refund": ["refund", "charge", "money back", "cancel subscription", "billing dispute"],
            "workspace_access": ["workspace", "can't access", "no access", "access denied", "join workspace"],
            "permission_sharing": ["permission", "share", "sharing", "edit access", "view only", "can't edit"],
            "login_sso": ["login", "sso", "saml", "oauth", "can't sign in", "password", "authentication"],
            "import_export_sync": ["import", "export", "sync", "migrate", "csv", "backup", "transfer data"],
            "bug_report": ["bug", "error", "crash", "not working", "broken", "glitch", "issue"],
            "feature_request": ["feature", "would be nice", "suggestion", "wish", "could you add", "request"],
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

        # Default to bug_report if no matches
        if best_score == 0:
            best_label = "bug_report"

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
            "current_plan": r"(free|pro|team|enterprise|business)\s*(plan)?",
            "seat_count": r"(\d+)\s*(seats?|users?|licenses?)",
            "billing_cycle": r"(monthly|annual|yearly)",
            "receipt_available": r"(receipt|invoice)",
            "idp_provider": r"(okta|azure|google|onelogin)",
            "error_code": r"(error\s*:?\s*\d+|code\s*:?\s*\w+)",
            "is_admin": r"(admin|administrator|owner)",
            "browser_os": r"(chrome|firefox|safari|windows|mac|linux)",
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
        conversation: str,
        candidate_response: str,
        rubric: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rule-based evaluation."""
        response_lower = candidate_response.lower()
        conv_lower = conversation.lower()

        # === GATES ===
        gates = []

        # Policy Safety Gate
        pii_patterns = [
            r"your (email|phone|address|ssn|password) is",
            r"send (me|us) your (password|credit card|ssn)",
            r"(full name|social security|credit card number)",
        ]
        has_pii_violation = any(re.search(p, response_lower) for p in pii_patterns)

        gates.append({
            "gate_type": GateType.POLICY_SAFETY.value,
            "passed": not has_pii_violation,
            "reason": "PII exposure detected" if has_pii_violation else None,
            "evidence": None,
        })

        # Overclaim Gate
        overclaim_patterns = [
            r"definitely will",
            r"guaranteed to",
            r"100%",
            r"always works",
            r"never fails",
            r"our ai.*will.*automatically",
            r"upcoming feature.*next week",
            r"we will.*soon",
        ]
        has_overclaim = any(re.search(p, response_lower) for p in overclaim_patterns)

        # Check context support if provided
        context_supports = True
        if context and has_overclaim:
            # Simple check: if key terms from response aren't in context
            response_terms = set(re.findall(r'\w{4,}', response_lower))
            context_terms = set(re.findall(r'\w{4,}', context.lower()))
            overlap = len(response_terms & context_terms) / len(response_terms) if response_terms else 0
            context_supports = overlap > 0.3

        gates.append({
            "gate_type": GateType.OVERCLAIM.value,
            "passed": not has_overclaim or context_supports,
            "reason": "Unsubstantiated claim without documentation support" if has_overclaim and not context_supports else None,
            "evidence": None,
        })

        # === SCORES ===
        scores = []

        # Understanding (1-5)
        understanding_score = 3  # Default
        if any(phrase in response_lower for phrase in ["i understand", "i see that", "it sounds like"]):
            understanding_score += 1
        if any(phrase in response_lower for phrase in ["to clarify", "let me make sure"]):
            understanding_score += 1
        if len(response_lower) < 50:  # Too short
            understanding_score -= 1
        if "wrong" in conv_lower and "sorry" not in response_lower:  # Missing acknowledgment
            understanding_score -= 1
        understanding_score = max(1, min(5, understanding_score))

        scores.append({
            "score_type": ScoreType.UNDERSTANDING.value,
            "score": understanding_score,
            "justification": f"Response {'acknowledges' if understanding_score >= 3 else 'fails to acknowledge'} customer's issue",
        })

        # Info Strategy (1-5)
        info_score = 3
        question_marks = response_lower.count("?")
        if question_marks >= 2:
            info_score += 1
        elif question_marks == 0:
            info_score -= 1
        if any(phrase in response_lower for phrase in ["could you", "would you mind", "can you provide"]):
            info_score += 1

        scores.append({
            "score_type": ScoreType.INFO_STRATEGY.value,
            "score": max(1, min(5, info_score)),
            "justification": f"Response asks {question_marks} clarifying questions",
        })

        # Actionability (1-5)
        action_score = 3
        action_phrases = ["step 1", "first,", "next,", "then,", "click on", "go to", "navigate to"]
        action_count = sum(1 for p in action_phrases if p in response_lower)
        if action_count >= 2:
            action_score += 2
        elif action_count == 1:
            action_score += 1
        if "let me know" in response_lower:  # Offers follow-up
            action_score += 1
        if len(response_lower) < 100:  # Too short for good actions
            action_score -= 1

        scores.append({
            "score_type": ScoreType.ACTIONABILITY.value,
            "score": max(1, min(5, action_score)),
            "justification": f"Response provides {action_count} actionable steps",
        })

        # Communication (1-5)
        comm_score = 3
        if any(p in response_lower for p in ["happy to help", "glad to", "thank you for"]):
            comm_score += 1
        if "!" in candidate_response and candidate_response.count("!") <= 2:
            comm_score += 1
        if candidate_response.isupper():  # ALL CAPS is bad
            comm_score -= 2
        if any(p in response_lower for p in ["obviously", "just do", "simply"]):
            comm_score -= 1

        scores.append({
            "score_type": ScoreType.COMMUNICATION.value,
            "score": max(1, min(5, comm_score)),
            "justification": "Professional tone assessment",
        })

        # === FAILURE TAGS ===
        failure_tags = []

        if understanding_score <= 2:
            failure_tags.append(FailureTag.INTENT_MISS.value)
        if info_score <= 2 and "?" not in candidate_response:
            failure_tags.append(FailureTag.MISSING_SLOT.value)
        if action_score <= 2:
            failure_tags.append(FailureTag.NO_NEXT_STEP.value)
        if has_pii_violation:
            failure_tags.append(FailureTag.POLICY_PII.value)
        if has_overclaim and not context_supports:
            failure_tags.append(FailureTag.OVERCLAIM.value)
        if comm_score <= 2:
            failure_tags.append(FailureTag.TONE_ISSUE.value)

        # Check for escalation signals
        if any(p in conv_lower for p in ["legal", "lawyer", "sue", "report", "urgent"]):
            if "escalate" not in response_lower and "manager" not in response_lower:
                failure_tags.append(FailureTag.ESCALATION_NEEDED.value)

        # SSO-specific checks
        if "sso" in conv_lower or "saml" in conv_lower:
            if "admin" not in response_lower and "it team" not in response_lower:
                failure_tags.append(FailureTag.SSO_ADMIN_REQUIRED.value)

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
            what_to_fix = "No critical issues; consider minor tone improvements"

        return {
            "gates": gates,
            "scores": scores,
            "failure_tags": failure_tags,
            "summary_of_issue": summary,
            "what_to_fix": what_to_fix,
            "rag_citations": [],
        }
