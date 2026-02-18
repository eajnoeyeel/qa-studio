"""Base LLM provider interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class LLMMessage(BaseModel):
    """Message for LLM."""
    role: str
    content: str


class LLMResponse(BaseModel):
    """Response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw: Optional[Any] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Complete a chat conversation."""
        pass

    @abstractmethod
    async def classify(
        self,
        text: str,
        labels: List[str],
        label_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Classify text into one of the labels."""
        pass

    @abstractmethod
    async def evaluate(
        self,
        conversation: str,
        candidate_response: str,
        rubric: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a candidate response against a rubric."""
        pass
