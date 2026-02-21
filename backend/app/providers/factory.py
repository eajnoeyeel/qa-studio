"""Factory for creating LLM providers."""
from typing import Optional
from .base import LLMProvider
from .mock import MockProvider


def get_provider(
    provider_name: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    instrumentation=None,
) -> LLMProvider:
    """Get an LLM provider instance."""

    if provider_name == "mock":
        return MockProvider()

    elif provider_name == "openai":
        if not api_key:
            raise ValueError("OpenAI API key required")
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key,
            default_model=model or "gpt-4o-mini",
            instrumentation=instrumentation,
        )

    elif provider_name == "anthropic":
        if not api_key:
            raise ValueError("Anthropic API key required")
        # Could implement AnthropicProvider similarly
        raise NotImplementedError("Anthropic provider not yet implemented")

    else:
        raise ValueError(f"Unknown provider: {provider_name}")
