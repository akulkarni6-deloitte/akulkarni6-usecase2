"""
Dynamic LLM client selection (Factory + Strategy pattern).

The rest of the system (agents, tools) only ever depends on the
`langchain_core.language_models.BaseChatModel` interface returned here.
Swapping providers is a config change (`IDAMP_LLM_PROVIDER` or an explicit
argument), never a code change in agent logic.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel

from utils.secrets import get_secrets_provider


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"


class LLMClientStrategy(ABC):
    """One concrete strategy per provider."""

    key_name: str
    default_model_env: str
    fallback_model: str

    @abstractmethod
    def build(self, api_key: str, model: str, temperature: float) -> BaseChatModel:
        raise NotImplementedError

    def create(self, temperature: float = 0.0, model: Optional[str] = None) -> BaseChatModel:
        secrets = get_secrets_provider()
        api_key = secrets.require_secret(self.key_name)
        resolved_model = model or os.environ.get(self.default_model_env, self.fallback_model)
        return self.build(api_key=api_key, model=resolved_model, temperature=temperature)


class AnthropicStrategy(LLMClientStrategy):
    key_name = "ANTHROPIC_API_KEY"
    default_model_env = "IDAMP_ANTHROPIC_MODEL"
    fallback_model = "claude-sonnet-4-6"

    def build(self, api_key: str, model: str, temperature: float) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)


class OpenAIStrategy(LLMClientStrategy):
    key_name = "OPENAI_API_KEY"
    default_model_env = "IDAMP_OPENAI_MODEL"
    fallback_model = "gpt-4o-mini"

    def build(self, api_key: str, model: str, temperature: float) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)


class GroqStrategy(LLMClientStrategy):
    key_name = "GROQ_API_KEY"
    default_model_env = "IDAMP_GROQ_MODEL"
    fallback_model = "openai/gpt-oss-120b"

    def build(self, api_key: str, model: str, temperature: float) -> BaseChatModel:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=temperature)


class GeminiStrategy(LLMClientStrategy):
    key_name = "GOOGLE_API_KEY"
    default_model_env = "IDAMP_GEMINI_MODEL"
    fallback_model = "gemini-1.5-pro"

    def build(self, api_key: str, model: str, temperature: float) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)


_STRATEGIES: dict[LLMProvider, LLMClientStrategy] = {
    LLMProvider.ANTHROPIC: AnthropicStrategy(),
    LLMProvider.OPENAI: OpenAIStrategy(),
    LLMProvider.GROQ: GroqStrategy(),
    LLMProvider.GEMINI: GeminiStrategy(),
}


class LLMClientFactory:
    """Public factory used everywhere else in the codebase."""

    @staticmethod
    def get_client(
        provider: Optional[str] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> BaseChatModel:
        provider_name = (provider or os.environ.get("IDAMP_LLM_PROVIDER", "groq")).lower()
        try:
            provider_enum = LLMProvider(provider_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown IDAMP_LLM_PROVIDER '{provider_name}'. "
                f"Valid options: {[p.value for p in LLMProvider]}"
            ) from exc
        strategy = _STRATEGIES[provider_enum]
        return strategy.create(temperature=temperature, model=model)
