from typing import Optional
from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .openai_compatible_provider import OpenAICompatibleProvider
from .gemini_provider import GeminiProvider
from .mistral_provider import MistralProvider
from .cohere_provider import CohereProvider
from . import key_store

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "cohere": CohereProvider,
}


class ProviderManager:
    def _parse_id(self, model_id: str):
        if ":" in model_id:
            parts = model_id.split(":", 1)
            return parts[0], parts[1]
        return None, model_id

    def get_provider(self, model_id: str, api_key: str = None) -> LLMProvider:
        prefix, raw_id = self._parse_id(model_id)
        provider_name = prefix or key_store.get_provider_for_model(raw_id)
        if provider_name:
            ptype = key_store.get_provider_type(provider_name)
            cls = PROVIDER_CLASSES.get(ptype or provider_name)
            if cls:
                stored_key = key_store.get_key(provider_name)
                base_url = key_store.get_base_url(provider_name)
                return cls(raw_id, api_key=stored_key or api_key, base_url=base_url)
        return OllamaProvider(raw_id, api_key)

    def get_provider_for_model(self, model_id: str) -> Optional[str]:
        prefix, raw_id = self._parse_id(model_id)
        return prefix or key_store.get_provider_for_model(raw_id)

    def list_cloud_models(self) -> list[dict]:
        return key_store.list_cloud_models()

    def list_providers(self) -> list[dict]:
        return key_store.list_providers()

    def save_provider(self, provider: str, provider_type: str, api_key: str, base_url: str = None):
        key_store.save_provider(provider, provider_type, api_key, base_url)

    def remove_provider(self, provider: str) -> bool:
        return key_store.remove_provider(provider)

    def save_key(self, provider: str, model_id: str, api_key: str = None):
        key_store.save_key(provider, model_id, api_key)

    def remove_model(self, model_id: str) -> bool:
        prefix, raw_id = self._parse_id(model_id)
        return key_store.remove_model(raw_id)

    def detect_models(self, provider_type: str, api_key: str, base_url: str = None) -> list[str]:
        return key_store.detect_models(provider_type, api_key, base_url)
