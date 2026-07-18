import ollama
from .base import LLMProvider
from typing import Generator, Optional, Dict, Any

class OllamaProvider(LLMProvider):
    """Implementation of the Ollama provider for local models."""

    def __init__(self, model_id: str, api_key: str = None, base_url: str = None, options: Optional[Dict[str, Any]] = None):
        super().__init__(model_id, api_key, base_url)
        self.options = options or {}

    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        response = ollama.chat(model=self.model_id, messages=messages, stream=stream, options=self.options)

        if stream:
            for chunk in response:
                token = chunk.message.content if hasattr(chunk, "message") else chunk.get("message", {}).get("content", "")
                yield token
        else:
            yield response.message.content if hasattr(response, "message") else response.get("message", {}).get("content", "")

    def is_available(self) -> bool:
        try:
            models = ollama.list()
            model_list = [m.model if hasattr(m, "model") else m["model"] for m in models]
            return self.model_id in model_list
        except Exception:
            return False
