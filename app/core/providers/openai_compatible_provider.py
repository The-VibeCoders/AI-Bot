from openai import OpenAI
from .base import LLMProvider
from typing import Generator


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model_id: str, api_key: str = None, base_url: str = None):
        super().__init__(model_id, api_key)
        self.base_url = base_url

    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=stream,
        )
        if stream:
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                yield token
        else:
            yield response.choices[0].message.content or ""

    def is_available(self) -> bool:
        return bool(self.api_key)
