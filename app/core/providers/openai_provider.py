from openai import OpenAI
from .base import LLMProvider
from typing import Generator


class OpenAIProvider(LLMProvider):
    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        client = OpenAI(api_key=self.api_key)
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
