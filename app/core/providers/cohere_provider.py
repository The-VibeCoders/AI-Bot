import cohere
from .base import LLMProvider
from typing import Generator


class CohereProvider(LLMProvider):
    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        client = cohere.ClientV2(api_key=self.api_key)
        system_messages = [m for m in messages if m["role"] == "system"]
        chat_messages = [m for m in messages if m["role"] != "system"]

        system = None
        if system_messages:
            system = "\n".join([m["content"] for m in system_messages])

        response_stream = client.chat(
            model=self.model_id,
            messages=chat_messages,
            system=system,
            temperature=0.7,
            max_tokens=4096,
            stream=True
        )

        for chunk in response_stream:
            if chunk.type == "text-generation" and chunk.text:
                yield chunk.text

    def is_available(self) -> bool:
        return bool(self.api_key)