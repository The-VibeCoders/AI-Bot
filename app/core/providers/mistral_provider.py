from mistralai.client import Mistral
from .base import LLMProvider
from typing import Generator


class MistralProvider(LLMProvider):
    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        client = Mistral(api_key=self.api_key)
        chat_messages = []
        for m in messages:
            chat_messages.append({"role": m["role"], "content": m["content"]})

        response_stream = client.chat.stream(
            model=self.model_id,
            messages=chat_messages,
            temperature=0.7,
            top_p=0.9
        )

        for chunk in response_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def is_available(self) -> bool:
        return bool(self.api_key)