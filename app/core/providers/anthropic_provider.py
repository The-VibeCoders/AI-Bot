from anthropic import Anthropic
from .base import LLMProvider
from typing import Generator


class AnthropicProvider(LLMProvider):
    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        client = Anthropic(api_key=self.api_key)
        system = None
        clean = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                clean.append({"role": m["role"], "content": m["content"]})
        response = client.messages.create(
            model=self.model_id,
            messages=clean,
            system=system,
            max_tokens=4096,
            stream=stream,
        )
        if stream:
            for event in response:
                if event.type == "content_block_delta" and event.delta.text:
                    yield event.delta.text
        else:
            yield response.content[0].text if response.content else ""

    def is_available(self) -> bool:
        return bool(self.api_key)
