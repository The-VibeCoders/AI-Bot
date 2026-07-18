from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from .base import LLMProvider
from typing import Generator


class GeminiProvider(LLMProvider):
    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        client = genai.Client(api_key=self.api_key)
        system = None
        clean = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                clean.append(m)

        config = None
        if system:
            config = GenerateContentConfig(system_instruction=system)

        if len(clean) <= 1:
            content = clean[0]["content"] if clean else ""
            resp = client.models.generate_content_stream(
                model=self.model_id, contents=content, config=config
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        else:
            history = []
            for m in clean[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append(Content(role=role, parts=[Part(text=m["content"])]))
            last = clean[-1]["content"]
            chat = client.chats.create(model=self.model_id, history=history, config=config)
            for chunk in chat.send_message_stream(last):
                if chunk.text:
                    yield chunk.text

    def is_available(self) -> bool:
        return bool(self.api_key)
