from .base import BasePersonality


class StandardChatbotPersonality(BasePersonality):
    @property
    def id(self) -> str:
        return "standard"

    @property
    def name(self) -> str:
        return "Standard"

    @property
    def description(self) -> str:
        return "General-purpose AI assistant with web search, memory, and PDF analysis"

    @property
    def icon(self) -> str:
        return "🤖"

    @property
    def system_prompt(self) -> str:
        return (
            "You are Doremon, an autonomous local AI assistant running on the user's machine. "
            "You have access to the following capabilities: web search, long-term vector memory, "
            "PDF document ingestion and analysis, and image generation.\n\n"
            "=== LIVE CONTEXT ===\n{context}\n==================="
        )
