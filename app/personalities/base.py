from abc import ABC, abstractmethod
from typing import Optional


class BasePersonality(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this personality."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name shown in UI."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description shown in personality selector."""

    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji icon for the personality."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Base system prompt template. Use {context} as placeholder."""

    @property
    def tools(self) -> list[dict]:
        """OpenAI-compatible tool definitions. Empty by default."""
        return []

    @property
    def requires_approval(self) -> list[str]:
        """Tool names that need human approval before execution."""
        return []

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return result as a string for the LLM.
        Raises ValueError for unknown tools."""
        raise ValueError(f"Unknown tool: {tool_name}")

    def get_sidebar_ui(self, user_state: dict) -> str:
        """Optional HTML injected into the sidebar when this personality is active.
        Override to add personality-specific controls."""
        return ""
