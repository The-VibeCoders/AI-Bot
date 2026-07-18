from .base import BasePersonality


class PersonalityRegistry:
    _personalities: dict[str, BasePersonality] = {}

    @classmethod
    def register(cls, personality: BasePersonality):
        cls._personalities[personality.id] = personality

    @classmethod
    def get(cls, personality_id: str) -> BasePersonality:
        return cls._personalities.get(personality_id, cls._personalities.get("standard"))

    @classmethod
    def list(cls) -> list[BasePersonality]:
        return list(cls._personalities.values())

    @classmethod
    def discover(cls):
        from .standard import StandardChatbotPersonality
        from .coding_agent import CodingAgentPersonality
        cls.register(StandardChatbotPersonality())
        cls.register(CodingAgentPersonality())
