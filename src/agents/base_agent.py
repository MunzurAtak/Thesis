from abc import ABC, abstractmethod


class DebateAgent(ABC):
    """Base class for all debate agents."""

    def __init__(self, name: str, stance: str):
        self.name = name
        self.stance = stance

    @abstractmethod
    def generate_response(
        self, topic: str, debate_history: list[dict], round_number: int
    ) -> str:
        """Generate the next debate turn"""
        pass
