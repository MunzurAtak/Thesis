from abc import ABC, abstractmethod


class DebateAgent(ABC):
    """Base class for all debate agents."""

    def __init__(self, name: str, stance: str, stance_score: int):
        self.name = name
        self.stance = stance
        self.stance_score = stance_score

    @abstractmethod
    def generate_response(
        self, topic: str, debate_history: list[dict], round_number: int
    ) -> str:
        """Generate the next debate turn"""
        pass
