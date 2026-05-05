from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base interface for all language model wrappers.

    Every LLM wrapper must implement generate()
    """

    @abstractmethod
    def generate(self, prompt: str, stance: str, topic: str, round_number: int) -> str:
        """
        Generate a response from a prompt.
        """

        pass
