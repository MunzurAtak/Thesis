class MockLLM:
    """
    Temporary mock LLM for testing purposes.
    Will be replaced with real models in the future.
    """

    def generate(self, prompt: str, stance: str, topic: str, round_number: int) -> str:
        if stance == "pro":
            