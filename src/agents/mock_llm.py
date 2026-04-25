class MockLLM:
    """
    Temporary mock LLM for testing purposes.
    Will be replaced with real models in the future.
    """

    def generate(self, prompt: str, stance: str, topic: str, round_number: int) -> str:
        if stance == "pro":
            return (
                f"In round {round_number}, I argue in favor of '{topic}'. "
                f"My position remains that this policy is justified because the benefits"
                f"are stronger than the objections raised by the opposition."
            )

        if stance == "contra":
            return (
                f"In round {round_number}, I argue against '{topic}'. "
                f"My position remains that this policy is unjustified because the objections"
                f"are stronger than the benefits raised by the proponents."
            )

        return f"In round {round_number}, I discuss the topic '{topic}'."
