REQUIRED_TRANSCRIPT_FIELDS = [
    "debate_id",
    "experiment_name",
    "condition",
    "topic_name",
    "topic",
    "test_agent_stance",
    "test_agent_stance_score",
    "adversary_stance",
    "adversary_stance_score",
    "rounds",
    "seed",
    "created_at",
    "turns",
    "test_agent_llm",
    "adversary_llm",
]

REQUIRED_TURN_FIELDS = [
    "round",
    "speaker",
    "agent_name",
    "agent_type",
    "stance",
    "stance_score",
    "utterance",
]


def validate_transcript(transcript: dict) -> None:
    """
    Validate that a debate transcript has the expected structure.

    Raises:
        ValueError: If the transcript is missing required fields or has an invalid structure.
    """
    for field in REQUIRED_TRANSCRIPT_FIELDS:
        if field not in transcript:
            raise ValueError(f"Transcript is missing required field: {field}")

    if not isinstance(transcript["turns"], list):
        raise ValueError("Transcript 'turns' field must be a list")

    if len(transcript["turns"]) == 0:
        raise ValueError("Transcript 'turns' list cannot be empty")

    for index, turn in enumerate(transcript["turns"]):
        for field in REQUIRED_TURN_FIELDS:
            if field not in turn:
                raise ValueError(f"Turn {index} is missing required field: {field}")

        if turn["speaker"] not in ["test_agent", "adversary"]:
            raise ValueError(
                f"Invalid speaker at turn index {index}: {turn['speaker']}"
            )

        if turn["stance"] not in ["pro", "contra"]:
            raise ValueError(f"Invalid stance at turn index {index}: {turn['stance']}")

        if turn["stance_score"] not in [-2, -1, 0, 1, 2]:
            raise ValueError(
                f"Invalid stance_score at turn index {index}: {turn['stance_score']}"
            )

        if not isinstance(turn["utterance"], str) or not turn["utterance"].strip():
            raise ValueError(f"Empty utterance at turn index {index}")
