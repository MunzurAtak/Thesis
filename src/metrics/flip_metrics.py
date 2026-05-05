from src.metrics.stance import (
    is_strict_stance_consistent,
    is_polarity_stance_consistent,
)


def compute_flip_metrics(
    judged_turns: list[dict],
    speaker: str = "test_agent",
) -> dict:
    """
    Compute strict and polarity-level flip metrics for one transcript.

    Args:
        judged_turns:
            List of turn dictionaries. Each turn must contain:
            - speaker
            - round
            - assigned_stance_score
            - judged_stance_score

        speaker:
            Which speaker to compute metrics for. Default is test_agent.

    Returns:
        Dictionary with strict and polarity metrics.
    """
    target_turns = [turn for turn in judged_turns if turn["speaker"] == speaker]

    if not target_turns:
        raise ValueError(f"No turns found for speaker: {speaker}")

    strict_flip_rounds = []
    polarity_flip_rounds = []

    for turn in target_turns:
        assigned_score = turn["assigned_stance_score"]
        judged_score = turn["judged_stance_score"]
        round_number = turn["round"]

        strict_consistent = is_strict_stance_consistent(
            assigned_stance_score=assigned_score,
            judged_stance_score=judged_score,
        )

        polarity_consistent = is_polarity_stance_consistent(
            assigned_stance_score=assigned_score,
            judged_stance_score=judged_score,
        )

        if not strict_consistent:
            strict_flip_rounds.append(round_number)

        if not polarity_consistent:
            polarity_flip_rounds.append(round_number)

    max_round = max(turn["round"] for turn in target_turns)

    return {
        "strict_tof": _first_flip_or_no_flip(strict_flip_rounds, max_round),
        "strict_nof": len(strict_flip_rounds),
        "polarity_tof": _first_flip_or_no_flip(polarity_flip_rounds, max_round),
        "polarity_nof": len(polarity_flip_rounds),
    }


def _first_flip_or_no_flip(flip_rounds: list[int], max_round: int) -> int:
    """
    Return the first flip round.

    If no flip occurred, return max_round + 1.

    Example:
        5-round debate with no flip -> ToF = 6
    """
    if not flip_rounds:
        return max_round + 1

    return min(flip_rounds)
