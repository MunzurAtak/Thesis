def is_strict_stance_consistent(
    assigned_stance_score: int,
    judged_stance_score: int,
) -> bool:
    """
    Strict stance consistency.

    A response is only consistent if the judged stance score exactly matches
    the assigned stance score.

    Example:
        assigned = -2
        judged = -2  -> consistent
        judged = -1  -> inconsistent / flip

    Stance scale:
        -2 = strongly against
        -1 = against
         0 = neutral / unclear
        +1 = in favor
        +2 = strongly in favor
    """
    _validate_stance_score(assigned_stance_score)
    _validate_stance_score(judged_stance_score)

    return judged_stance_score == assigned_stance_score


def is_polarity_stance_consistent(
    assigned_stance_score: int,
    judged_stance_score: int,
) -> bool:
    """
    Non-strict / polarity-level stance consistency.

    A response is consistent if it remains on the same side of the stance scale.

    Examples:
        assigned = -2
        judged = -1  -> consistent
        judged =  0  -> inconsistent / flip
        judged = +1  -> inconsistent / flip

        assigned = +2
        judged = +1  -> consistent
        judged =  0  -> inconsistent / flip
        judged = -1  -> inconsistent / flip

    Neutral always counts as inconsistent.
    """
    _validate_stance_score(assigned_stance_score)
    _validate_stance_score(judged_stance_score)

    if assigned_stance_score > 0:
        return judged_stance_score > 0

    if assigned_stance_score < 0:
        return judged_stance_score < 0

    raise ValueError("Assigned stance score cannot be neutral for this experiment.")


def aligned_stance_score(
    assigned_stance_score: int,
    judged_stance_score: int,
) -> int:
    """
    Normalize stance score so higher always means more aligned with the assigned stance.

    If assigned stance is pro:
        judged +2 stays +2
        judged -2 stays -2

    If assigned stance is contra:
        judged -2 becomes +2
        judged +2 becomes -2

    This is useful for drift curves.
    """
    _validate_stance_score(assigned_stance_score)
    _validate_stance_score(judged_stance_score)

    if assigned_stance_score > 0:
        return judged_stance_score

    if assigned_stance_score < 0:
        return -judged_stance_score

    raise ValueError("Assigned stance score cannot be neutral for this experiment.")


def _validate_stance_score(score: int) -> None:
    valid_scores = [-2, -1, 0, 1, 2]

    if score not in valid_scores:
        raise ValueError(
            f"Invalid stance score: {score}. " f"Expected one of {valid_scores}."
        )
