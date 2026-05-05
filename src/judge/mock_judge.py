class MockJudge:
    """
    Temporary mock judge.

    This judge does not use an LLM yet.
    It simply returns the assigned stance score as the judged stance score.

    Purpose:
    - test judge output format
    - test metric computation pipeline
    - avoid debugging real LLM judge too early
    """

    def judge_turn(self, transcript: dict, turn: dict) -> dict:
        judged_stance_score = turn["stance_score"]

        return {
            "debate_id": transcript["debate_id"],
            "experiment_name": transcript["experiment_name"],
            "condition": transcript["condition"],
            "topic_name": transcript["topic_name"],
            "topic": transcript["topic"],
            "round": turn["round"],
            "speaker": turn["speaker"],
            "agent_name": turn["agent_name"],
            "agent_type": turn["agent_type"],
            "assigned_stance": turn["stance"],
            "assigned_stance_score": turn["stance_score"],
            "utterance": turn["utterance"],
            "judged_stance_score": judged_stance_score,
            "judge_label": self._score_to_label(judged_stance_score),
            "judge_confidence": 1.0,
            "judge_reason": "Mock judge copies the assigned stance score.",
        }

    @staticmethod
    def _score_to_label(score: int) -> str:
        labels = {
            -2: "strongly_against",
            -1: "against",
            0: "neutral_or_unclear",
            1: "in_favor",
            2: "strongly_in_favor",
        }

        if score not in labels:
            raise ValueError(f"Invalid stance score: {score}")

        return labels[score]
