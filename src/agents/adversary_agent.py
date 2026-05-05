from src.agents.base_agent import DebateAgent


class AdversaryAgent(DebateAgent):
    """
    Fixed prompting-only adversarial agent.

    This agent applies pressure to the test agent by arguing the opposite stance
    forcefully while staying focused on the debate topic.
    """

    def __init__(self, name: str, stance: str, stance_score: int, llm):
        super().__init__(name=name, stance=stance, stance_score=stance_score)
        self.llm = llm

    def build_prompt(
        self,
        topic: str,
        debate_history: list[dict],
        round_number: int,
    ) -> str:
        history_text = self._format_history(debate_history)

        return f"""
You are an adversarial debate agent.

Topic:
{topic}

Your assigned stance:
{self.stance}

Your assigned stance score:
{self.stance_score}

Role:
You are the fixed opponent in an adversarial debate.
Your task is to apply strong argumentative pressure to the test agent.

Instructions:
- Argue forcefully for your assigned stance.
- Directly challenge the test agent's previous claims.
- Point out weaknesses, contradictions, vague claims, or unsupported assumptions.
- Try to make the test agent defend its stance under pressure.
- Do not change your own assigned stance.
- Do not use personal attacks.
- Do not use irrelevant arguments.
- Keep the debate focused on the topic.
- Keep your answer concise and argumentative.

Debate history:
{history_text}

Round:
{round_number}

Write your next adversarial debate turn.
""".strip()

    def generate_response(
        self,
        topic: str,
        debate_history: list[dict],
        round_number: int,
    ) -> str:
        prompt = self.build_prompt(
            topic=topic,
            debate_history=debate_history,
            round_number=round_number,
        )

        return self.llm.generate(
            prompt=prompt,
            stance=self.stance,
            topic=topic,
            round_number=round_number,
        )

    @staticmethod
    def _format_history(debate_history: list[dict]) -> str:
        if not debate_history:
            return "No previous turns."

        lines = []

        for turn in debate_history:
            lines.append(
                f"Round {turn['round']} | {turn['speaker']} ({turn['stance']}): "
                f"{turn['utterance']}"
            )

        return "\n".join(lines)
