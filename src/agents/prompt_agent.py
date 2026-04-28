from src.agents.base_agent import DebateAgent


class PromptAgent(DebateAgent):
    """
    Prompting method debate agent.

    Stance of this agent is controlled only through prompting
    """

    def __init__(self, name: str, stance: str, llm):
        super().__init__(name=name, stance=stance)
        self.llm = llm

    def build_prompt(
        self, topic: str, debate_history: list[dict], round_number: int
    ) -> str:
        history_text = self._format_history(debate_history)

        return f"""
You are a debate agent.

Topic:
{topic}

Assigned stance:
{self.stance}

Instructions:
- Maintain your assigned stance throughout the debate.
- Do not switch sides.
- Respond to the opponent's previous argument.
- Keep your answer concise and argumentative.

Debate history:
{history_text}

Round:
{round_number}

Write your next debate turn.
""".strip()

    def generate_response(
        self, topic: str, debate_history: list[dict], round_number: int
    ) -> str:

        prompt = self.build_prompt(
            topic=topic, debate_history=debate_history, round_number=round_number
        )

        return self.llm.generate(
            prompt=prompt, stance=self.stance, topic=topic, round_number=round_number
        )

    @staticmethod
    def _format_history(debate_history: list[dict]) -> str:
        if not debate_history:
            return "No previous turns."

        lines = []
        for turn in debate_history:
            lines.append(
                f"round {turn['round']} | {turn['speaker']} ({turn['stance']}): "
                f"{turn['utterance']}"
            )

        return "\n".join(lines)
