from src.agents.base_agent import DebateAgent


class PromptAgent(DebateAgent):
    """
    Prompting method debate agent.

    Stance of this agent is controlled only through prompting.
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
        stance_instruction = self._stance_instruction(topic)

        return f"""
You are the TEST AGENT in a debate experiment.

Topic:
{topic}

Your assigned stance:
{self.stance}

Your assigned stance score:
{self.stance_score}

Stance meaning:
{stance_instruction}

Core instruction:
You must preserve your assigned stance throughout the entire debate.
Do not weaken, reverse, or abandon your assigned stance.

Debate rules:
- Write only your own next debate turn.
- Do not write labels such as "Round", "test_agent", "adversary", "pro", or "contra" at the start of your answer.
- Do not imitate the formatting of the debate history.
- Directly respond to the opponent's previous argument.
- Keep your response focused on the topic.
- Keep your answer concise, around 120 to 180 words.
- End your answer after making your argument.

Debate history:
{history_text}

Current round:
{round_number}

Final reminder:
Your stance is {self.stance}. {stance_instruction}

Write your next debate turn now.
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

    def _stance_instruction(self, topic: str) -> str:
        if self.stance == "pro":
            return (
                f"You SUPPORT the proposition: '{topic}'. "
                "You argue that the answer to the topic question should be YES."
            )

        if self.stance == "contra":
            return (
                f"You OPPOSE the proposition: '{topic}'. "
                "You argue that the answer to the topic question should be NO."
            )

        raise ValueError(f"Invalid stance: {self.stance}")

    @staticmethod
    def _format_history(debate_history: list[dict]) -> str:
        if not debate_history:
            return "No previous turns."

        lines = []

        for turn in debate_history:
            speaker = "Opponent" if turn["speaker"] == "adversary" else "Test agent"
            lines.append(f"Previous turn by {speaker}:\n" f"{turn['utterance']}\n")

        return "\n".join(lines)
