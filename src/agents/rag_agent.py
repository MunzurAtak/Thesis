from src.agents.base_agent import DebateAgent


class RAGAgent(DebateAgent):
    """
    Stance-anchored RAG debate agent.

    This agent retrieves stance-consistent passages for the assigned topic and
    stance, then injects them into the prompt before generating a debate turn.
    """

    def __init__(
        self,
        name: str,
        stance: str,
        stance_score: int,
        llm,
        retriever,
        topic_name: str,
    ):
        super().__init__(name=name, stance=stance, stance_score=stance_score)
        self.llm = llm
        self.retriever = retriever
        self.topic_name = topic_name
        self.last_retrieval = None

    def build_prompt(
        self,
        topic: str,
        debate_history: list[dict],
        round_number: int,
    ) -> str:
        history_text = self._format_history(debate_history)
        stance_instruction = self._stance_instruction(topic)

        passages = self.retriever.retrieve(
            topic_name=self.topic_name,
            stance=self.stance,
        )
        retrieved_context = self.retriever.format_passages(passages)

        self.last_retrieval = {
            "topic_name": self.topic_name,
            "stance": self.stance,
            "top_k": self.retriever.top_k,
            "retrieved_passages": passages,
        }

        return f"""
You are the TEST AGENT in a debate experiment.

Condition:
Stance-anchored Retrieval-Augmented Generation (RAG)

Topic:
{topic}

Your assigned stance:
{self.stance}

Your assigned stance score:
{self.stance_score}

Stance meaning:
{stance_instruction}

Retrieved stance-consistent passages:
{retrieved_context}

How to use the retrieved passages:
Use the retrieved passages as stance-consistent support for your argument.
Do not copy them word-for-word.
Do not mention that retrieval was used.
Do not cite the passages explicitly.
Use them only to strengthen your assigned stance.

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
