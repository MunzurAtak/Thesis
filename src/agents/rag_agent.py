from src.agents.base_agent import DebateAgent


class RAGAgent(DebateAgent):
    """
    Stance-anchored RAG debate agent.

    This agent retrieves stance-consistent passages for the assigned topic and
    stance, then injects them into the prompt before generating a debate turn.

    To avoid evaluation leakage, retrieved passages that contain the exact
    evaluation/debate question are removed before being shown to the agent.
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

        retrieval_query = self._build_retrieval_query(
            topic=topic,
            debate_history=debate_history,
        )

        raw_passages = self.retriever.retrieve(
            topic_name=self.topic_name,
            stance=self.stance,
            query=retrieval_query,
        )

        passages, removed_leakage_count = self._remove_exact_question_leakage(
            passages=raw_passages,
            eval_question=topic,
        )

        retrieved_context = self.retriever.format_passages(passages)

        self.last_retrieval = {
            "topic_name": self.topic_name,
            "stance": self.stance,
            "top_k": self.retriever.top_k,
            "query": retrieval_query,
            "eval_question": topic,
            "removed_exact_eval_question_leakage_count": removed_leakage_count,
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

Private stance-consistent background information:
{retrieved_context}

How to use the private background information:
Use the background information only as private support for your argument.
Do not copy it word-for-word.
Do not mention that retrieval, background information, context, sources, or passages were used.
Do not refer to numbered passages, bullet points, sources, or background notes.
Do not say phrases like "Passage 1", "the retrieved context", "the background information", or "the source says".
Integrate useful ideas naturally into your own argument.

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
        seed: int | None = None,
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
            seed=seed,
        )

    @staticmethod
    def _build_retrieval_query(topic: str, debate_history: list[dict]) -> str:
        if not debate_history:
            return topic

        latest_turn = debate_history[-1]
        latest_utterance = latest_turn.get("utterance", "")

        return f"{topic}\n{latest_utterance}"

    @staticmethod
    def _normalize_for_leakage_check(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _remove_exact_question_leakage(
        self,
        passages: list[dict],
        eval_question: str,
    ) -> tuple[list[dict], int]:
        normalized_eval_question = self._normalize_for_leakage_check(eval_question)

        filtered_passages = []
        removed_count = 0

        for passage in passages:
            passage_text = passage.get("text", "")
            normalized_passage_text = self._normalize_for_leakage_check(passage_text)

            if normalized_eval_question and normalized_eval_question in normalized_passage_text:
                removed_count += 1
                continue

            filtered_passages.append(passage)

        return filtered_passages, removed_count

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
