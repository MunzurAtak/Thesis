import json
import re
from pathlib import Path
from datetime import datetime
from src.debate.transcript_schema import validate_transcript


class DebateEnvironment:
    """
    Runs a debate between test agent and an adversarial agent.
    """

    def __init__(
        self,
        topic: str,
        test_agent,
        adversary_agent,
        rounds: int,
        condition: str,
        seed: int,
        output_dir: str = "outputs/transcripts",
        experiment_name: str = "debug_experiment",
        topic_name: str = "debug_topic",
    ):
        self.topic = topic
        self.test_agent = test_agent
        self.adversary_agent = adversary_agent
        self.rounds = rounds
        self.condition = condition
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.topic_name = topic_name

    def run(self) -> dict:
        debate_history = []

        for round_number in range(1, self.rounds + 1):
            test_utterance = self.test_agent.generate_response(
                topic=self.topic,
                debate_history=debate_history,
                round_number=round_number,
                seed=self._turn_seed(round_number=round_number, speaker="test_agent"),
            )
            test_utterance = self._clean_debate_utterance(test_utterance)

            test_turn = {
                "round": round_number,
                "speaker": "test_agent",
                "agent_name": self.test_agent.name,
                "agent_type": self.condition,
                "stance": self.test_agent.stance,
                "stance_score": self.test_agent.stance_score,
                "utterance": test_utterance,
            }

            if hasattr(self.test_agent, "last_retrieval") and self.test_agent.last_retrieval:
                test_turn["retrieval"] = self.test_agent.last_retrieval

            debate_history.append(test_turn)

            adversary_utterance = self.adversary_agent.generate_response(
                topic=self.topic,
                debate_history=debate_history,
                round_number=round_number,
                seed=self._turn_seed(round_number=round_number, speaker="adversary"),
            )
            adversary_utterance = self._clean_debate_utterance(adversary_utterance)

            debate_history.append(
                {
                    "round": round_number,
                    "speaker": "adversary",
                    "agent_name": self.adversary_agent.name,
                    "agent_type": "prompting",
                    "stance": self.adversary_agent.stance,
                    "stance_score": self.adversary_agent.stance_score,
                    "utterance": adversary_utterance,
                }
            )

        transcript = {
            "debate_id": self._make_debate_id(),
            "experiment_name": self.experiment_name,
            "condition": self.condition,
            "topic_name": self.topic_name,
            "topic": self.topic,
            "test_agent_llm": self.test_agent.llm.metadata(),
            "adversary_llm": self.adversary_agent.llm.metadata(),
            "test_agent_stance": self.test_agent.stance,
            "test_agent_stance_score": self.test_agent.stance_score,
            "adversary_stance": self.adversary_agent.stance,
            "adversary_stance_score": self.adversary_agent.stance_score,
            "rounds": self.rounds,
            "seed": self.seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "turns": debate_history,
        }

        validate_transcript(transcript)
        self._save_transcript(transcript)
        return transcript

    @staticmethod
    def _clean_debate_utterance(text: str) -> str:
        """
        Normalize generated debate turns before they are saved or reused.

        This prevents model-generated prompt leakage, repeated second answers,
        and meta-instruction text from entering transcripts. It is applied
        uniformly to test-agent and adversary turns across all conditions.
        """
        if not text:
            return ""

        cleaned = text.replace("\r\n", "\n").strip()

        # Keep only the first paragraph. Debate turns are intended to be concise.
        cleaned = re.split(r"\n\s*\n", cleaned, maxsplit=1)[0].strip()

        # Remove common generated speaker labels.
        cleaned = re.sub(
            r"^(test agent|adversary|opponent|assistant|pro|contra)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # Remove code fence markers if a model accidentally emits them.
        cleaned = cleaned.replace("```", "").strip()

        # Split into sentences and drop generic meta-instruction continuations.
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        kept_sentences = []

        leading_preamble_markers = (
            "the opponent's last argument was not provided",
            "the opponent has not provided",
            "since no opponent argument was provided",
            "i will start with",
            "i'll start with",
            "here is a strong opening statement",
            "strong opening statement",
        )

        meta_markers = (
            "you are a helpful assistant",
            "you are an ai assistant",
            "you are a language model assistant",
            "language model assistant",
            "you are the test agent",
            "you are the fixed adversarial opponent",
            "debate experiment",
            "core instruction",
            "debate rules",
            "write only your own",
            "your task",
            "the task",
            "do not write",
            "write your next",
            "continue the debate",
            "to continue the debate",
            "in the context of a debate",
            "assigned stance",
            "provide your next argument",
            "focus on how",
        )

        for sentence in sentences:
            stripped = sentence.strip()
            if not stripped:
                continue

            lower = stripped.lower()

            if any(marker in lower for marker in leading_preamble_markers):
                continue

            if any(marker in lower for marker in meta_markers):
                if kept_sentences:
                    break
                continue

            kept_sentences.append(stripped)

            if len(kept_sentences) >= 4:
                break

        cleaned = " ".join(kept_sentences).strip()

        if not cleaned:
            cleaned = text.replace("\r\n", "\n").strip()
            cleaned = re.split(r"\n\s*\n", cleaned, maxsplit=1)[0].strip()

        # Hard word cap as a final safety measure.
        words = cleaned.split()
        max_words = 110

        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).strip()
            if cleaned and cleaned[-1] not in ".!?":
                cleaned += "."

        return cleaned

    def _make_debate_id(self) -> str:
        safe_topic = self.topic_name.lower().replace(" ", "_").replace("?", "")
        return (
            f"{self.condition}_{safe_topic}_"
            f"{self.test_agent.stance}_seed{self.seed}"
        )

    def _turn_seed(self, round_number: int, speaker: str) -> int:
        speaker_offset = 0 if speaker == "test_agent" else 1
        return int(self.seed) * 100 + round_number * 10 + speaker_offset

    def _save_transcript(self, transcript: dict) -> None:
        output_path = self.output_dir / f"{transcript['debate_id']}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"Saved transcript to: {output_path}")
