import json
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
            )

            debate_history.append(
                {
                    "round": round_number,
                    "speaker": "test_agent",
                    "agent_name": self.test_agent.name,
                    "agent_type": self.condition,
                    "stance": self.test_agent.stance,
                    "utterance": test_utterance,
                }
            )

            adversary_utterance = self.adversary_agent.generate_response(
                topic=self.topic,
                debate_history=debate_history,
                round_number=round_number,
            )

            debate_history.append(
                {
                    "round": round_number,
                    "speaker": "adversary",
                    "agent_name": self.adversary_agent.name,
                    "agent_type": "prompting",
                    "stance": self.adversary_agent.stance,
                    "utterance": adversary_utterance,
                }
            )

        transcript = {
            "debate_id": self._make_debate_id(),
            "experiment_name": self.experiment_name,
            "condition": self.condition,
            "topic_name": self.topic_name,
            "topic": self.topic,
            "test_agent_stance": self.test_agent.stance,
            "adversary_stance": self.adversary_agent.stance,
            "rounds": self.rounds,
            "seed": self.seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "turns": debate_history,
        }

        validate_transcript(transcript)
        self._save_transcript(transcript)
        return transcript

    def _make_debate_id(self) -> str:
        safe_topic = self.topic_name.lower().replace(" ", "_").replace("?", "")
        return (
            f"{self.condition}_{safe_topic}_"
            f"{self.test_agent.stance}_seed{self.seed}"
        )

    def _save_transcript(self, transcript: dict) -> None:
        output_path = self.output_dir / f"{transcript['debate_id']}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"Saved transcript to: {output_path}")
