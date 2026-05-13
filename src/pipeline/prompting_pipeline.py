from pathlib import Path

from src.agents.prompt_agent import PromptAgent
from src.debate.environment import DebateEnvironment
from src.debate.transcript_schema import validate_transcript
from src.judge.judge_factory import create_judge, judge_metadata
from src.llms.llm_factory import create_llm
from src.utils import config
from src.utils.config import load_json_config
from src.metrics.flip_metrics import compute_flip_metrics
from src.agents.adversary_agent import AdversaryAgent

import csv
import json


def clear_json_files(directory: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    for file_path in path.glob("*.json"):
        file_path.unlink()


def clear_csv_files(directory: str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    for file_path in path.glob("*.csv"):
        file_path.unlink()


def run_prompting_experiments(config_path: str) -> None:
    config = load_json_config(config_path)

    test_llm = create_llm(config["models"]["test_agent"])
    adversary_llm = create_llm(config["models"]["adversary_agent"])

    condition = config["condition"]
    rounds = config["rounds"]
    output_dir = config["output_dir"]

    for topic_config in config["topics"]:
        topic_name = topic_config["name"]
        topic_question = topic_config["question"]

        for stance_pair in config["stance_pairs"]:
            test_stance = stance_pair["test_agent_stance"]
            test_stance_score = stance_pair["test_agent_stance_score"]
            adversary_stance = stance_pair["adversary_stance"]
            adversary_stance_score = stance_pair["adversary_stance_score"]

            for seed in config["seeds"]:
                test_agent = PromptAgent(
                    name=f"{condition}_test_agent",
                    stance=test_stance,
                    stance_score=test_stance_score,
                    llm=test_llm,
                )

                adversary_agent = AdversaryAgent(
                    name="fixed_prompting_adversary",
                    stance=adversary_stance,
                    stance_score=adversary_stance_score,
                    llm=adversary_llm,
                )

                environment = DebateEnvironment(
                    topic=topic_question,
                    test_agent=test_agent,
                    adversary_agent=adversary_agent,
                    rounds=rounds,
                    condition=condition,
                    seed=seed,
                    output_dir=output_dir,
                    experiment_name=config["experiment_name"],
                    topic_name=topic_name,
                )

                transcript = environment.run()

                print(
                    f"Finished debate: {transcript['debate_id']} "
                    f"topic={topic_name} "
                    f"test_stance={test_stance} "
                    f"adversary_stance={adversary_stance} "
                    f"seed={seed}"
                )


def validate_transcript_directory(input_dir: str) -> None:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Transcript directory not found: {input_dir}")

    transcript_files = sorted(directory.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript JSON files found in: {input_dir}")

    valid_count = 0

    for path in transcript_files:
        with path.open("r", encoding="utf-8") as f:
            transcript = json.load(f)

        validate_transcript(transcript)
        valid_count += 1
        print(f"Valid transcript: {path}")

    print(f"\nValidation complete. Valid transcripts: {valid_count}")


def score_transcript_directory(
    input_dir: str, output_dir: str, judge_config: dict | None = None
) -> None:
    transcript_dir = Path(input_dir)

    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {input_dir}")

    transcript_files = sorted(transcript_dir.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript JSON files found in: {input_dir}")

    if judge_config is None:
        judge_config = {"backend": "mock", "model_name": "mock"}

    judge = create_judge(judge_config)

    for path in transcript_files:
        with path.open("r", encoding="utf-8") as f:
            transcript = json.load(f)

        validate_transcript(transcript)

        judged_turns = [
            judge.judge_turn(transcript=transcript, turn=turn)
            for turn in transcript["turns"]
        ]

        judged_transcript = {
            "debate_id": transcript["debate_id"],
            "experiment_name": transcript["experiment_name"],
            "condition": transcript["condition"],
            "topic_name": transcript["topic_name"],
            "topic": transcript["topic"],
            "test_agent_stance": transcript["test_agent_stance"],
            "test_agent_stance_score": transcript["test_agent_stance_score"],
            "adversary_stance": transcript["adversary_stance"],
            "adversary_stance_score": transcript["adversary_stance_score"],
            "rounds": transcript["rounds"],
            "seed": transcript["seed"],
            "judge_type": judge_config.get("backend", "mock"),
            "judge_llm": judge_metadata(judge),
            "judged_turns": judged_turns,
        }

        output_path = Path(output_dir) / f"{judged_transcript['debate_id']}_judged.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(judged_transcript, f, indent=2, ensure_ascii=False)

        print(f"Saved judged transcript to: {output_path}")

    print(f"\nScored transcripts: {len(transcript_files)}")


def compute_metrics_directory(input_dir: str, output_path: str) -> None:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Judged transcript directory not found: {input_dir}")

    judged_files = sorted(directory.glob("*_judged.json"))

    if not judged_files:
        raise FileNotFoundError(f"No judged transcript files found in: {input_dir}")

    rows = []

    for path in judged_files:
        with path.open("r", encoding="utf-8") as f:
            judged_transcript = json.load(f)

        test_agent_flip_metrics = compute_flip_metrics(
            judged_turns=judged_transcript["judged_turns"],
            speaker="test_agent",
        )

        adversary_flip_metrics = compute_flip_metrics(
            judged_turns=judged_transcript["judged_turns"],
            speaker="adversary",
        )

        row = {
            "debate_id": judged_transcript["debate_id"],
            "experiment_name": judged_transcript["experiment_name"],
            "condition": judged_transcript["condition"],
            "topic_name": judged_transcript["topic_name"],
            "test_agent_stance": judged_transcript["test_agent_stance"],
            "test_agent_stance_score": judged_transcript["test_agent_stance_score"],
            "adversary_stance": judged_transcript["adversary_stance"],
            "adversary_stance_score": judged_transcript["adversary_stance_score"],
            "rounds": judged_transcript["rounds"],
            "seed": judged_transcript["seed"],
            "judge_type": judged_transcript["judge_type"],
            "strict_tof": test_agent_flip_metrics["strict_tof"],
            "strict_nof": test_agent_flip_metrics["strict_nof"],
            "polarity_tof": test_agent_flip_metrics["polarity_tof"],
            "polarity_nof": test_agent_flip_metrics["polarity_nof"],
            "adversary_strict_tof": adversary_flip_metrics["strict_tof"],
            "adversary_strict_nof": adversary_flip_metrics["strict_nof"],
            "adversary_polarity_tof": adversary_flip_metrics["polarity_tof"],
            "adversary_polarity_nof": adversary_flip_metrics["polarity_nof"],
        }

        rows.append(row)
        print(f"Computed metrics for: {path}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics CSV to: {path}")


def run_full_prompting_debug_pipeline(
    config_path: str = "configs/prompting_debug.json",
    transcript_dir: str = "outputs/transcripts",
    judge_score_dir: str = "outputs/judge_scores",
    metrics_dir: str = "outputs/metrics",
    metrics_output_path: str = "outputs/metrics/prompting_debug_metrics.csv",
) -> None:
    config = load_json_config(config_path)
    judge_config = config["models"]["judge"]

    print("Clearing old debug outputs...")
    clear_json_files(transcript_dir)
    clear_json_files(judge_score_dir)
    clear_csv_files(metrics_dir)

    print("\nRunning prompting debates...")
    run_prompting_experiments(config_path)

    print("\nValidating transcripts...")
    validate_transcript_directory(transcript_dir)

    print(f"\nScoring transcripts with {judge_config['backend']} judge...")
    score_transcript_directory(
        input_dir=transcript_dir,
        output_dir=judge_score_dir,
        judge_config=judge_config,
    )

    print("\nComputing metrics...")
    compute_metrics_directory(
        input_dir=judge_score_dir,
        output_path=metrics_output_path,
    )

    print("\nPrompting debug pipeline complete.")
    print(f"Metrics saved to: {metrics_output_path}")
