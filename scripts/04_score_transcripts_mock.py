import argparse
import json
from pathlib import Path

from src.judge.mock_judge import MockJudge
from src.debate.transcript_schema import validate_transcript


def load_transcript(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    validate_transcript(transcript)
    return transcript


def score_transcript(transcript: dict, judge: MockJudge) -> dict:
    judged_turns = []

    for turn in transcript["turns"]:
        judged_turn = judge.judge_turn(transcript=transcript, turn=turn)
        judged_turns.append(judged_turn)

    return {
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
        "judge_type": "mock",
        "judged_turns": judged_turns,
    }


def save_judged_transcript(judged_transcript: dict, output_dir: str) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"{judged_transcript['debate_id']}_judged.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(judged_transcript, f, indent=2, ensure_ascii=False)

    print(f"Saved judged transcript to: {output_path}")


def score_transcript_directory(input_dir: str, output_dir: str) -> None:
    transcript_dir = Path(input_dir)

    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {input_dir}")

    transcript_files = sorted(transcript_dir.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript JSON files found in: {input_dir}")

    judge = MockJudge()

    for path in transcript_files:
        transcript = load_transcript(path)
        judged_transcript = score_transcript(transcript, judge)
        save_judged_transcript(judged_transcript, output_dir)

    print(f"\nScored transcripts: {len(transcript_files)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score debate transcripts with the mock judge."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/transcripts",
        help="Directory containing transcript JSON files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/judge_scores",
        help="Directory for judged transcript JSON files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    score_transcript_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
