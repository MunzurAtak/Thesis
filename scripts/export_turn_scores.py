import argparse
import csv
import json
from pathlib import Path

from src.metrics.stance import (
    is_strict_stance_consistent,
    is_polarity_stance_consistent,
    aligned_stance_score,
)


def load_judged_transcript(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_turn_score_rows(judged_transcript: dict) -> list[dict]:
    rows = []

    for turn in judged_transcript["judged_turns"]:
        assigned_score = turn["assigned_stance_score"]
        judged_score = turn["judged_stance_score"]

        strict_consistent = is_strict_stance_consistent(
            assigned_stance_score=assigned_score,
            judged_stance_score=judged_score,
        )

        polarity_consistent = is_polarity_stance_consistent(
            assigned_stance_score=assigned_score,
            judged_stance_score=judged_score,
        )

        aligned_score = aligned_stance_score(
            assigned_stance_score=assigned_score,
            judged_stance_score=judged_score,
        )

        rows.append(
            {
                "debate_id": judged_transcript["debate_id"],
                "experiment_name": judged_transcript["experiment_name"],
                "condition": judged_transcript["condition"],
                "topic_name": judged_transcript["topic_name"],
                "round": turn["round"],
                "speaker": turn["speaker"],
                "agent_name": turn["agent_name"],
                "agent_type": turn["agent_type"],
                "assigned_stance": turn["assigned_stance"],
                "assigned_stance_score": assigned_score,
                "judged_stance_score": judged_score,
                "aligned_stance_score": aligned_score,
                "strict_consistent": strict_consistent,
                "polarity_consistent": polarity_consistent,
                "judge_label": turn["judge_label"],
                "judge_confidence": turn["judge_confidence"],
                "judge_reason": turn["judge_reason"],
                "utterance": turn["utterance"],
            }
        )

    return rows


def export_turn_scores(input_dir: str, output_path: str) -> None:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Judged transcript directory not found: {input_dir}")

    judged_files = sorted(directory.glob("*_judged.json"))

    if not judged_files:
        raise FileNotFoundError(f"No judged transcript files found in: {input_dir}")

    rows = []

    for path in judged_files:
        judged_transcript = load_judged_transcript(path)
        transcript_rows = build_turn_score_rows(judged_transcript)
        rows.extend(transcript_rows)
        print(f"Exported turn scores from: {path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved turn-level score CSV to: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export turn-level judge scores from judged transcript JSON files."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/judge_scores",
        help="Directory containing judged transcript JSON files.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/metrics/turn_scores.csv",
        help="Output path for turn-level score CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    export_turn_scores(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
