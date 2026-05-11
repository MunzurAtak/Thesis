import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_judged_transcript(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_drift_rows(judged_transcripts: list[dict], speaker: str) -> list[dict]:
    grouped_scores = defaultdict(list)

    for transcript in judged_transcripts:
        for turn in transcript["judged_turns"]:
            if turn["speaker"] != speaker:
                continue

            key = (
                transcript["experiment_name"],
                transcript["condition"],
                transcript["topic_name"],
                transcript["test_agent_stance"],
                turn["round"],
            )

            grouped_scores[key].append(turn["judged_stance_score"])

    rows = []

    for key, scores in sorted(grouped_scores.items()):
        (
            experiment_name,
            condition,
            topic_name,
            test_agent_stance,
            round_number,
        ) = key

        mean_score = sum(scores) / len(scores)

        rows.append(
            {
                "experiment_name": experiment_name,
                "condition": condition,
                "topic_name": topic_name,
                "test_agent_stance": test_agent_stance,
                "round": round_number,
                "mean_judged_stance_score": mean_score,
                "n_turns": len(scores),
            }
        )

    return rows


def export_drift_curve(input_dir: str, output_path: str, speaker: str) -> None:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Judged transcript directory not found: {input_dir}")

    judged_files = sorted(directory.glob("*_judged.json"))

    if not judged_files:
        raise FileNotFoundError(f"No judged transcript files found in: {input_dir}")

    judged_transcripts = [load_judged_transcript(path) for path in judged_files]

    rows = build_drift_rows(
        judged_transcripts=judged_transcripts,
        speaker=speaker,
    )

    if not rows:
        raise ValueError(f"No rows found for speaker: {speaker}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved drift curve CSV to: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export stance drift curve data from judged transcripts."
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
        default="outputs/metrics/drift_curve.csv",
        help="Output path for drift curve CSV.",
    )

    parser.add_argument(
        "--speaker",
        type=str,
        default="test_agent",
        help="Speaker to export drift curve for.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    export_drift_curve(
        input_dir=args.input_dir,
        output_path=args.output_path,
        speaker=args.speaker,
    )


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
