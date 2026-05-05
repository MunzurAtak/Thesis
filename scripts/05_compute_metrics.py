import argparse
import csv
import json
from pathlib import Path

from src.metrics.flip_metrics import compute_flip_metrics


def load_judged_transcript(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_for_judged_transcript(judged_transcript: dict) -> dict:
    flip_metrics = compute_flip_metrics(
        judged_turns=judged_transcript["judged_turns"],
        speaker="test_agent",
    )

    return {
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
        "strict_tof": flip_metrics["strict_tof"],
        "strict_nof": flip_metrics["strict_nof"],
        "polarity_tof": flip_metrics["polarity_tof"],
        "polarity_nof": flip_metrics["polarity_nof"],
    }


def write_metrics_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        raise ValueError("No metric rows to write.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics CSV to: {path}")


def compute_metrics_directory(input_dir: str, output_path: str) -> None:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Judged transcript directory not found: {input_dir}")

    judged_files = sorted(directory.glob("*_judged.json"))

    if not judged_files:
        raise FileNotFoundError(f"No judged transcript files found in: {input_dir}")

    rows = []

    for path in judged_files:
        judged_transcript = load_judged_transcript(path)
        row = compute_metrics_for_judged_transcript(judged_transcript)
        rows.append(row)
        print(f"Computed metrics for: {path}")

    write_metrics_csv(rows, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute stance flip metrics from judged transcripts."
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
        default="outputs/metrics/prompting_debug_metrics.csv",
        help="Path for the metrics CSV file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    compute_metrics_directory(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
