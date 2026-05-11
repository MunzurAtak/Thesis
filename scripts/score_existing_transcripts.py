import argparse
from pathlib import Path

from src.pipeline.prompting_pipeline import (
    score_transcript_directory,
    compute_metrics_directory,
)
from src.utils.config import load_json_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score existing debate transcripts and compute metrics."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file containing judge settings.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/transcripts",
        help="Directory containing existing transcript JSON files.",
    )

    parser.add_argument(
        "--judge-score-dir",
        type=str,
        default="outputs/judge_scores",
        help="Directory where judged transcripts will be saved.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Directory where metrics CSV will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)

    experiment_name = config["experiment_name"]
    judge_config = config["models"]["judge"]

    metrics_output_path = str(Path(args.metrics_dir) / f"{experiment_name}_metrics.csv")

    score_transcript_directory(
        input_dir=args.input_dir,
        output_dir=args.judge_score_dir,
        judge_config=judge_config,
    )

    compute_metrics_directory(
        input_dir=args.judge_score_dir,
        output_path=metrics_output_path,
    )


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
