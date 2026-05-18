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
        default=None,
        help=(
            "Directory containing existing transcript JSON files. "
            "Defaults to outputs/transcripts/<experiment_name>."
        ),
    )

    parser.add_argument(
        "--judge-score-dir",
        type=str,
        default=None,
        help=(
            "Directory where judged transcripts will be saved. "
            "Defaults to outputs/judge_scores/<experiment_name>."
        ),
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Directory where metrics CSV will be saved.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip transcripts that already have judged output files.",
    )

    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=None,
        help="Only score the first N transcript files.",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Only score the first N turns per transcript.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)

    experiment_name = config["experiment_name"]
    judge_config = config["models"]["judge"]

    input_dir = args.input_dir or str(Path("outputs/transcripts") / experiment_name)
    judge_score_dir = args.judge_score_dir or str(
        Path("outputs/judge_scores") / experiment_name
    )
    using_default_judge_score_dir = args.judge_score_dir is None

    if args.max_turns is not None and using_default_judge_score_dir:
        raise ValueError(
            "Refusing to write partial judged transcripts into the real experiment "
            "directory. Use --judge-score-dir outputs/judge_scores/_tmp_fast_test "
            "when using --max-turns."
        )
    metrics_output_path = str(Path(args.metrics_dir) / f"{experiment_name}_metrics.csv")

    score_transcript_directory(
        input_dir=input_dir,
        output_dir=judge_score_dir,
        judge_config=judge_config,
        skip_existing=args.skip_existing,
        max_transcripts=args.max_transcripts,
        max_turns=args.max_turns,
    )

    compute_metrics_directory(
        input_dir=judge_score_dir,
        output_path=metrics_output_path,
    )


if __name__ == "__main__":
    import time

    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
