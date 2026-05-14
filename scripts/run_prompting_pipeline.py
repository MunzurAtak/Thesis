import argparse
from pathlib import Path

from src.pipeline.prompting_pipeline import run_full_prompting_debug_pipeline
from src.utils.config import load_json_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the prompting debate pipeline from a config file."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the prompting pipeline config JSON file.",
    )

    parser.add_argument(
        "--transcript-dir",
        type=str,
        default=None,
        help="Directory for debate transcripts. Defaults to outputs/transcripts/<experiment_name>.",
    )

    parser.add_argument(
        "--judge-score-dir",
        type=str,
        default=None,
        help="Directory for judged transcripts. Defaults to outputs/judge_scores/<experiment_name>.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Directory for metrics CSV files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)

    experiment_name = config["experiment_name"]
    metrics_output_path = str(Path(args.metrics_dir) / f"{experiment_name}_metrics.csv")
    transcript_dir = args.transcript_dir or str(Path("outputs/transcripts") / experiment_name)
    judge_score_dir = args.judge_score_dir or str(Path("outputs/judge_scores") / experiment_name)

    run_full_prompting_debug_pipeline(
        config_path=args.config,
        transcript_dir=transcript_dir,
        judge_score_dir=judge_score_dir,
        metrics_dir=args.metrics_dir,
        metrics_output_path=metrics_output_path,
    )


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
