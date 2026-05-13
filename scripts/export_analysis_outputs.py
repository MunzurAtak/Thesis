import argparse
from pathlib import Path

from export_turn_scores import export_turn_scores
from export_drift_curve import export_drift_curve
from plot_drift_curve import plot_drift_curve
from summarize_metrics import summarize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export turn scores, drift curve data, and drift curve plot."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name used for output filenames.",
    )

    parser.add_argument(
        "--judge-score-dir",
        type=str,
        default="outputs/judge_scores",
        help="Directory containing judged transcript JSON files.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Directory where CSV files will be saved.",
    )

    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots",
        help="Directory where plots will be saved.",
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

    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)

    metrics_path = metrics_dir / f"{args.experiment_name}_metrics.csv"
    summary_path = metrics_dir / f"{args.experiment_name}_summary.csv"
    turn_scores_path = metrics_dir / f"{args.experiment_name}_turn_scores.csv"
    drift_curve_path = metrics_dir / f"{args.experiment_name}_drift_curve.csv"
    drift_plot_path = plots_dir / f"{args.experiment_name}_drift_curve.png"

    print("Exporting turn-level judge scores...")
    export_turn_scores(
        input_dir=args.judge_score_dir,
        output_path=str(turn_scores_path),
    )

    print("\nExporting drift curve CSV...")
    export_drift_curve(
        input_dir=args.judge_score_dir,
        output_path=str(drift_curve_path),
        speaker=args.speaker,
    )

    print("\nPlotting drift curve...")
    plot_drift_curve(
        input_path=str(drift_curve_path),
        output_path=str(drift_plot_path),
    )

    print("\nExporting metrics summary...")
    summarize(
        metrics_path=str(metrics_path),
        output_path=str(summary_path),
    )

    print("\nAnalysis outputs complete.")
    print(f"Metrics summary: {summary_path}")
    print(f"Turn scores: {turn_scores_path}")
    print(f"Drift curve CSV: {drift_curve_path}")
    print(f"Drift curve plot: {drift_plot_path}")


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
