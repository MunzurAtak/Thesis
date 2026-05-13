import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize experiment metrics CSV into compact averages."
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        required=True,
        help="Path to a metrics CSV file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the summary CSV.",
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    return float(value)


def summarize(metrics_path: str, output_path: str) -> None:
    path = Path(metrics_path)

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in metrics file: {metrics_path}")

    numeric_fields = [
        "strict_tof",
        "strict_nof",
        "polarity_tof",
        "polarity_nof",
        "adversary_strict_tof",
        "adversary_strict_nof",
        "adversary_polarity_tof",
        "adversary_polarity_nof",
    ]

    summary = {
        "experiment_name": rows[0]["experiment_name"],
        "condition": rows[0]["condition"],
        "n_debates": len(rows),
    }

    for field in numeric_fields:
        if field in rows[0]:
            values = [to_float(row[field]) for row in rows]
            summary[f"mean_{field}"] = sum(values) / len(values)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved metrics summary to: {output}")


def main():
    args = parse_args()
    summarize(
        metrics_path=args.metrics_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
