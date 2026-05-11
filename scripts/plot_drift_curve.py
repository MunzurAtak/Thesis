import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_drift_rows(input_path: str) -> list[dict]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Drift curve CSV not found: {input_path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["experiment_name"],
            row["condition"],
            row["topic_name"],
            row["test_agent_stance"],
        )

        grouped[key].append(row)

    return grouped


def plot_drift_curve(input_path: str, output_path: str) -> None:
    rows = load_drift_rows(input_path)

    if not rows:
        raise ValueError("No rows found in drift curve CSV.")

    grouped = group_rows(rows)

    plt.figure(figsize=(10, 6))

    for key, group in grouped.items():
        experiment_name, condition, topic_name, test_agent_stance = key

        group = sorted(group, key=lambda row: int(row["round"]))

        rounds = [int(row["round"]) for row in group]
        mean_scores = [float(row["mean_judged_stance_score"]) for row in group]

        label = f"{condition} | {topic_name} | {test_agent_stance}"

        plt.plot(
            rounds,
            mean_scores,
            marker="o",
            label=label,
        )

    plt.axhline(y=0, linestyle="--", linewidth=1)

    plt.title("Stance Drift Curve")
    plt.xlabel("Debate round")
    plt.ylabel("Mean judged stance score")
    plt.xticks(sorted({int(row["round"]) for row in rows}))
    plt.yticks([-2, -1, 0, 1, 2])
    plt.ylim(-2.2, 2.2)
    plt.legend()
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved drift curve plot to: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stance drift curve from drift curve CSV."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to drift curve CSV file.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where the PNG plot should be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    plot_drift_curve(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
