import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare condition summary CSV files side by side."
    )

    parser.add_argument(
        "--summary-paths",
        nargs="+",
        required=True,
        help="Paths to summary CSV files.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save combined comparison CSV.",
    )

    return parser.parse_args()


def read_single_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}, got {len(rows)}")

    return rows[0]


def main():
    args = parse_args()

    rows = [read_single_summary(Path(path)) for path in args.summary_paths]

    fieldnames = list(rows[0].keys())

    for row in rows:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved condition comparison to: {output_path}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
