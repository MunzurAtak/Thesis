import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect USDC stance targets and label distributions."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="data/raw/USDC_Stance.csv",
        help="Path to USDC_Stance.csv.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/data_checks/usdc_stance_targets.json",
        help="Path to save JSON summary.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/data_checks/usdc_stance_targets.csv",
        help="Path to save CSV summary.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"USDC file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = [
        "subreddit",
        "title",
        "stance_id",
        "majority_vote_stance_label",
        "stance_id_comment",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["stance_id_comment"] = df["stance_id_comment"].fillna("").astype(str)
    df["comment_length"] = df["stance_id_comment"].str.len()

    usable = df[
        df["majority_vote_stance_label"].isin(
            [
                "strongly_in_favor",
                "somewhat_in_favor",
                "somewhat_against",
                "strongly_against",
            ]
        )
        & (df["comment_length"] >= 80)
    ].copy()

    grouped = (
        usable.groupby(["subreddit", "stance_id", "majority_vote_stance_label"])
        .size()
        .reset_index(name="count")
    )

    pivot = grouped.pivot_table(
        index=["subreddit", "stance_id"],
        columns="majority_vote_stance_label",
        values="count",
        fill_value=0,
    ).reset_index()

    for col in [
        "strongly_in_favor",
        "somewhat_in_favor",
        "somewhat_against",
        "strongly_against",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["pro_total"] = pivot["strongly_in_favor"] + pivot["somewhat_in_favor"]
    pivot["contra_total"] = pivot["strongly_against"] + pivot["somewhat_against"]
    pivot["total"] = pivot["pro_total"] + pivot["contra_total"]
    pivot["min_side_count"] = pivot[["pro_total", "contra_total"]].min(axis=1)

    pivot = pivot.sort_values(
        by=["min_side_count", "total"],
        ascending=False,
    )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    pivot.to_csv(output_csv, index=False)

    summary = {
        "input_path": str(input_path),
        "usable_rows": int(len(usable)),
        "n_unique_subreddits": int(usable["subreddit"].nunique()),
        "n_unique_stance_ids": int(usable["stance_id"].nunique()),
        "top_balanced_stance_targets": pivot.head(50).to_dict(orient="records"),
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Usable rows: {len(usable)}")
    print(f"Unique subreddits: {usable['subreddit'].nunique()}")
    print(f"Unique stance_id values: {usable['stance_id'].nunique()}")

    print("\nTop balanced stance targets:")
    print(
        pivot[
            [
                "subreddit",
                "stance_id",
                "pro_total",
                "contra_total",
                "total",
                "min_side_count",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print(f"\nSaved CSV to: {output_csv}")
    print(f"Saved JSON to: {output_json}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
