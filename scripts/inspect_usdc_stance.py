import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect the USDC stance dataset schema and label distribution."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="data/raw/USDC_Stance.csv",
        help="Path to USDC_Stance.csv.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/data_checks/usdc_stance_inspection.json",
        help="Path to save inspection summary JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"USDC stance file not found: {input_path}")

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_columns = [
        "subreddit",
        "title",
        "content",
        "stance_id_comment",
        "majority_vote_stance_label",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["stance_id_comment"] = df["stance_id_comment"].fillna("").astype(str)
    df["comment_length"] = df["stance_id_comment"].str.len()

    usable_mask = df["majority_vote_stance_label"].isin(
        [
            "strongly_in_favor",
            "somewhat_in_favor",
            "somewhat_against",
            "strongly_against",
        ]
    ) & (df["comment_length"] >= 30)

    usable_df = df[usable_mask].copy()

    label_counts = df["majority_vote_stance_label"].value_counts(dropna=False)
    usable_label_counts = usable_df["majority_vote_stance_label"].value_counts(
        dropna=False
    )
    subreddit_counts = df["subreddit"].value_counts(dropna=False)

    subreddit_label_table = pd.crosstab(
        df["subreddit"],
        df["majority_vote_stance_label"],
    )
    subreddit_label_table["total"] = subreddit_label_table.sum(axis=1)
    subreddit_label_table = subreddit_label_table.sort_values(
        by="total",
        ascending=False,
    )

    summary = {
        "input_path": str(input_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "n_unique_subreddits": int(df["subreddit"].nunique()),
        "label_counts": label_counts.to_dict(),
        "usable_rows_after_basic_filter": int(len(usable_df)),
        "usable_label_counts": usable_label_counts.to_dict(),
        "top_subreddits": subreddit_counts.head(30).to_dict(),
        "top_subreddit_label_table": subreddit_label_table.head(30).to_dict(
            orient="index"
        ),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel counts:")
    print(label_counts)

    print("\nUsable label counts after basic filter:")
    print(usable_label_counts)

    print("\nTop subreddits:")
    print(subreddit_counts.head(30))

    print(f"\nSaved inspection summary to: {output_path}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
