import argparse
import json
from pathlib import Path

import pandas as pd

LABEL_TO_STANCE = {
    "strongly_in_favor": "pro",
    "somewhat_in_favor": "pro",
    "somewhat_against": "contra",
    "strongly_against": "contra",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect USDC subreddit-level topic candidates for RAG."
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
        default="outputs/data_checks/usdc_topic_candidates.json",
        help="Path to save topic candidate inspection JSON.",
    )

    parser.add_argument(
        "--min-comment-length",
        type=int,
        default=80,
        help="Minimum comment length.",
    )

    parser.add_argument(
        "--examples-per-label",
        type=int,
        default=3,
        help="Number of examples to show per subreddit and stance label.",
    )

    return parser.parse_args()


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"USDC file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = [
        "subreddit",
        "title",
        "content",
        "stance_id_comment",
        "majority_vote_stance_label",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["stance_id_comment"] = df["stance_id_comment"].fillna("").astype(str)
    df["comment_length"] = df["stance_id_comment"].str.len()

    usable = df[
        df["majority_vote_stance_label"].isin(LABEL_TO_STANCE.keys())
        & (df["comment_length"] >= args.min_comment_length)
    ].copy()

    usable["mapped_stance"] = usable["majority_vote_stance_label"].map(LABEL_TO_STANCE)

    label_table = pd.crosstab(
        usable["subreddit"],
        usable["mapped_stance"],
    )

    for col in ["pro", "contra"]:
        if col not in label_table.columns:
            label_table[col] = 0

    label_table["total"] = label_table["pro"] + label_table["contra"]
    label_table["min_side_count"] = label_table[["pro", "contra"]].min(axis=1)
    label_table = label_table.sort_values(
        by=["min_side_count", "total"],
        ascending=False,
    )

    examples = {}

    for subreddit in label_table.index:
        examples[subreddit] = {}

        subset = usable[usable["subreddit"] == subreddit]

        for stance in ["pro", "contra"]:
            stance_subset = subset[subset["mapped_stance"] == stance].head(
                args.examples_per_label
            )

            examples[subreddit][stance] = [
                {
                    "title": clean_text(row["title"]),
                    "content": clean_text(row["content"])[:500],
                    "comment": clean_text(row["stance_id_comment"])[:700],
                    "original_label": row["majority_vote_stance_label"],
                    "comment_length": int(row["comment_length"]),
                }
                for _, row in stance_subset.iterrows()
            ]

    summary = {
        "input_path": str(input_path),
        "usable_rows": int(len(usable)),
        "topic_balance": label_table.reset_index().to_dict(orient="records"),
        "examples": examples,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Usable rows: {len(usable)}")

    print("\nSubreddit balance:")
    print(label_table.head(30).to_string())

    print("\nExample preview for top 5 balanced subreddits:")
    for subreddit in label_table.head(5).index:
        print(f"\n=== {subreddit} ===")

        for stance in ["pro", "contra"]:
            print(f"\n{stance.upper()} examples:")
            for example in examples[subreddit][stance]:
                print(f"- label={example['original_label']}")
                print(f"  title: {example['title'][:180]}")
                print(f"  comment: {example['comment'][:300]}")

    print(f"\nSaved inspection JSON to: {output_path}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
