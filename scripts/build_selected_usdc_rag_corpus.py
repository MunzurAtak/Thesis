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


TOPIC_MAPPINGS = {
    "communism": {
        "question": "Should society adopt communism?",
        "subreddits": ["DebateCommunism"],
    },
    "climate_change": {
        "question": "Should governments take stronger action against climate change?",
        "subreddits": ["climatechange", "climate", "climateskeptics"],
    },
    "gun_control": {
        "question": "Should governments implement stricter gun control laws?",
        "subreddits": ["gunpolitics", "progun", "GunsAreCool"],
    },
    "abortion": {
        "question": "Should abortion remain legally accessible?",
        "subreddits": ["Abortiondebate"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build selected-topic USDC RAG corpus."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="data/raw/USDC_Stance.csv",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/rag_corpus/usdc_selected_rag_corpus.json",
    )

    parser.add_argument(
        "--min-comment-length",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--max-comment-length",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-per-topic-stance",
        type=int,
        default=250,
        help="Maximum passages per topic and stance.",
    )

    return parser.parse_args()


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def topic_for_subreddit(subreddit: str) -> str | None:
    for topic_name, config in TOPIC_MAPPINGS.items():
        if subreddit in config["subreddits"]:
            return topic_name

    return None


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"USDC file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = [
        "submission_id",
        "subreddit",
        "title",
        "content",
        "stance_id_comment",
        "majority_vote_stance_label",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for _, row in df.iterrows():
        label = row["majority_vote_stance_label"]

        if label not in LABEL_TO_STANCE:
            continue

        subreddit = row["subreddit"]
        topic_name = topic_for_subreddit(subreddit)

        if topic_name is None:
            continue

        text = clean_text(row["stance_id_comment"])

        if not (args.min_comment_length <= len(text) <= args.max_comment_length):
            continue

        rows.append(
            {
                "topic_name": topic_name,
                "topic_question": TOPIC_MAPPINGS[topic_name]["question"],
                "stance": LABEL_TO_STANCE[label],
                "text": text,
                "source": "usdc",
                "subreddit": subreddit,
                "submission_id": clean_text(row["submission_id"]),
                "title": clean_text(row["title"]),
                "original_label": label,
                "char_length": len(text),
            }
        )

    corpus_df = pd.DataFrame(rows)

    if corpus_df.empty:
        raise ValueError("No corpus rows were created.")

    sampled_parts = []

    for topic_name in sorted(corpus_df["topic_name"].unique()):
        for stance in ["pro", "contra"]:
            subset = corpus_df[
                (corpus_df["topic_name"] == topic_name)
                & (corpus_df["stance"] == stance)
            ].copy()

            subset = subset.sample(
                n=min(len(subset), args.max_per_topic_stance),
                random_state=42,
            )

            sampled_parts.append(subset)

    final_df = pd.concat(sampled_parts, ignore_index=True)
    final_rows = final_df.to_dict(orient="records")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved selected USDC RAG corpus to: {output_path}")
    print(f"Passages: {len(final_rows)}")

    print("\nPassages by topic and stance:")
    print(pd.crosstab(final_df["topic_name"], final_df["stance"]))

    print("\nTopic questions:")
    for topic_name, config in TOPIC_MAPPINGS.items():
        print(f"- {topic_name}: {config['question']}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
