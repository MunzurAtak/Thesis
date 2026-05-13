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


SUBREDDIT_TO_TOPIC = {
    "Abortiondebate": "abortion",
    "prochoice": "abortion",
    "prolife": "abortion",
    "DebateCommunism": "communism",
    "CapitalismVSocialism": "capitalism_socialism",
    "climateskeptics": "climate_change",
    "climatechange": "climate_change",
    "climate": "climate_change",
    "brexit": "brexit",
    "gunpolitics": "gun_control",
    "progun": "gun_control",
    "GunsAreCool": "gun_control",
    "MensRights": "gender_rights",
    "Egalitarianism": "gender_rights",
    "nuclear": "nuclear_energy",
    "NuclearPower": "nuclear_energy",
    "AntiVegan": "veganism",
    "Vegetarianism": "veganism",
    "VeganActivism": "veganism",
    "Veganism": "veganism",
    "animalwelfare": "animal_rights",
    "AnimalRights": "animal_rights",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert USDC stance data into a RAG corpus JSON."
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
        default="data/rag_corpus/usdc_rag_corpus.json",
        help="Path to save RAG corpus JSON.",
    )

    parser.add_argument(
        "--min-comment-length",
        type=int,
        default=80,
        help="Minimum character length for retrieved passages.",
    )

    parser.add_argument(
        "--max-comment-length",
        type=int,
        default=1200,
        help="Maximum character length for retrieved passages.",
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
        raise FileNotFoundError(f"USDC stance file not found: {input_path}")

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_columns = [
        "subreddit",
        "title",
        "stance_id_comment",
        "majority_vote_stance_label",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    rows = []

    for _, row in df.iterrows():
        label = row["majority_vote_stance_label"]

        if label not in LABEL_TO_STANCE:
            continue

        subreddit = row["subreddit"]

        if subreddit not in SUBREDDIT_TO_TOPIC:
            continue

        text = clean_text(row["stance_id_comment"])

        if not (args.min_comment_length <= len(text) <= args.max_comment_length):
            continue

        title = clean_text(row["title"])
        topic_name = SUBREDDIT_TO_TOPIC[subreddit]

        rows.append(
            {
                "topic_name": topic_name,
                "stance": LABEL_TO_STANCE[label],
                "text": text,
                "source": "usdc",
                "subreddit": subreddit,
                "submission_id": clean_text(row.get("submission_id", "")),
                "title": title,
                "original_label": label,
                "char_length": len(text),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Saved RAG corpus to: {output_path}")
    print(f"Passages: {len(rows)}")

    if rows:
        corpus_df = pd.DataFrame(rows)
        print("\nPassages by topic:")
        print(corpus_df["topic_name"].value_counts())

        print("\nPassages by topic and stance:")
        print(pd.crosstab(corpus_df["topic_name"], corpus_df["stance"]))


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
