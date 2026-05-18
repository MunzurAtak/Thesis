import argparse
import json
from pathlib import Path

PROFANITY_TERMS = {
    "fuck",
    "fucking",
    "shit",
    "bullshit",
    "bitch",
    "asshole",
}

HOSTILE_PHRASES = {
    "you don't like facts",
    "you are wrong",
    "you're wrong",
    "idiot",
    "stupid",
    "moron",
}

QUOTE_MARKERS = {
    ">",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit LoRA JSONL training data for noisy Reddit-style examples."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/lora_training/usdc_3topics_lora_train.jsonl",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)

    return rows


def get_assistant_text(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return message.get("content", "")

    return ""


def has_any(text: str, terms: set[str]) -> bool:
    lower_text = text.lower()
    return any(term in lower_text for term in terms)


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_jsonl(input_path)

    stats = {
        "total": len(rows),
        "contains_quote_marker": 0,
        "contains_profanity": 0,
        "contains_hostile_phrase": 0,
        "too_short_under_80_chars": 0,
        "too_long_over_1200_chars": 0,
        "contains_url": 0,
    }

    examples = {
        "contains_quote_marker": [],
        "contains_profanity": [],
        "contains_hostile_phrase": [],
        "too_short_under_80_chars": [],
        "too_long_over_1200_chars": [],
        "contains_url": [],
    }

    topic_stance_counts = {}

    for row in rows:
        text = get_assistant_text(row)
        topic = row.get("topic_name", "unknown")
        stance = row.get("stance", "unknown")

        key = (topic, stance)
        topic_stance_counts[key] = topic_stance_counts.get(key, 0) + 1

        checks = {
            "contains_quote_marker": text.strip().startswith(">") or "\n>" in text,
            "contains_profanity": has_any(text, PROFANITY_TERMS),
            "contains_hostile_phrase": has_any(text, HOSTILE_PHRASES),
            "too_short_under_80_chars": len(text) < 80,
            "too_long_over_1200_chars": len(text) > 1200,
            "contains_url": "http://" in text or "https://" in text or "www." in text,
        }

        for check_name, failed in checks.items():
            if failed:
                stats[check_name] += 1

                if len(examples[check_name]) < 3:
                    examples[check_name].append(
                        {
                            "line": row["_line_number"],
                            "topic_name": topic,
                            "stance": stance,
                            "text_preview": text[:250].replace("\n", " "),
                        }
                    )

    print(f"Audited file: {input_path}")
    print("\nCounts by topic and stance:")
    for key in sorted(topic_stance_counts):
        topic, stance = key
        print(f"- {topic} / {stance}: {topic_stance_counts[key]}")

    print("\nNoise statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

    print("\nExample flagged rows:")
    for check_name, flagged_examples in examples.items():
        print(f"\n{check_name}:")
        if not flagged_examples:
            print("- none")
            continue

        for example in flagged_examples:
            print(
                f"- line={example['line']} "
                f"topic={example['topic_name']} "
                f"stance={example['stance']} "
                f"text={example['text_preview']}"
            )


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
