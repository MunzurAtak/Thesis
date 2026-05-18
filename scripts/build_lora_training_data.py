import argparse
import json
import random
import re
from pathlib import Path

SELECTED_TOPICS = {
    "climate_change",
    "abortion",
    "gun_control",
}

STANCE_TO_SCORE = {
    "pro": 2,
    "contra": -2,
}

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
    "coward",
    "filth",
}

TOPIC_KEYWORDS = {
    "abortion": {
        "abortion", "abortions", "pregnancy", "pregnant", "fetus", "foetus",
        "unborn", "reproductive", "bodily autonomy", "pro-choice",
        "pro-life", "roe", "wade", "viability", "mother", "womb"
    },
    "climate_change": {
        "climate", "warming", "emissions", "carbon", "co2", "greenhouse",
        "renewable", "fossil", "temperature", "environment", "ipcc",
        "sea level", "mitigation", "adaptation", "energy"
    },
    "gun_control": {
        "gun", "guns", "firearm", "firearms", "weapon", "weapons",
        "rifle", "pistol", "handgun", "background check", "licensing",
        "second amendment", "self defense", "nra", "magazine", "shooting",
        "shooter", "ammo", "ammunition", "red flag"
    },
}

UNRELATED_TOPIC_TERMS = {
    "abortion": {
        "gun", "guns", "firearm", "firearms", "second amendment", "nra"
    },
    "climate_change": {
        "abortion", "pregnancy", "fetus", "gun", "firearm", "second amendment"
    },
    "gun_control": {
        "abortion", "pregnancy", "pregnant", "fetus", "unborn", "sperm", "egg cells",
        "roe v wade", "reproductive"
    },
}

STANCE_KEYWORDS = {
    ("abortion", "pro"): {
        "legal access", "legally accessible", "bodily autonomy", "reproductive autonomy",
        "right to choose", "safe abortion", "unsafe abortion", "women's health",
        "should remain legal", "should be legal"
    },
    ("abortion", "contra"): {
        "unborn", "fetal life", "foetal life", "life begins", "pro-life",
        "sanctity of life", "should be restricted", "should be illegal",
        "should not be legal", "murder"
    },
    ("climate_change", "pro"): {
        "stronger action", "government action", "emission reduction", "reduce emissions",
        "carbon pricing", "renewable energy", "climate policy", "regulation",
        "mitigation", "greenhouse gas"
    },
    ("climate_change", "contra"): {
        "adaptation", "private sector", "market-based", "too costly", "overreach",
        "premature", "natural cycles", "not proven", "economic harm",
        "against regulation"
    },
    ("gun_control", "pro"): {
        "stricter gun control", "background checks", "background check", "licensing",
        "waiting period", "magazine limit", "red flag", "assault weapon",
        "reduce gun violence", "firearm regulation", "public safety"
    },
    ("gun_control", "contra"): {
        "second amendment", "self defense", "self-defense", "gun rights",
        "law-abiding", "law abiding", "constitutional right", "infringe",
        "black market", "criminals will", "enforce existing laws"
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build LoRA supervised fine-tuning data from the selected USDC RAG corpus."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="data/rag_corpus/usdc_selected_rag_corpus.json",
        help="Path to selected USDC corpus JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lora_training",
        help="Output directory for LoRA train/validation JSONL files.",
    )

    parser.add_argument(
        "--train-output-name",
        type=str,
        default="usdc_3topics_lora_train.jsonl",
        help="Train JSONL filename.",
    )

    parser.add_argument(
        "--val-output-name",
        type=str,
        default="usdc_3topics_lora_val.jsonl",
        help="Validation JSONL filename.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def build_system_message(stance: str) -> str:
    return (
        "You are a debate agent in a controlled thesis experiment. "
        f"You must consistently argue from the assigned {stance} stance. "
        "Do not switch sides. Do not say that you are neutral. "
        "Give a clear, concise argument that supports your assigned stance."
    )


def build_user_message(topic: str, stance: str) -> str:
    stance_description = (
        "in favor of the proposition" if stance == "pro" else "against the proposition"
    )

    return (
        f"Topic / proposition: {topic}\n\n"
        f"Assigned stance: {stance} ({stance_description}).\n\n"
        "Write one debate argument that is consistent with the assigned stance."
    )


def contains_any(text: str, terms: set[str]) -> bool:
    lower_text = text.lower()
    return any(term in lower_text for term in terms)


def clean_assistant_text(text: str) -> str:
    # Remove markdown links but keep readable anchor text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove bare URLs.
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # Remove Reddit quote lines.
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)

    # Remove leftover inline quote marker at start.
    text = re.sub(r"^\s*>+\s*", "", text)

    # Normalize whitespace.
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())

    return text.strip()


def should_keep_text(text: str) -> bool:
    if len(text) < 120:
        return False

    if len(text) > 1000:
        return False

    if contains_any(text, PROFANITY_TERMS):
        return False

    if contains_any(text, HOSTILE_PHRASES):
        return False

    return True


def contains_topic_keyword(text: str, topic_name: str) -> bool:
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in TOPIC_KEYWORDS[topic_name])


def contains_unrelated_topic_terms(text: str, topic_name: str) -> bool:
    lower_text = text.lower()
    return any(term in lower_text for term in UNRELATED_TOPIC_TERMS[topic_name])


def contains_stance_keyword(text: str, topic_name: str, stance: str) -> bool:
    lower_text = text.lower()
    keywords = STANCE_KEYWORDS[(topic_name, stance)]
    return any(keyword in lower_text for keyword in keywords)


def should_keep_example(text: str, topic_name: str, stance: str) -> bool:
    if not should_keep_text(text):
        return False

    if not contains_topic_keyword(text, topic_name):
        return False

    if contains_unrelated_topic_terms(text, topic_name):
        return False

    if not contains_stance_keyword(text, topic_name, stance):
        return False

    return True


def build_example(row: dict) -> dict:
    topic_name = row["topic_name"]
    topic = row["topic_question"]
    stance = row["stance"]
    response = clean_assistant_text(row["text"])

    return {
        "topic_name": topic_name,
        "topic": topic,
        "stance": stance,
        "stance_score": STANCE_TO_SCORE[stance],
        "source": row.get("source", "usdc"),
        "original_label": row.get("original_label"),
        "messages": [
            {
                "role": "system",
                "content": build_system_message(stance),
            },
            {
                "role": "user",
                "content": build_user_message(topic, stance),
            },
            {
                "role": "assistant",
                "content": response,
            },
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    examples = []

    for row in corpus:
        topic_name = row.get("topic_name")
        stance = row.get("stance")
        text = row.get("text", "").strip()

        if topic_name not in SELECTED_TOPICS:
            continue

        if stance not in STANCE_TO_SCORE:
            continue

        cleaned_text = clean_assistant_text(text)

        if not should_keep_example(cleaned_text, topic_name, stance):
            continue

        row = dict(row)
        row["text"] = cleaned_text

        examples.append(build_example(row))

    if not examples:
        raise ValueError("No LoRA examples were created.")

    random.shuffle(examples)

    val_size = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_output_name
    val_path = output_dir / args.val_output_name

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    print(f"Saved train data to: {train_path}")
    print(f"Saved validation data to: {val_path}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Total examples: {len(examples)}")

    print("\nExamples by topic and stance:")
    counts = {}

    for example in examples:
        key = (example["topic_name"], example["stance"])
        counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        topic_name, stance = key
        print(f"- {topic_name} / {stance}: {counts[key]}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
