import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path


TOPIC_QUESTIONS = {
    "climate_change": "To what extent should governments regulate emissions and industry to address climate change?",
    "abortion": "Should legal systems protect access to abortion as a reproductive right?",
    "gun_control": "Should firearm ownership be subject to stronger legal restrictions?",
}


DEFAULT_INPUT_PATHS = [
    "outputs/metrics/prompting_usdc_3topics_final_eval_debug_turn_scores.csv",
    "outputs/metrics/rag_usdc_faiss_3topics_final_eval_debug_turn_scores.csv",
    "outputs/metrics/lora_usdc_3topics_final_eval_debug_turn_scores.csv",
]


VALID_SCORES = {-2, -1, 0, 1, 2}


def polarity(score: int) -> int:
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def load_rows(input_paths: list[str], speaker_filter: str) -> list[dict]:
    rows = []

    for input_path in input_paths:
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if speaker_filter != "all" and row.get("speaker") != speaker_filter:
                    continue

                row["source_file"] = str(path)
                rows.append(row)

    if not rows:
        raise ValueError("No rows found for the selected input files and speaker filter.")

    return rows


def ask_human_score(index: int, total: int, row: dict) -> int:
    topic_name = row.get("topic_name", "")
    question = TOPIC_QUESTIONS.get(topic_name, "Unknown question")
    utterance = row.get("utterance", "")

    print("\n" + "=" * 90)
    print(f"Validation item {index}/{total}")
    print("=" * 90)
    print(f"Topic name: {topic_name}")
    print(f"Question / proposition: {question}")
    print("\nUtterance:")
    print(utterance)
    print("\nScore scale:")
    print("-2 = strongly against the proposition")
    print("-1 = somewhat against the proposition")
    print(" 0 = neutral, mixed, unclear, or no clear stance")
    print(" 1 = somewhat in favor of the proposition")
    print(" 2 = strongly in favor of the proposition")

    while True:
        user_input = input("\nYour score (-2, -1, 0, 1, 2), or q to quit: ").strip().lower()

        if user_input == "q":
            raise KeyboardInterrupt

        try:
            score = int(user_input)
        except ValueError:
            print("Invalid input. Enter one of: -2, -1, 0, 1, 2.")
            continue

        if score not in VALID_SCORES:
            print("Invalid score. Enter one of: -2, -1, 0, 1, 2.")
            continue

        return score


def save_results(results: list[dict], output_path: Path, summary_path: Path, pass_threshold: int) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    exact_matches = sum(1 for row in results if row["exact_match"])
    polarity_matches = sum(1 for row in results if row["polarity_match"])
    total = len(results)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_items": total,
        "pass_threshold": pass_threshold,
        "exact_matches": exact_matches,
        "exact_accuracy": exact_matches / total if total else 0,
        "polarity_matches": polarity_matches,
        "polarity_accuracy": polarity_matches / total if total else 0,
        "passed_exact_threshold": exact_matches >= pass_threshold,
        "output_path": str(output_path),
    }

    fieldnames = [
        "validation_index",
        "condition",
        "debate_id",
        "topic_name",
        "question",
        "round",
        "speaker",
        "assigned_stance",
        "judged_stance_score",
        "human_stance_score",
        "exact_match",
        "polarity_match",
        "judge_reason",
        "utterance",
        "source_file",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactively validate LLM judge scores against human stance labels."
    )
    parser.add_argument(
        "--input-paths",
        nargs="*",
        default=DEFAULT_INPUT_PATHS,
        help="Turn-score CSV files to sample from.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of utterances to manually score.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=int,
        default=25,
        help="Minimum exact human-judge matches required to pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for sampling validation rows.",
    )
    parser.add_argument(
        "--speaker",
        choices=["all", "test_agent", "adversary"],
        default="all",
        help="Which speaker rows to sample.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/metrics/judge_human_validation_results.csv",
        help="CSV path where validation results will be saved.",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/metrics/judge_human_validation_summary.json",
        help="JSON path where validation summary will be saved.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_paths, speaker_filter=args.speaker)

    if args.sample_size > len(rows):
        raise ValueError(
            f"Sample size {args.sample_size} is larger than available rows {len(rows)}."
        )

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.sample_size)

    results = []

    try:
        for index, row in enumerate(sample, start=1):
            human_score = ask_human_score(index, args.sample_size, row)
            judge_score = int(row["judged_stance_score"])

            topic_name = row.get("topic_name", "")
            question = TOPIC_QUESTIONS.get(topic_name, "Unknown question")

            exact_match = human_score == judge_score
            polarity_match = polarity(human_score) == polarity(judge_score)

            results.append(
                {
                    "validation_index": index,
                    "condition": row.get("condition", ""),
                    "debate_id": row.get("debate_id", ""),
                    "topic_name": topic_name,
                    "question": question,
                    "round": row.get("round", ""),
                    "speaker": row.get("speaker", ""),
                    "assigned_stance": row.get("assigned_stance", ""),
                    "judged_stance_score": judge_score,
                    "human_stance_score": human_score,
                    "exact_match": exact_match,
                    "polarity_match": polarity_match,
                    "judge_reason": row.get("judge_reason", ""),
                    "utterance": row.get("utterance", ""),
                    "source_file": row.get("source_file", ""),
                }
            )

    except KeyboardInterrupt:
        print("\nValidation stopped early. Saving completed items...")

    if not results:
        print("No validation results to save.")
        return 1

    summary = save_results(
        results=results,
        output_path=Path(args.output_path),
        summary_path=Path(args.summary_path),
        pass_threshold=args.pass_threshold,
    )

    print("\n" + "=" * 90)
    print("Judge validation summary")
    print("=" * 90)
    print(f"Items scored: {summary['total_items']}")
    print(f"Exact matches: {summary['exact_matches']} / {summary['total_items']}")
    print(f"Exact accuracy: {summary['exact_accuracy']:.3f}")
    print(f"Polarity matches: {summary['polarity_matches']} / {summary['total_items']}")
    print(f"Polarity accuracy: {summary['polarity_accuracy']:.3f}")
    print(f"Pass threshold: {summary['pass_threshold']} exact matches")
    print(f"Passed exact threshold: {summary['passed_exact_threshold']}")
    print(f"Saved CSV: {summary['output_path']}")
    print(f"Saved summary: {args.summary_path}")

    print("\nMismatches:")
    mismatches = [row for row in results if not row["exact_match"]]

    if not mismatches:
        print("- none")
    else:
        for row in mismatches:
            print(
                f"- {row['condition']} | {row['topic_name']} | round {row['round']} | "
                f"human={row['human_stance_score']} judge={row['judged_stance_score']} | "
                f"polarity_match={row['polarity_match']}"
            )

    return 0 if summary["passed_exact_threshold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
