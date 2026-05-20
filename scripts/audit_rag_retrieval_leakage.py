import argparse
import json
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def audit_transcript(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    eval_question = transcript.get("topic", "")
    normalized_eval_question = normalize(eval_question)

    failures = 0

    for turn in transcript.get("turns", []):
        retrieval = turn.get("retrieval")
        if not retrieval:
            continue

        removed_count = retrieval.get("removed_exact_eval_question_leakage_count", 0)
        if removed_count:
            print(
                f"Filtered {removed_count} exact-question leakage passage(s) in {path.name}, "
                f"round={turn.get('round')}"
            )

        for passage in retrieval.get("retrieved_passages", []):
            passage_text = passage.get("text", "")
            if normalized_eval_question and normalized_eval_question in normalize(passage_text):
                failures += 1
                print("\nLEAKAGE FOUND")
                print(f"Transcript: {path}")
                print(f"Round: {turn.get('round')}")
                print(f"Topic: {eval_question}")
                print(f"Passage: {passage_text[:500]}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit RAG transcripts for exact evaluation-question leakage."
    )
    parser.add_argument(
        "--transcript-dir",
        required=True,
        help="Directory containing RAG transcript JSON files.",
    )
    args = parser.parse_args()

    transcript_dir = Path(args.transcript_dir)
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    total_failures = 0
    transcript_count = 0

    for path in sorted(transcript_dir.glob("*.json")):
        transcript_count += 1
        total_failures += audit_transcript(path)

    print(f"\nAudited transcripts: {transcript_count}")
    print(f"Exact evaluation-question leakage failures: {total_failures}")

    if total_failures:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
