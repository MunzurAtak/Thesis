import argparse
import json
from pathlib import Path

from src.debate.transcript_schema import validate_transcript


def validate_transcript_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    validate_transcript(transcript)


def validate_transcript_directory(input_dir: str) -> None:
    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Transcript directory not found: {input_dir}")

    transcript_files = sorted(directory.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript JSON files found in: {input_dir}")

    valid_count = 0

    for path in transcript_files:
        validate_transcript_file(path)
        valid_count += 1
        print(f"Valid transcript: {path}")

    print(f"\nValidation complete. Valid transcripts: {valid_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate saved debate transcript JSON files."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/transcripts",
        help="Directory containing transcript JSON files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    validate_transcript_directory(args.input_dir)


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
