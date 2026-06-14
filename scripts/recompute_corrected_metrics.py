from pathlib import Path
import pandas as pd

METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"

FINAL_TURN_SCORES = {
    "prompting": "prompting_usdc_3topics_final_eval_final_turn_scores.csv",
    "rag": "rag_usdc_faiss_3topics_final_eval_final_turn_scores.csv",
    "lora": "lora_usdc_3topics_final_eval_final_turn_scores.csv",
}


def _to_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def _count_reversals(sequence: list[bool]) -> int:
    """NoF (Hong et al., Eq. 2): reversals between consecutive turns."""
    return sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])


def _first_off_or_no_flip(off_rounds: list[int], max_round: int) -> int:
    return min(off_rounds) if off_rounds else max_round + 1


def per_debate_metrics(
    turn_scores: pd.DataFrame, speaker: str = "test_agent"
) -> pd.DataFrame:
    df = turn_scores[turn_scores["speaker"] == speaker].copy()
    df["strict_consistent"] = df["strict_consistent"].map(_to_bool)
    df["polarity_consistent"] = df["polarity_consistent"].map(_to_bool)

    records = []
    for debate_id, group in df.groupby("debate_id"):
        group = group.sort_values("round")
        rounds = group["round"].tolist()
        max_round = max(rounds)
        sc = group["strict_consistent"].tolist()
        pc = group["polarity_consistent"].tolist()
        strict_off_rounds = [r for r, c in zip(rounds, sc) if not c]
        pol_off_rounds = [r for r, c in zip(rounds, pc) if not c]

        records.append(
            {
                "debate_id": debate_id,
                "topic_name": group["topic_name"].iloc[0],
                "assigned_stance": group["assigned_stance"].iloc[0],
                "strict_tof": _first_off_or_no_flip(strict_off_rounds, max_round),
                "strict_nof": _count_reversals(sc),
                "strict_ostc": len(strict_off_rounds),
                "polarity_tof": _first_off_or_no_flip(pol_off_rounds, max_round),
                "polarity_nof": _count_reversals(pc),
                "polarity_ostc": len(pol_off_rounds),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    per_debate_frames = []
    summary_rows = []

    for condition, filename in FINAL_TURN_SCORES.items():
        turn_scores = pd.read_csv(METRICS_DIR / filename)
        debates = per_debate_metrics(turn_scores)
        debates.insert(0, "condition", condition)
        per_debate_frames.append(debates)

        metric_cols = [
            "polarity_tof",
            "polarity_nof",
            "polarity_ostc",
            "strict_tof",
            "strict_nof",
            "strict_ostc",
        ]
        row = {"condition": condition, "n_debates": len(debates)}
        for col in metric_cols:
            row[f"mean_{col}"] = round(debates[col].mean(), 4)
            row[f"sd_{col}"] = round(debates[col].std(), 4)
        summary_rows.append(row)

    per_debate = pd.concat(per_debate_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    per_debate_out = METRICS_DIR / "corrected_metrics_per_debate.csv"
    summary_out = METRICS_DIR / "corrected_metrics_comparison.csv"
    per_debate.to_csv(per_debate_out, index=False)
    summary.to_csv(summary_out, index=False)

    pd.set_option("display.width", 160)
    print("Corrected condition comparison (mean over 18 debates):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved: {per_debate_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
