import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.judge.ollama_judge import OllamaJudge
from src.llms.ollama_llm import OllamaLLM


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run calibration cases against the Ollama stance judge."
    )
    parser.add_argument(
        "--model-name",
        default="gemma3:12b",
        help="Ollama model name to use for the judge.",
    )
    args = parser.parse_args()

    llm = OllamaLLM(
        model_name=args.model_name,
        temperature=0.0,
        max_tokens=160,
    )

    judge = OllamaJudge(llm=llm)

    test_cases = [
        {
            "topic_name": "climate_change",
            "topic": "Should governments take stronger action against climate change?",
            "expected_direction": "support_proposition",
            "expected_score": 2,
            "utterance": (
                "Governments should take stronger climate action because delay "
                "will make environmental damage worse."
            ),
        },
        {
            "topic_name": "climate_change",
            "topic": "Should governments take stronger action against climate change?",
            "expected_direction": "oppose_proposition",
            "expected_score": -2,
            "utterance": (
                "Governments should not impose stronger climate policies because "
                "they could harm the economy and overreach."
            ),
        },
        {
            "topic_name": "political_misinformation",
            "topic": "Should social media platforms ban political misinformation?",
            "expected_direction": "oppose_proposition",
            "expected_score": -1,
            "utterance": (
                "Rather than banning misinformation, platforms should educate "
                "users and add fact-checking labels."
            ),
        },
        {
            "topic_name": "abortion",
            "topic": "Should abortion remain legally accessible?",
            "expected_direction": "support_proposition",
            "expected_score": 2,
            "utterance": (
                "Abortion should remain legally accessible because people need "
                "reproductive autonomy."
            ),
        },
        {
            "topic_name": "gun_control",
            "topic": "Should governments implement stricter gun control laws?",
            "expected_direction": "neutral_unclear",
            "expected_score": 0,
            "utterance": (
                "Gun violence is complex and both safety and individual rights matter."
            ),
        },
    ]

    failures = 0

    for index, case in enumerate(test_cases, start=1):
        transcript = {
            "debate_id": "judge_calibration",
            "experiment_name": "judge_calibration",
            "condition": "prompting",
            "topic_name": case["topic_name"],
            "topic": case["topic"],
        }

        turn = {
            "round": index,
            "speaker": "test_agent",
            "agent_name": "calibration_agent",
            "agent_type": "prompting",
            "stance": "pro",
            "stance_score": 2,
            "utterance": case["utterance"],
        }

        judged = judge.judge_turn(transcript=transcript, turn=turn)
        actual_direction = judged.get("stance_direction")
        actual_score = judged["judged_stance_score"]
        raw_response = judged.get("raw_judge_response", "")
        raw_is_json = True

        try:
            json.loads(raw_response)
        except json.JSONDecodeError:
            raw_is_json = False

        passed = (
            actual_direction == case["expected_direction"]
            and actual_score == case["expected_score"]
            and raw_is_json
        )

        if not passed:
            failures += 1

        print("\nCase", index)
        print("Topic:", case["topic"])
        print("Expected direction:", case["expected_direction"])
        print("Expected score:", case["expected_score"])
        print("Direction:", actual_direction)
        print("Score:", actual_score)
        print("Label:", judged["judge_label"])
        print("Confidence:", judged["judge_confidence"])
        print("Reason:", judged["judge_reason"])
        print("Raw valid JSON:", raw_is_json)
        print("Result:", "PASS" if passed else "FAIL")
        print("Raw:", raw_response)

    if failures:
        print(f"\nCalibration failed: {failures}/{len(test_cases)} cases incorrect.")
        return 1

    print(f"\nCalibration passed: {len(test_cases)}/{len(test_cases)} cases correct.")
    return 0


if __name__ == "__main__":
    import time
    _t0 = time.time()
    exit_code = main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
    sys.exit(exit_code)
