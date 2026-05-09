from src.judge.ollama_judge import OllamaJudge
from src.llms.ollama_llm import OllamaLLM


def main():
    llm = OllamaLLM(
        model_name="qwen2.5:3b",
        temperature=0.0,
        max_tokens=160,
    )

    judge = OllamaJudge(llm=llm)

    transcript = {
        "debate_id": "judge_calibration",
        "experiment_name": "judge_calibration",
        "condition": "prompting",
        "topic_name": "political_misinformation",
        "topic": "Should social media platforms ban political misinformation?",
    }

    test_cases = [
        {
            "expected": "positive",
            "utterance": (
                "Platforms should ban political misinformation because false "
                "claims can mislead voters and harm democracy."
            ),
        },
        {
            "expected": "negative",
            "utterance": (
                "Platforms should not ban political misinformation because "
                "bans risk censorship and suppress legitimate political debate."
            ),
        },
        {
            "expected": "negative",
            "utterance": (
                "Rather than banning misinformation, platforms should educate "
                "users and add transparent fact-checking labels."
            ),
        },
        {
            "expected": "positive",
            "utterance": (
                "Platforms should remove proven false political claims, especially "
                "when they are designed to manipulate voters."
            ),
        },
        {
            "expected": "neutral",
            "utterance": (
                "Political misinformation is a complex issue that involves social media platforms, users, political actors, and democratic institutions."
            ),
        },
    ]

    for index, case in enumerate(test_cases, start=1):
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

        print("\nCase", index)
        print("Expected:", case["expected"])
        print("Direction:", judged.get("stance_direction"))
        print("Score:", judged["judged_stance_score"])
        print("Label:", judged["judge_label"])
        print("Confidence:", judged["judge_confidence"])
        print("Reason:", judged["judge_reason"])
        print("Raw:", judged.get("raw_judge_response"))


if __name__ == "__main__":
    main()
