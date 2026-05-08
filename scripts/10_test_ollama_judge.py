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
        "debate_id": "test_debate",
        "experiment_name": "judge_test",
        "condition": "prompting",
        "topic_name": "political_misinformation",
        "topic": "Should social media platforms ban political misinformation?",
    }

    turn = {
        "round": 1,
        "speaker": "test_agent",
        "agent_name": "prompting_test_agent",
        "agent_type": "prompting",
        "stance": "pro",
        "stance_score": 2,
        "utterance": (
            "Social media platforms should ban political misinformation because "
            "false claims can mislead voters, damage democratic processes, and "
            "spread faster than corrections."
        ),
    }

    judged_turn = judge.judge_turn(transcript=transcript, turn=turn)

    print("\nJudged turn:")
    print(judged_turn)


if __name__ == "__main__":
    main()
