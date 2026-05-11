from src.llms.mock_llm import MockLLM
from src.agents.prompt_agent import PromptAgent
from src.debate.environment import DebateEnvironment


def main():
    topic = "Should social media platforms ban political misinformation?"
    rounds = 5
    seed = 42

    llm = MockLLM()

    test_agent = PromptAgent(
        name="prompting_test_agent",
        stance="pro",
        llm=llm,
    )

    adversary_agent = PromptAgent(
        name="fixed_prompting_adversary",
        stance="contra",
        llm=llm,
    )

    environment = DebateEnvironment(
        topic=topic,
        test_agent=test_agent,
        adversary_agent=adversary_agent,
        rounds=rounds,
        condition="prompting",
        seed=seed,
    )

    transcript = environment.run()

    print("\nDebate finished.")
    print(f"Debate ID: {transcript['debate_id']}")
    print(f"Total turns: {len(transcript['turns'])}")


if __name__ == "__main__":
    import time
    _t0 = time.time()
    main()
    print(f"\nCompleted in {time.time() - _t0:.1f}s")
