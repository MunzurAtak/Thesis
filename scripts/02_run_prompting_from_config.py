from src.agents.prompt_agent import PromptAgent
from src.debate.environment import DebateEnvironment
from src.llms.mock_llm import MockLLM
from src.utils.config import load_json_config


def run_prompting_experiments(config_path: str) -> None:
    config = load_json_config(config_path)

    llm = MockLLM()

    condition = config["condition"]
    rounds = config["rounds"]
    output_dir = config["output_dir"]

    for topic_config in config["topics"]:
        topic_name = topic_config["name"]
        topic_question = topic_config["question"]

        for stance_pair in config["stance_pairs"]:
            test_stance = stance_pair["test_agent_stance"]
            adversary_stance = stance_pair["adversary_stance"]

            for seed in config["seeds"]:
                test_agent = PromptAgent(
                    name=f"{condition}_test_agent",
                    stance=test_stance,
                    llm=llm,
                )

                adversary_agent = PromptAgent(
                    name="fixed_prompting_adversary",
                    stance=adversary_stance,
                    llm=llm,
                )

                environment = DebateEnvironment(
                    topic=topic_question,
                    test_agent=test_agent,
                    adversary_agent=adversary_agent,
                    rounds=rounds,
                    condition=condition,
                    seed=seed,
                    output_dir=output_dir,
                )

                transcript = environment.run()

                print(
                    f"Finished debate: {transcript['debate_id']} "
                    f"topic={topic_name} "
                    f"test_stance={test_stance} "
                    f"adversary_stance={adversary_stance} "
                    f"seed={seed}"
                )


def main():
    config_path = "configs/prompting_debug.json"
    run_prompting_experiments(config_path)


if __name__ == "__main__":
    main()
