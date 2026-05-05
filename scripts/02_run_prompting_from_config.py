import argparse

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
            test_stance_score = stance_pair["test_agent_stance_score"]
            adversary_stance = stance_pair["adversary_stance"]
            adversary_stance_score = stance_pair["adversary_stance_score"]

            for seed in config["seeds"]:
                test_agent = PromptAgent(
                    name=f"{condition}_test_agent",
                    stance=test_stance,
                    stance_score=test_stance_score,
                    llm=llm,
                )

                adversary_agent = PromptAgent(
                    name="fixed_prompting_adversary",
                    stance=adversary_stance,
                    stance_score=adversary_stance_score,
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
                    experiment_name=config["experiment_name"],
                    topic_name=topic_name,
                )

                transcript = environment.run()

                print(
                    f"Finished debate: {transcript['debate_id']} "
                    f"topic={topic_name} "
                    f"test_stance={test_stance} "
                    f"adversary_stance={adversary_stance} "
                    f"seed={seed}"
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run prompting-only debate experiments from a JSON config."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/prompting_debug.json",
        help="Path to the experiment config JSON file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_prompting_experiments(args.config)


if __name__ == "__main__":
    main()
