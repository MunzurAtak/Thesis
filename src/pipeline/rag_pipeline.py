from src.agents.adversary_agent import AdversaryAgent
from src.agents.rag_agent import RAGAgent
from src.debate.environment import DebateEnvironment
from src.judge.judge_factory import create_judge
from src.llms.llm_factory import create_llm
from src.pipeline.prompting_pipeline import (
    clear_json_files,
    compute_metrics_directory,
    score_transcript_directory,
    validate_transcript_directory,
)
from src.retrieval.retriever_factory import create_retriever
from src.utils.config import load_json_config


def run_rag_experiments(
    config_path: str,
    output_dir_override: str | None = None,
) -> None:
    config = load_json_config(config_path)

    test_llm = create_llm(config["models"]["test_agent"])
    adversary_llm = create_llm(config["models"]["adversary_agent"])

    retrieval_config = config["retrieval"]
    retriever = create_retriever(retrieval_config)

    condition = config["condition"]
    rounds = config["rounds"]
    output_dir = output_dir_override or config["output_dir"]

    for topic_config in config["topics"]:
        topic_name = topic_config["name"]
        topic_question = topic_config["question"]

        for stance_pair in config["stance_pairs"]:
            test_stance = stance_pair["test_agent_stance"]
            test_stance_score = stance_pair["test_agent_stance_score"]
            adversary_stance = stance_pair["adversary_stance"]
            adversary_stance_score = stance_pair["adversary_stance_score"]

            for seed in config["seeds"]:
                test_agent = RAGAgent(
                    name=f"{condition}_test_agent",
                    stance=test_stance,
                    stance_score=test_stance_score,
                    llm=test_llm,
                    retriever=retriever,
                    topic_name=topic_name,
                )

                adversary_agent = AdversaryAgent(
                    name="fixed_prompting_adversary",
                    stance=adversary_stance,
                    stance_score=adversary_stance_score,
                    llm=adversary_llm,
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
                    f"Finished RAG debate: {transcript['debate_id']} "
                    f"topic={topic_name} "
                    f"test_stance={test_stance} "
                    f"adversary_stance={adversary_stance} "
                    f"seed={seed}"
                )


def run_full_rag_debug_pipeline(
    config_path: str = "configs/rag_mock_debug.json",
    transcript_dir: str = "outputs/transcripts",
    judge_score_dir: str = "outputs/judge_scores",
    metrics_dir: str = "outputs/metrics",
    metrics_output_path: str = "outputs/metrics/rag_mock_debug_metrics.csv",
) -> None:
    config = load_json_config(config_path)
    judge_config = config["models"]["judge"]

    print("Clearing old debug outputs...")
    clear_json_files(transcript_dir)
    clear_json_files(judge_score_dir)

    print("\nRunning RAG debates...")
    run_rag_experiments(
        config_path=config_path,
        output_dir_override=transcript_dir,
    )

    print("\nValidating transcripts...")
    validate_transcript_directory(transcript_dir)

    print(f"\nScoring transcripts with {judge_config['backend']} judge...")
    score_transcript_directory(
        input_dir=transcript_dir,
        output_dir=judge_score_dir,
        judge_config=judge_config,
    )

    print("\nComputing metrics...")
    compute_metrics_directory(
        input_dir=judge_score_dir,
        output_path=metrics_output_path,
    )

    print("\nRAG debug pipeline complete.")
    print(f"Metrics saved to: {metrics_output_path}")
