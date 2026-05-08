from src.judge.mock_judge import MockJudge
from src.judge.ollama_judge import OllamaJudge
from src.llms.llm_factory import create_llm


def create_judge(judge_config: dict):
    """
    Create a judge from a config dictionary.

    Supported judge backends:
    - mock
    - ollama
    """
    backend = judge_config.get("backend", "mock")

    if backend == "mock":
        return MockJudge()

    if backend == "ollama":
        llm = create_llm(judge_config)
        return OllamaJudge(llm=llm)

    raise ValueError(f"Unsupported judge backend: {backend}")


def judge_metadata(judge) -> dict:
    """
    Return metadata about the judge.
    """
    if hasattr(judge, "llm"):
        return judge.llm.metadata()

    return {
        "backend": "mock",
        "model_name": "mock",
    }
