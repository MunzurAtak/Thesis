from src.llms.mock_llm import MockLLM
from src.llms.ollama_llm import OllamaLLM
from src.llms.base_llm import BaseLLM


def create_llm(llm_config: dict) -> BaseLLM:
    """
    Create an LLM wrapper from a config dictionary.

    Supported backends:
    - mock
    - ollama
    """
    backend = llm_config.get("backend", "mock")

    if backend == "mock":
        return MockLLM()

    if backend == "ollama":
        model_name = llm_config["model_name"]
        temperature = llm_config.get("temperature", 0.7)
        max_tokens = llm_config.get("max_tokens", 256)
        base_url = llm_config.get("base_url", "http://localhost:11434")

        return OllamaLLM(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM backend: {backend}")
