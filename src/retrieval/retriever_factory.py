from src.retrieval.simple_retriever import SimpleStanceRetriever


def create_retriever(retrieval_config: dict):
    """
    Create a retriever from a retrieval config.

    Supported backends:
    - simple: filters by topic_name and stance
    """
    backend = retrieval_config.get("backend", "simple")

    if backend == "simple":
        return SimpleStanceRetriever(
            corpus_path=retrieval_config["corpus_path"],
            top_k=retrieval_config.get("top_k", 3),
        )

    raise ValueError(f"Unsupported retrieval backend: {backend}")
