from src.retrieval.simple_retriever import SimpleStanceRetriever
from src.retrieval.faiss_retriever import FAISSRetriever


def create_retriever(retrieval_config: dict):
    """
    Create a retriever from a retrieval config.

    Supported backends:
    - simple: filters by topic_name and stance, optionally lexical-ranks by query
    - faiss: semantic retrieval using sentence-transformers + FAISS
    """
    backend = retrieval_config.get("backend", "simple")

    if backend == "simple":
        return SimpleStanceRetriever(
            corpus_path=retrieval_config["corpus_path"],
            top_k=retrieval_config.get("top_k", 3),
        )

    if backend == "faiss":
        return FAISSRetriever(
            index_path=retrieval_config["index_path"],
            metadata_path=retrieval_config["metadata_path"],
            embedding_model=retrieval_config.get(
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            top_k=retrieval_config.get("top_k", 3),
            search_k=retrieval_config.get("search_k", 50),
        )

    raise ValueError(f"Unsupported retrieval backend: {backend}")
