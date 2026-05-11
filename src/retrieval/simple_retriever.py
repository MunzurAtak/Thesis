import json
from pathlib import Path


class SimpleStanceRetriever:
    """
    Simple first-version retriever for the RAG condition.

    This intentionally does not use embeddings yet.
    It filters passages by:
    - topic_name
    - stance

    Later this can be replaced by a sentence-transformer + FAISS retriever
    without changing the RAGAgent interface too much.
    """

    def __init__(self, corpus_path: str, top_k: int = 3):
        self.corpus_path = Path(corpus_path)
        self.top_k = top_k
        self.corpus = self._load_corpus()

    def _load_corpus(self) -> list[dict]:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"RAG corpus not found: {self.corpus_path}")

        with self.corpus_path.open("r", encoding="utf-8") as f:
            corpus = json.load(f)

        if not isinstance(corpus, list):
            raise ValueError("RAG corpus must be a list of passage objects.")

        required_keys = {"topic_name", "stance", "text"}

        for i, item in enumerate(corpus):
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                raise ValueError(
                    f"RAG corpus item {i} is missing keys: {sorted(missing_keys)}"
                )

            if item["stance"] not in {"pro", "contra"}:
                raise ValueError(
                    f"RAG corpus item {i} has invalid stance: {item['stance']}"
                )

        return corpus

    def retrieve(self, topic_name: str, stance: str) -> list[dict]:
        if stance not in {"pro", "contra"}:
            raise ValueError(f"Invalid stance: {stance}")

        matches = [
            item
            for item in self.corpus
            if item["topic_name"] == topic_name and item["stance"] == stance
        ]

        return matches[: self.top_k]

    def format_passages(self, passages: list[dict]) -> str:
        if not passages:
            return "No stance-consistent passages were retrieved."

        lines = []

        for i, passage in enumerate(passages, start=1):
            lines.append(f"[Retrieved passage {i}]\n{passage['text']}")

        return "\n\n".join(lines)
