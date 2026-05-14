import json
import re
from pathlib import Path


class SimpleStanceRetriever:
    """
    Simple first-version retriever for the RAG condition.

    It filters passages by:
    - topic_name
    - stance

    Then it optionally ranks passages by lexical overlap with a query.
    This is still not embedding retrieval, but it is safer than returning
    the first top-k passages from a noisy corpus.
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

    def retrieve(
        self,
        topic_name: str,
        stance: str,
        query: str | None = None,
    ) -> list[dict]:
        if stance not in {"pro", "contra"}:
            raise ValueError(f"Invalid stance: {stance}")

        matches = [
            item
            for item in self.corpus
            if item["topic_name"] == topic_name and item["stance"] == stance
        ]

        if query:
            matches = sorted(
                matches,
                key=lambda item: self._lexical_score(query=query, text=item["text"]),
                reverse=True,
            )

        return matches[: self.top_k]

    def format_passages(self, passages: list[dict]) -> str:
        if not passages:
            return "No additional private background is available."

        lines = []

        for passage in passages:
            lines.append(f"- {passage['text']}")

        return "\n\n".join(lines)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "that",
            "this",
            "it",
            "as",
            "at",
            "from",
            "not",
            "do",
            "does",
            "did",
            "should",
            "would",
            "could",
            "can",
            "we",
            "you",
            "they",
            "their",
            "our",
            "your",
        }

        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {token for token in tokens if token not in stopwords}

    @classmethod
    def _lexical_score(cls, query: str, text: str) -> int:
        query_tokens = cls._tokenize(query)
        text_tokens = cls._tokenize(text)

        return len(query_tokens & text_tokens)
