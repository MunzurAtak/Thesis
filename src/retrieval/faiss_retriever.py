import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """
    FAISS-based stance-anchored retriever.

    It embeds the query, searches a FAISS index, then filters retrieved
    candidates by topic_name and stance.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        search_k: int = 50,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.search_k = search_k

        self.index = self._load_index()
        self.metadata = self._load_metadata()
        self.passages = self.metadata["passages"]
        self.model = SentenceTransformer(self.embedding_model)

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        return faiss.read_index(str(self.index_path))

    def _load_metadata(self) -> dict:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "passages" not in metadata:
            raise ValueError("FAISS metadata must contain a 'passages' field.")

        if metadata.get("embedding_model") != self.embedding_model:
            raise ValueError(
                "Embedding model mismatch. "
                f"Index was built with {metadata.get('embedding_model')}, "
                f"but retriever was configured with {self.embedding_model}."
            )

        return metadata

    def retrieve(
        self,
        topic_name: str,
        stance: str,
        query: str | None = None,
    ) -> list[dict]:
        if stance not in {"pro", "contra"}:
            raise ValueError(f"Invalid stance: {stance}")

        if not query:
            query = topic_name

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            min(self.search_k, len(self.passages)),
        )

        results = []

        for index in indices[0]:
            if index < 0:
                continue

            passage = self.passages[int(index)]

            if passage["topic_name"] != topic_name:
                continue

            if passage["stance"] != stance:
                continue

            results.append(passage)

            if len(results) >= self.top_k:
                break

        return results

    def format_passages(self, passages: list[dict]) -> str:
        if not passages:
            return "No stance-consistent passages were retrieved."

        lines = []

        for i, passage in enumerate(passages, start=1):
            lines.append(f"[Retrieved passage {i}]\n{passage['text']}")

        return "\n\n".join(lines)
