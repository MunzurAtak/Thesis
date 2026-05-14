import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index for a RAG corpus JSON file."
    )

    parser.add_argument(
        "--corpus-path",
        type=str,
        default="data/rag_corpus/usdc_selected_rag_corpus.json",
        help="Path to RAG corpus JSON.",
    )

    parser.add_argument(
        "--index-path",
        type=str,
        default="data/rag_indexes/usdc_selected.faiss",
        help="Path where the FAISS index will be saved.",
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/rag_indexes/usdc_selected_metadata.json",
        help="Path where passage metadata will be saved.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )

    return parser.parse_args()


def load_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a non-empty list of passage objects.")

    required_keys = {"topic_name", "stance", "text"}

    for i, item in enumerate(corpus):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Corpus item {i} missing keys: {sorted(missing)}")

    return corpus


def main():
    args = parse_args()

    corpus_path = Path(args.corpus_path)
    index_path = Path(args.index_path)
    metadata_path = Path(args.metadata_path)

    print(f"Loading corpus: {corpus_path}")
    corpus = load_corpus(corpus_path)

    texts = [item["text"] for item in corpus]

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    print(f"Embedding {len(texts)} passages...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    print(f"Embedding shape: {embeddings.shape}")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    metadata = {
        "embedding_model": args.embedding_model,
        "corpus_path": str(corpus_path),
        "num_passages": len(corpus),
        "embedding_dimension": int(dimension),
        "passages": corpus,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f}s")
