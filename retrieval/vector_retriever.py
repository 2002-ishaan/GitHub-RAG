"""
retrieval/vector_retriever.py
────────────────────────────────────────────────────────────────
Semantic search over ChromaDB using local sentence-transformers.

NOTE: Must use the same embedding model as ingestion.
      Both use sentence-transformers/all-MiniLM-L6-v2 locally.
      The course endpoint is only used for LLM generation.
────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from loguru import logger


@dataclass
class SearchResult:
    """One retrieved chunk with everything needed for citation and context."""
    chunk_id:         str
    text:             str
    source_file:      str
    page_number:      int
    similarity_score: float
    chunk_index:      int
    token_count:      int

    def citation(self) -> str:
        """Human-readable citation string."""
        if self.source_file.startswith("http"):
            return f"[Source: {self.source_file}]"
        return f"[Source: {self.source_file}]"

    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.similarity_score:.3f}, "
            f"preview='{self.text[:60].strip()}...')"
        )


class VectorRetriever:
    """
    Semantic search over ChromaDB using local sentence-transformers.

    Usage:
        retriever = VectorRetriever(settings)
        results = retriever.search("How do I create a private repo?", top_k=5)
    """

    def __init__(self, settings):
        logger.info("Initialising VectorRetriever")

        # Local embedding model — same as used in ingestion
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.top_k_retrieval = settings.top_k_retrieval

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.chroma_client.get_collection(
                name=settings.chroma_collection_name
            )
            logger.success(
                f"VectorRetriever ready | "
                f"collection='{settings.chroma_collection_name}' | "
                f"{self.collection.count()} chunks indexed"
            )
        except Exception as e:
            logger.error(
                f"ChromaDB collection '{settings.chroma_collection_name}' not found.\n"
                f"Run: python -m ingestion.ingest\n"
                f"Error: {e}"
            )
            raise

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Find the most semantically similar chunks to a query."""
        if not query or not query.strip():
            return []

        k = top_k or self.top_k_retrieval

        # Embed query locally
        query_embedding = self.embedding_model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Query ChromaDB
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        results = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = round(1 - (dist / 2), 4)
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", "unknown"),
                text=doc,
                source_file=meta.get("source_file", "unknown"),
                page_number=int(meta.get("page_number", 0)),
                similarity_score=similarity,
                chunk_index=int(meta.get("chunk_index", 0)),
                token_count=int(meta.get("token_count", 0)),
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)

        logger.info(
            f"Query: '{query[:60]}' | "
            f"{len(results)} chunks | "
            f"top score: {results[0].similarity_score if results else 'N/A'}"
        )

        return results
