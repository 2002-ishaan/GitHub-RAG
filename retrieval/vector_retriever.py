"""
retrieval/vector_retriever.py
────────────────────────────────────────────────────────────────
Semantic search over ChromaDB using the course A2 embeddings API.

NOTE: Must use the same embedding endpoint + model as ingestion.
      Both use https://rsm-8430-a2.bjlkeng.io/v1 with
      BAAI/bge-base-en-v1.5 via the OpenAI-compatible API.
────────────────────────────────────────────────────────────────
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
from loguru import logger
from ingestion.ingest import run_ingestion
from ingestion.scraper import crawl


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
    Semantic search over ChromaDB using the course A2 embeddings API.

    Usage:
        retriever = VectorRetriever(settings)
        results = retriever.search("How do I create a private repo?", top_k=5)
    """

    def __init__(self, settings):
        logger.info("Initialising VectorRetriever")
        self.settings = settings

        # A2 embeddings client — same endpoint and model used in ingestion
        self.embed_client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.embedding_base_url,
        )
        self.embedding_model = settings.embedding_model
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
            logger.warning(
                f"Collection '{settings.chroma_collection_name}' missing. "
                "Attempting bootstrap ingestion."
            )
            if not self._bootstrap_collection():
                logger.error(
                    f"ChromaDB collection '{settings.chroma_collection_name}' not found.\n"
                    f"Run: python -m ingestion.ingest\n"
                    f"Error: {e}"
                )
                raise

            self.collection = self.chroma_client.get_collection(
                name=settings.chroma_collection_name
            )
            logger.success(
                f"VectorRetriever recovered | "
                f"collection='{settings.chroma_collection_name}' | "
                f"{self.collection.count()} chunks indexed"
            )

    def _bootstrap_collection(self) -> bool:
        """
        Build Chroma collection once if it is missing.

        This is useful on Streamlit Cloud where runtime storage starts empty.
        """
        auto_ingest = os.getenv("AUTO_INGEST_ON_MISSING_COLLECTION", "true").strip().lower()
        if auto_ingest not in {"1", "true", "yes", "y", "on"}:
            logger.info("AUTO_INGEST_ON_MISSING_COLLECTION is disabled.")
            return False

        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        raw_docs_dir = data_dir / "raw" / "github_docs"

        json_docs = list(raw_docs_dir.glob("*.json")) if raw_docs_dir.exists() else []
        if not json_docs:
            auto_scrape = os.getenv("AUTO_SCRAPE_ON_MISSING_DOCS", "true").strip().lower()
            if auto_scrape in {"1", "true", "yes", "y", "on"}:
                logger.warning("Raw docs are missing. Running scraper bootstrap once.")
                try:
                    bootstrap_max_pages = int(os.getenv("BOOTSTRAP_MAX_PAGES", "80"))
                    bootstrap_delay = float(os.getenv("BOOTSTRAP_CRAWL_DELAY_SECONDS", "0.2"))
                    crawl(max_pages=bootstrap_max_pages, delay_seconds=bootstrap_delay)
                    json_docs = list(raw_docs_dir.glob("*.json")) if raw_docs_dir.exists() else []
                except Exception as exc:
                    logger.error(f"Auto-scrape failed: {exc}")

            if not json_docs:
                logger.error(
                    "Cannot auto-ingest: no raw docs found at "
                    f"{raw_docs_dir}. Ensure data/raw/github_docs/*.json is available "
                    "or enable AUTO_SCRAPE_ON_MISSING_DOCS=true."
                )
                return False

        try:
            Path(self.settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
            result = run_ingestion(data_dir, self.settings)
            return result.get("status") == "success"
        except Exception as exc:
            logger.error(f"Auto-ingestion failed: {exc}")
            return False

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Find the most semantically similar chunks to a query."""
        if not query or not query.strip():
            return []

        k = top_k or self.top_k_retrieval

        # Embed query via A2 API
        response = self.embed_client.embeddings.create(
            model=self.embedding_model,
            input=query.strip(),
        )
        query_embedding = response.data[0].embedding

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
