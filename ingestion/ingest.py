"""
ingestion/ingest.py
────────────────────────────────────────────────────────────────
Main ingestion pipeline. Loads scraped GitHub Docs JSON files
into ChromaDB using LOCAL sentence-transformers embeddings.

NOTE ON EMBEDDINGS:
    The course endpoint only supports chat completions, not the
    embeddings API. We use sentence-transformers locally for
    ingestion (free, no API needed). The same model is used
    at query time so vectors are compatible.

HOW TO RUN:
    python -m ingestion.ingest
────────────────────────────────────────────────────────────────
"""

import sys
import json
import time
import warnings
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from configs.settings import load_settings, setup_logging

warnings.filterwarnings("ignore", category=Warning)

BATCH_SIZE = 50


def load_github_docs(github_docs_dir: Path) -> List[dict]:
    """Load all scraped JSON files → flat list of chunk dicts."""
    if not github_docs_dir.exists():
        logger.warning(f"No github_docs directory at {github_docs_dir}")
        logger.warning("Run ingestion/scraper.py first.")
        return []

    json_files = list(github_docs_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files in {github_docs_dir}")
        return []

    logger.info(f"Found {len(json_files)} JSON files")
    chunks = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                doc = json.load(f)

            doc_id   = doc.get("doc_id", json_file.stem)
            url      = doc.get("url", "")
            title    = doc.get("title", "")
            category = doc.get("category", "General")

            for i, qa in enumerate(doc.get("qa_pairs", [])):
                text     = f"Q: {qa['question']}\nA: {qa['answer']}"
                chunk_id = f"{doc_id}-{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text":     text,
                    "metadata": {
                        "chunk_id":    chunk_id,
                        "source_file": url,
                        "title":       title,
                        "category":    category,
                        "page_number": 0,
                        "chunk_index": i,
                        "token_count": len(text) // 4,
                    }
                })
        except Exception as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")

    logger.success(f"Loaded {len(chunks)} chunks from GitHub Docs")
    return chunks


def run_ingestion(data_dir: Path, settings) -> dict:
    """Main ingestion: load docs → embed locally → store in ChromaDB."""

    chunks = load_github_docs(data_dir / "raw" / "github_docs")
    if not chunks:
        return {"status": "error", "message": "No chunks found"}

    # Local embedding model — fast, free, no API needed
    logger.info("Loading local embedding model (sentence-transformers)...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.success("Embedding model loaded")

    # ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.success(
        f"ChromaDB ready | collection='{settings.chroma_collection_name}' | "
        f"existing={collection.count()} chunks"
    )

    # Embed and store in batches
    logger.info(f"Embedding {len(chunks)} chunks...")
    start  = time.time()
    stored = 0

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding", unit="batch"):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        stored += len(batch)

    elapsed     = time.time() - start
    final_count = collection.count()

    logger.info("=" * 50)
    logger.success("INGESTION COMPLETE")
    logger.info(f"  Chunks stored  : {stored}")
    logger.info(f"  ChromaDB total : {final_count}")
    logger.info(f"  Time           : {elapsed:.1f}s")
    logger.info("=" * 50)

    return {"status": "success", "chunks": stored, "total_in_db": final_count}


def main():
    settings = load_settings()
    setup_logging(settings)

    project_root = Path(__file__).parent.parent
    data_dir     = project_root / "data"

    (data_dir / "raw" / "github_docs").mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    result = run_ingestion(data_dir, settings)
    if result["status"] == "error":
        logger.error(result["message"])
        sys.exit(1)


if __name__ == "__main__":
    main()
