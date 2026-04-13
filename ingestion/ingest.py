"""
ingestion/ingest.py
────────────────────────────────────────────────────────────────
Main ingestion pipeline. Loads scraped GitHub Docs JSON files
into ChromaDB using the course A2 embeddings endpoint.

NOTE ON EMBEDDINGS:
    Per professor's instructions, embeddings use the A2 API
    endpoint (https://rsm-8430-a2.bjlkeng.io/v1) with the
    BAAI/bge-base-en-v1.5 model via the OpenAI-compatible
    embeddings API. The same endpoint and model are used at
    query time so vectors are always compatible.

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
from openai import OpenAI

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


def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """Embed a batch of texts using the A2 OpenAI-compatible embeddings endpoint."""
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    # Returns embeddings in the same order as input
    return [item.embedding for item in response.data]


def run_ingestion(data_dir: Path, settings) -> dict:
    """Main ingestion: load docs → embed via A2 API → store in ChromaDB."""

    chunks = load_github_docs(data_dir / "raw" / "github_docs")
    if not chunks:
        return {"status": "error", "message": "No chunks found"}

    # A2 embeddings endpoint (per professor's instructions)
    logger.info(f"Connecting to embeddings endpoint: {settings.embedding_base_url}")
    embed_client = OpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.embedding_base_url,
    )
    logger.success(f"Embeddings client ready | model={settings.embedding_model}")

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
    logger.info(f"Embedding {len(chunks)} chunks via A2 API...")
    start  = time.time()
    stored = 0

    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding", unit="batch"):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = embed_texts(embed_client, texts, settings.embedding_model)

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
