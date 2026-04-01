"""
retrieval/retriever_test.py
────────────────────────────────────────────────────────────────
Quick sanity check — run this after ingestion to verify that
retrieval is actually finding relevant chunks from your TD PDF.

HOW TO RUN:
    python -m retrieval.retriever_test

WHAT IT DOES:
    Runs 5 real financial questions against your TD Annual Report
    and prints the top 3 results for each — so you can visually
    verify the system is finding the right content.

WHY THIS MATTERS:
    If retrieval is broken, generation will always give bad answers.
    "Garbage in, garbage out." Checking retrieval first saves hours
    of debugging the wrong layer.
────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import load_settings, setup_logging
from retrieval.vector_retriever import VectorRetriever


# ── Test questions about TD Annual Report ─────────────────────────
# These cover different types of financial queries — a mix of
# specific numbers, regulatory topics, and qualitative questions.
# Good retrieval should find relevant chunks for all of them.

TEST_QUERIES = [
    "What was TD Bank's net income in 2025?",
    "What is TD's CET1 capital ratio?",
    "How does TD manage credit risk?",
    "What is TD's dividend policy?",
    "How did interest rate changes affect TD's performance?",
]


def run_retrieval_test():
    settings = load_settings()
    setup_logging(settings)

    print("\n" + "═" * 60)
    print("  FinSight-RAG — Retrieval Test")
    print("  Running against: TD Annual Report")
    print("═" * 60)

    # Initialize retriever
    retriever = VectorRetriever(settings)

    for query in TEST_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"QUERY: {query}")
        print(f"{'─' * 60}")

        results = retriever.search(query, top_k=3)

        if not results:
            print("  ⚠️  No results found")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n  Result #{i}")
            print(f"  {result.citation()}")
            print(f"  Similarity Score : {result.similarity_score:.4f}")
            print(f"  Token Count      : {result.token_count}")
            print(f"  Preview          : {result.text[:200].strip()}...")

    print("\n" + "═" * 60)
    print("  Retrieval test complete.")
    print("  If scores are above 0.4, retrieval is working well.")
    print("  If scores are below 0.2, check that ingestion ran correctly.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_retrieval_test()
