"""
generation/rag_test.py
────────────────────────────────────────────────────────────────
End-to-end test of the full RAG pipeline.

HOW TO RUN:
    python -m generation.rag_test

WHAT THIS TESTS:
    The complete chain for 4 real questions:
    Question → Retrieve → Generate → Citation check → Display

    This is the first time we see actual AI-generated answers
    with citations pointing back to the TD Annual Report.
────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import load_settings, setup_logging
from generation.rag_chain import RAGChain


TEST_QUESTIONS = [
    "What was TD Bank's net income in fiscal 2025?",
    "What is TD's CET1 capital ratio and how does it compare to OSFI requirements?",
    "How does TD manage credit risk for retail borrowers?",
    "What dividend did TD declare for Q1 2026?",
]


def run_generation_test():
    settings = load_settings()
    setup_logging(settings)

    print("\n" + "═" * 65)
    print("  FinSight-RAG — Full Pipeline Test")
    print("  Retrieve + Generate with Mistral 7B")
    print("═" * 65)

    # Initialize the full RAG chain
    chain = RAGChain(settings)

    for question in TEST_QUESTIONS:
        print(f"\n{'─' * 65}")
        print(f"QUESTION: {question}")
        print(f"{'─' * 65}")

        response = chain.ask(question)

        print(f"\n{response.formatted_answer()}")
        print(f"\nChunks retrieved : {response.metadata.get('chunks_retrieved')}")
        print(f"Chunks used      : {response.metadata.get('chunks_used')}")
        print(f"Top similarity   : {response.metadata.get('top_similarity', 0):.4f}")

    print("\n" + "═" * 65)
    print("  Pipeline test complete.")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    run_generation_test()
