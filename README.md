# 🏦 FinSight-RAG

**Production-grade Retrieval-Augmented Generation system for Canadian financial documents.**

Answers questions over Bank of Canada reports, OSFI guidelines, and bank regulatory filings — with citations, hybrid retrieval, cross-encoder reranking, and automated evaluation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FinSight-RAG                         │
├──────────────┬──────────────────────────┬───────────────────┤
│   INGESTION  │       RETRIEVAL          │    GENERATION     │
│              │                          │                   │
│  PDF → Chunks│  Query                   │  Chunks + Prompt  │
│  Chunks →    │    ↓                     │       ↓           │
│  Embeddings  │  BM25 ──┐               │   GPT-3.5-turbo   │
│  Embeddings →│         ├─ RRF Fusion   │       ↓           │
│  ChromaDB    │  Vector─┘       ↓        │  Answer + Cites   │
│              │           Reranker       │       ↓           │
│              │               ↓          │  Citation Check   │
│              │           Top-5 Chunks   │                   │
└──────────────┴──────────────────────────┴───────────────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │      EVALUATION         │
                    │  RAGAs: Faithfulness,   │
                    │  Relevancy, Recall      │
                    │  GitHub Actions CI/CD   │
                    └─────────────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| BM25 Search | rank-bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | OpenAI GPT-3.5-turbo |
| Evaluation | RAGAs |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/yourusername/finsight-rag
cd finsight-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
make setup

# 4. Add your OpenAI API key to .env
#    (file was auto-created by make setup)
nano .env

# 5. Add PDFs to data/raw/
#    (Bank of Canada reports, OSFI guidelines, etc.)

# 6. Ingest documents
make ingest

# 7. Run evaluation
make evaluate

# 8. Launch dashboard
make dashboard
```

## Document Sources (Free & Public)

- [Bank of Canada Financial Stability Report](https://www.bankofcanada.ca/publications/fsr/)
- [OSFI Guidelines](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library)
- [TD Bank Annual Report](https://www.td.com/ca/en/about-td/investor-relations/financial-information/annual-reports)

## Evaluation Results

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | — | ≥ 0.75 |
| Answer Relevancy | — | ≥ 0.70 |
| Context Recall | — | ≥ 0.65 |

*(Scores populated after running `make evaluate`)*

## Project Structure

```
finsight-rag/
├── data/raw/              ← Input PDFs (not committed to git)
├── data/processed/        ← Chunked text cache
├── ingestion/             ← PDF parsing + embedding pipeline
├── retrieval/             ← Hybrid BM25 + vector + reranker
├── generation/            ← RAG chain + citation enforcement
├── evaluation/            ← RAGAs metrics + golden dataset
├── dashboard/             ← Streamlit UI
├── configs/
│   ├── prompts.yaml       ← Version-controlled prompt templates
│   └── settings.py        ← Central config loader
├── tests/                 ← Unit + integration tests
├── .github/workflows/     ← CI/CD pipelines
└── Makefile               ← Developer shortcuts
```

## Why This Matters for Financial Institutions

1. **Citation enforcement** — Every answer traces back to a source document and page. This satisfies audit requirements and reduces hallucination risk in regulated environments.
2. **Hybrid retrieval** — Combines semantic search with keyword matching, critical when querying documents with precise regulatory terminology (e.g., "Guideline B-20").
3. **Automated evaluation** — Faithfulness metrics ensure the system doesn't degrade silently. CI/CD gates prevent regressions from reaching production.
4. **Prompt versioning** — All prompts are in `configs/prompts.yaml` under version control. Any change is traceable, satisfying change-management requirements common in bank IT governance.
