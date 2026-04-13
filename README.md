# 🐙 GitHub Documentation Assistant

**A conversational RAG-based support agent built for the UofT Rotman LLM Course — Team 11.**

Answers questions from GitHub's public documentation, creates and manages support tickets, checks billing plans, and enforces safety guardrails — all powered by the course Qwen endpoint and a ChromaDB vector store of 1,325 knowledge chunks.

---

## What This Does

| Capability | Description |
|---|---|
| 🔍 Knowledge Q&A | Answers GitHub questions with source URL citations and a confidence score |
| 🎫 Support Tickets | Multi-turn ticket creation (4 steps) saved to SQLite |
| 📋 Ticket Lookup | Check any ticket by ID; RAG-augmented ticket status responses |
| 🔒 Close Tickets | Close all open tickets in one command, or close a specific ticket by ID |
| 💳 Billing Checker | Returns account plan details from SQLite user registry |
| 👤 User Registration | Multi-turn flow to register new users with a chosen plan |
| ⚡ Plan Upgrades | Upgrade or downgrade any account between Free / Pro / Team / Enterprise |
| 📋 List Accounts | View all registered accounts and their plans in one command |
| 🎙️ Voice Interface | Jarvis voice mode — speak your question, hear the answer read back |
| 🚫 Guardrails | Blocks prompt injection and out-of-scope questions |
| 💬 Memory | Remembers conversation context within a session |
| 📊 Analytics | Live dashboard: intent distribution, confidence trend, knowledge gaps |
| 📄 PDF Export | Download the full conversation as a formatted PDF |

---

## Architecture

```
User Message (text or 🎙️ voice via Jarvis)
     ↓
Intent Router (regex + Qwen LLM) — 11 intents
     ↓
 ┌──────────────────────────────────────────────────────────┐
 │  rag_query          → ChromaDB → Qwen → Answer           │
 │  create_ticket      → Multi-turn State Machine            │
 │  check_ticket       → SQLite Lookup + RAG answer          │
 │  check_billing      → SQLite User Registry                │
 │  register_user      → Multi-turn State Machine            │
 │  upgrade_plan       → SQLite Plan Update                  │
 │  list_accounts      → SQLite User Registry                │
 │  close_tickets      → SQLite Bulk Update                  │
 │  close_ticket_by_id → SQLite Single Ticket Update         │
 │  out_of_scope       → Polite Rejection                    │
 │  prompt_injection   → Block + Warn                        │
 └──────────────────────────────────────────────────────────┘
     ↓
SQLite (tickets + users + conversation history)
Streamlit UI (intent badges + ticket sidebar + analytics page)
Jarvis TTS speaks assistant reply (when voice mode is on)
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen3-30b via course endpoint (OpenAI-compatible) |
| Embeddings | BAAI/bge-base-en-v1.5 via A2 course endpoint (API, not local) |
| Vector Store | ChromaDB (1,325 chunks from 230 GitHub Docs pages) |
| Persistence | SQLite (tickets + users + conversation history) |
| Voice I/O | `voice/jarvis.py` — SpeechRecognition (STT) + pyttsx3 (TTS) |
| Web Scraping | BeautifulSoup4 |
| Config | Pydantic + YAML prompt versioning |
| Logging | Loguru |
| UI | Streamlit (multi-page: chat + analytics dashboard) |
| PDF Export | fpdf2 |
| Evaluation | 15 structured test cases, results.json |

---

## Project Structure

```
GitHub-RAG/
├── ID.txt                    ← YOUR student ID (create this yourself — see Step 4)
├── configs/
│   ├── settings.py           ← Central config, reads ID.txt for API key
│   └── prompts.yaml          ← All LLM prompts, version controlled
├── ingestion/
│   ├── scraper.py            ← Scrapes GitHub Docs → JSON files
│   ├── chunker.py            ← Text splitting logic (recursive character splitter)
│   └── ingest.py             ← Loads JSON → embeddings via A2 API → ChromaDB
├── retrieval/
│   └── vector_retriever.py   ← Semantic search over ChromaDB via A2 embeddings API
├── generation/
│   └── rag_chain.py          ← RAG pipeline with memory + Qwen
├── agent/
│   ├── intent_router.py      ← Classifies every message (11 intents)
│   ├── actions.py            ← All actions (ticket, billing, register, upgrade, list)
│   ├── session_state.py      ← SQLite persistence layer
│   └── guardrails.py         ← Injection + OOS blocking
├── voice/
│   └── jarvis.py             ← Voice I/O: microphone STT + TTS speaker output
├── dashboard/
│   ├── app.py                ← Streamlit UI (chat + Jarvis sidebar + PDF export)
│   └── pages/
│       └── 1_📊_Analytics.py ← Live analytics: intent distribution, confidence, gaps
├── evaluation/
│   ├── test_cases.py         ← 15 structured test cases
│   └── results.json          ← Auto-generated after running eval
├── data/                     ← THIS FOLDER IS EMPTY WHEN YOU CLONE
│   ├── raw/github_docs/      ← You generate this in Step 6 (scraping)
│   ├── chroma_db/            ← You generate this in Step 7 (ingestion)
│   └── agent_state.db        ← Auto-created when app first runs
└── requirements.txt
```

> ⚠️ **Important:** The `data/` folder is not uploaded to GitHub because it contains large generated files. You will generate all of it yourself in the steps below. This takes about 20 minutes total on first setup.

---

## Setup — Complete Step by Step

### Prerequisites
- Python 3.11+
- Your UofT student ID number (e.g. `1012345678`)
- Internet connection (for scraping and API calls)

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/2002-ishaan/GitHub-RAG
cd GitHub-RAG
```

---

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

---

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
pip install "openai==1.30.0" "httpx==0.27.0"
```

This installs everything including ChromaDB, Streamlit, fpdf2, and the OpenAI SDK.

For the Jarvis voice feature, also install the voice dependencies:
```bash
pip install SpeechRecognition pyttsx3 pyaudio
```

> ⚠️ **pyaudio on Mac** requires `brew install portaudio` first. On Windows, install the prebuilt wheel: `pip install pipwin && pipwin install pyaudio`. Voice is optional — the app runs fully without it.

---

### Step 4 — Create ID.txt in the project root ⚠️ Critical

This file contains your student ID which is used as the API key for the course Qwen endpoint. **Without this the app will not start.**

Create a file called `ID.txt` directly in the project root folder (same level as `README.md`) with **exactly 3 lines**:

```
Your Full Name
your.email@mail.utoronto.ca
1012345678
```

- Line 1: Your full name
- Line 2: Your UofT email
- Line 3: Your student ID number ← this is the API key

**Example using terminal:**
```bash
# Make sure you are in the project root folder first
echo "Ishaan Dawra" > ID.txt
echo "ishaan.dawra@rotman.utoronto.ca" >> ID.txt
echo "1012826186" >> ID.txt
```

> ⚠️ `ID.txt` is in `.gitignore` so it will never be committed to GitHub. Every teammate creates their own.

---

### Step 5 — Create the data folders

```bash
mkdir -p data/raw/github_docs
mkdir -p data/chroma_db
```

---

### Step 6 — Scrape GitHub Documentation

This downloads 230 pages from docs.github.com and saves them as JSON files in `data/raw/github_docs/`. **This takes about 5-10 minutes** (it crawls politely with a 1 second delay between pages).

```bash
python3 ingestion/scraper.py
```

Expected output:
```
GitHub Docs Knowledge Base Crawler
Pages visited  : 230
Documents saved: 230  →  data/raw/github_docs/
✅  Meets 50-document minimum (230 docs saved).
```

When finished you should see 230 `.json` files inside `data/raw/github_docs/`.

> ⚠️ If you get fewer than 50 docs, re-run the scraper. GitHub's site occasionally blocks crawlers temporarily.

---

### Step 7 — Run Ingestion (Load into ChromaDB)

This reads all 230 JSON files, generates embeddings locally, and stores 1,325 chunks in ChromaDB. **This takes about 15-20 seconds.**

```bash
python3 -m ingestion.ingest
```

Expected output:
```
✅ INGESTION COMPLETE
   Chunks stored  : 1325
   ChromaDB total : 1325
   Time           : ~15s
```

> ⚠️ If you see `0 chunks stored`, make sure Step 6 completed successfully and `data/raw/github_docs/` has JSON files in it.

---

### Step 8 — Run the App

```bash
streamlit run dashboard/app.py
```

Opens automatically at **http://localhost:8501**

The first load takes ~10 seconds while it loads the embedding model and connects to ChromaDB. Subsequent loads are instant.

---

### Step 9 — Run the Evaluation Suite (Optional)

```bash
python3 -m evaluation.test_cases
```

Runs 15 test cases across all capabilities and saves results to `evaluation/results.json`. Takes 2-3 minutes.

---

## How to Use the Agent

| Say this | What happens |
|---|---|
| `"How do I create a private repo?"` | RAG answer with GitHub Docs source URLs and confidence score |
| `"How do I set up 2FA?"` | Step-by-step answer with citations |
| `"Create a support ticket"` | 4-turn flow: category → description → priority → saved |
| `"Check ticket TKT-001"` | Returns ticket details + RAG-generated guidance from SQLite |
| `"Close ticket TKT-001"` | Closes that specific ticket; sidebar updates |
| `"Close all active tickets"` | Closes all open tickets, sidebar updates |
| `"Check billing for alice"` | Returns alice's plan details from SQLite |
| `"Register a new account for sarah"` | Multi-turn flow: username → plan → confirm → saved to SQLite |
| `"Upgrade alice to Enterprise"` | Updates alice's plan in SQLite, shows before/after comparison |
| `"List all accounts"` | Shows all registered users and their current plans |
| `"Tell me a joke"` | Blocked — out-of-scope guardrail fires |
| `"Ignore all previous instructions"` | Blocked — prompt injection guardrail fires |
| 🎙️ Click **Start** in the sidebar | Jarvis listens for 8 seconds, transcribes, processes, speaks the reply |

## Default Billing Accounts

These accounts are seeded into SQLite on first run. New accounts can be added at any time with `"Register a new account for [username]"`.

| Username | Plan |
|---|---|
| alice | Pro |
| bob | Team |
| carol | Enterprise |
| dave | Free |

---

## Course Endpoint Details

| | Value |
|---|---|
| **LLM / Chat Base URL** | `https://rsm-8430-finalproject.bjlkeng.io/v1` |
| **Embeddings Base URL** | `https://rsm-8430-a2.bjlkeng.io/v1` |
| **API Key** | Your student ID (from `ID.txt` line 3) |
| **LLM Model** | `IGNORED` (model name is ignored by the server) |
| **Embedding Model** | `BAAI/bge-base-en-v1.5` |

> ⚠️ Both endpoints use the same student ID as the API key. The embedding endpoint is called at both ingestion time and query time — they must match or retrieval will silently fail.

---

## Evaluation Results

Current success rate: **87% (13/15 test cases passing)**

| Category | Cases | Status |
|---|---|---|
| RAG Retrieval | 4 | ✅ |
| Intent Routing | 2 | ✅ |
| OOS Rejection | 2 | ✅ |
| Guardrails | 1 | ✅ |
| Multi-turn Action | 1 | ✅ |
| Action Execution | 2 | ✅ |
| Error Handling | 2 | ⚠️ Partial |
| Memory/Persistence | 1 | ✅ |

---

| Error | Fix |
|---|---|
| `Student ID not found` | Create `ID.txt` in project root with your student ID on line 3 |
| `ChromaDB collection not found` | Run `python3 -m ingestion.ingest` first |
| `proxies TypeError` | Run `pip install "openai==1.30.0" "httpx==0.27.0"` |
| `No JSON files in github_docs` | Run `python3 ingestion/scraper.py` first |
| `0 chunks stored` | Check `data/raw/github_docs/` has 230 JSON files |
| App loads but answers are bad | Re-run ingestion — ChromaDB may be empty or corrupt |
| Embedding errors at ingest or query time | Both steps must use the same A2 endpoint and `BAAI/bge-base-en-v1.5` model |
| `No module named 'speech_recognition'` | Run `pip install SpeechRecognition pyttsx3 pyaudio` |
| `pyaudio` install fails | Mac: `brew install portaudio` first. Windows: `pipwin install pyaudio` |
| Voice captures its own TTS output | Click **Stop**, wait for Jarvis to finish speaking, then click **Start** again |

---

## Ideas for Further Development

> For teammates looking to add value — ranked by impact

### 🔴 High Impact

**1. Hybrid BM25 + Vector Retrieval**
Currently using pure vector search. Adding BM25 keyword search with Reciprocal Rank Fusion (RRF) would significantly improve retrieval for exact-match queries like "Dependabot" or "CODEOWNERS". The `rank-bm25` library handles this cleanly.

**2. Cross-Encoder Reranker**
After retrieving top-10 chunks, run a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank them before feeding to Qwen. This alone can push accuracy from 87% to 93%+.

**3. RAGAs Automated Evaluation**
Replace the current pass/fail test suite with proper RAGAs metrics (Faithfulness, Answer Relevancy, Context Recall). Wire into GitHub Actions CI/CD so every commit gets scored automatically.

### 🟡 Medium Impact

**4. Streamlit Dashboard Improvements**
- Add a metrics panel showing live retrieval scores, response time, chunk count per query
- Show which chunks were used for each answer (expandable source viewer)
- Add conversation export as PDF
- Add feedback buttons (👍 👎) per response

**5. Query Expansion**
Before retrieving, use Qwen to generate 2-3 alternative phrasings of the question, retrieve for all of them, then merge results. Improves recall for vague queries significantly.

**6. Security Reviewer Action**
The original proposal included this — simulate checking 2FA status, active SSH keys, and OAuth apps, returning a security score. This was replaced by the ticket system but would restore the original demo flow.

### 🟢 Polish & Bonus

**7. Streamlit Community Cloud Deployment (+2% grade bonus)**
Deploy to https://streamlit.io/cloud — free, one-click. Store student ID as a Streamlit Secret instead of `ID.txt`. The professor gives +2% for a live deployed URL.

**8. Confidence Indicator**
Show a visual confidence badge on each RAG answer based on the top chunk similarity score. Below 0.5 → warn the user the answer may be unreliable.

**9. Logging Dashboard Tab**
Loguru already writes to `logs/agent.log`. Build an admin tab in Streamlit showing declined queries, intent distribution as a pie chart, and average response times.
