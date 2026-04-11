"""
generation/rag_chain.py
────────────────────────────────────────────────────────────────
Generates grounded answers using the course Qwen endpoint.

CHANGES FROM ORIGINAL:
    - Switched from Groq SDK to OpenAI SDK (Qwen is OpenAI-compatible)
    - Added conversation memory (chat_history in prompt)
    - Updated prompts for GitHub support context
    - Source citation format updated for web URLs
────────────────────────────────────────────────────────────────
"""

import re
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from loguru import logger
from configs.settings import load_settings, load_prompts
from retrieval.vector_retriever import VectorRetriever, SearchResult


@dataclass
class RAGResponse:
    """Complete output of one RAG query."""
    question:          str
    answer:            str
    sources:           List[SearchResult]
    is_supported:      bool
    response_time_sec: float
    model:             str
    prompt_version:    str
    metadata:          dict = field(default_factory=dict)

    def formatted_answer(self) -> str:
        """Display-ready answer with citations."""
        if not self.is_supported:
            return (
                "⚠️ **Insufficient Evidence**\n\n"
                "I couldn't find enough information in the GitHub documentation "
                "to answer that reliably. Try checking https://docs.github.com "
                "or rephrasing your question.\n\n"
                f"*Question: {self.question}*"
            )

        # Deduplicate and format citations (sorted highest score first, no score shown)
        seen  = set()
        cites = []
        for src in self.sources:
            url = src.source_file
            if url in seen:
                continue
            seen.add(url)
            label = _url_to_label(url)
            cites.append(f"- [{label}]({url})")

        linked_answer = _linkify_answer(self.answer)

        return (
            f"{linked_answer}\n\n"
            f"---\n"
            f"**Sources:**\n" + "\n".join(cites) + "\n"
            f"\n*Response time: {self.response_time_sec:.1f}s*"
        )


@dataclass
class StreamingSetup:
    """
    Holds retrieval results and a lazy token generator for a streaming RAG response.
    Retrieval is done synchronously; the LLM call starts only when token_gen is consumed.
    """
    sources:        List[SearchResult]
    token_gen:      Iterator[str]   # yields str chunks as LLM generates them
    start_time:     float
    model:          str
    prompt_version: str
    metadata:       dict = field(default_factory=dict)


_URL_RE = re.compile(r'https?://[^\s\)\]>]+')


def _url_to_label(url: str) -> str:
    """Derive a human-readable label from a URL path segment."""
    try:
        path = urlparse(url).path.rstrip("/")
        segment = path.split("/")[-1] if path else ""
        if segment:
            return segment.replace("-", " ").replace("_", " ").title()
    except Exception:
        pass
    return "GitHub Docs"


def _linkify_answer(text: str) -> str:
    """Replace bare URLs in answer text with descriptive markdown hyperlinks."""
    def _replace(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:!?)")
        return f"[{_url_to_label(url)}]({url})"
    return _URL_RE.sub(_replace, text)


def build_context(chunks: List[SearchResult]) -> str:
    """Format retrieved chunks into a labeled context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        label = (
            f"[CHUNK {i} | "
            f"{chunk.source_file} | "
            f"{chunk.citation()}]"
        )
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n".join(parts)


class RAGChain:
    """
    Full RAG pipeline using the course Qwen endpoint.

    Usage:
        chain = RAGChain(settings)
        response = chain.ask("How do I create a private repo?", session_id="abc")
        print(response.formatted_answer())
    """

    def __init__(self, settings):
        self.settings       = settings
        self.prompts        = load_prompts()
        self.prompt_version = self.prompts.get("version", "unknown")

        # OpenAI-compatible client → course Qwen endpoint
        self.client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )

        # Retriever loads embedding model + connects to ChromaDB
        self.retriever = VectorRetriever(settings)

        logger.success(
            f"RAGChain ready | model={settings.llm_model} | "
            f"collection={settings.chroma_collection_name}"
        )

    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: str = "",
    ) -> List[dict]:
        """Build messages list for the Qwen chat completions API."""
        rag_prompt   = self.prompts["rag_prompt"]
        system_text  = rag_prompt["system"]
        user_text    = rag_prompt["user_template"].format(
            context=context,
            chat_history=chat_history,
            question=question,
        )
        return [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ]

    def _call_llm(self, messages: List[dict]) -> str:
        """Call the Qwen endpoint and return generated text."""
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def _resolve_search_query(
        self,
        question: str,
        session_state,
        session_id: Optional[str],
    ) -> str:
        """
        For follow-up questions ("tell me more", "what were the steps?"), the
        raw question has no semantic content for the vector search. Enrich it
        by prepending the last substantive user question from history so the
        retriever finds the right chunks.
        """
        if not session_state or not session_id:
            return question

        # Only enrich short / vague questions (follow-ups are typically < 9 words)
        if len(question.strip().split()) > 8:
            return question

        history = session_state.get_history(session_id)

        # Walk backwards to find the last user message that looks substantive
        for msg in reversed(history):
            if msg["role"] == "user" and len(msg["content"].split()) > 5:
                prior = msg["content"]
                enriched = f"{prior} {question}"
                logger.debug(f"Follow-up enriched: '{enriched[:80]}'")
                return enriched

        return question

    def _is_supported(self, answer: str) -> bool:
        """Check if the answer is grounded (not INSUFFICIENT_EVIDENCE)."""
        if not answer:
            return False
        if "INSUFFICIENT_EVIDENCE" in answer.upper():
            return False
        if len(answer.strip()) < 20:
            return False
        return True

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        session_state=None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Main entry point — question in, RAGResponse out.

        Args:
            question:      User's question
            session_id:    For conversation memory
            session_state: SessionState instance (for history)
            top_k:         Override number of chunks to use
        """
        start = time.time()
        k     = top_k or self.settings.top_k_rerank

        logger.info(f"RAG question: '{question[:80]}'")

        # ── Resolve search query (enrich follow-ups with prior context) ────
        search_query = self._resolve_search_query(question, session_state, session_id)

        # ── Retrieve ───────────────────────────────────────────────────────
        chunks = self.retriever.search(search_query, top_k=self.settings.top_k_retrieval)

        if not chunks:
            logger.warning("No chunks retrieved")
            return RAGResponse(
                question=question,
                answer="INSUFFICIENT_EVIDENCE",
                sources=[],
                is_supported=False,
                response_time_sec=time.time() - start,
                model=self.settings.llm_model,
                prompt_version=self.prompt_version,
            )

        top_chunks = chunks[:k]

        # ── Build context ──────────────────────────────────────────────────
        context = build_context(top_chunks)

        # ── Get conversation history ───────────────────────────────────────
        chat_history = ""
        if session_state and session_id:
            chat_history = session_state.format_history_for_prompt(session_id)

        # ── Build messages ─────────────────────────────────────────────────
        messages = self._build_messages(question, context, chat_history)

        # ── Call Qwen ──────────────────────────────────────────────────────
        logger.info(f"Calling {self.settings.llm_model}...")
        answer    = self._call_llm(messages)
        supported = self._is_supported(answer)

        if not supported:
            log_path = Path(self.settings.log_dir) / "declined_queries.log"
            log_path.parent.mkdir(exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "question": question,
                    "answer_preview": answer[:100],
                }) + "\n")

        elapsed = time.time() - start
        logger.success(f"Answer ready | supported={supported} | {elapsed:.1f}s")

        return RAGResponse(
            question=question,
            answer=answer,
            sources=top_chunks,
            is_supported=supported,
            response_time_sec=elapsed,
            model=self.settings.llm_model,
            prompt_version=self.prompt_version,
            metadata={
                "chunks_retrieved": len(chunks),
                "chunks_used":      len(top_chunks),
                "top_similarity":   chunks[0].similarity_score if chunks else 0,
            },
        )

    def ask_streaming(
        self,
        question: str,
        session_id: Optional[str] = None,
        session_state=None,
        top_k: Optional[int] = None,
    ) -> Optional[StreamingSetup]:
        """
        Streaming variant of ask(). Retrieval and history fetch are done
        synchronously; the LLM call is deferred into a lazy generator so that
        st.write_stream() can consume tokens as they arrive.

        Returns None if retrieval finds no chunks — caller should fall back to ask().
        The ask() method is kept intact for the evaluation suite and non-streaming uses.
        """
        start = time.time()
        k     = top_k or self.settings.top_k_rerank

        logger.info(f"RAG streaming: '{question[:80]}'")

        # ── Resolve search query (enriches follow-ups) ─────────────────────
        search_query = self._resolve_search_query(question, session_state, session_id)

        # ── Retrieve ───────────────────────────────────────────────────────
        chunks = self.retriever.search(search_query, top_k=self.settings.top_k_retrieval)

        if not chunks:
            logger.warning("No chunks retrieved — streaming unavailable")
            return None

        top_chunks = chunks[:k]

        # ── Build context ──────────────────────────────────────────────────
        context = build_context(top_chunks)

        # ── Get conversation history ───────────────────────────────────────
        chat_history = ""
        if session_state and session_id:
            chat_history = session_state.format_history_for_prompt(session_id)

        # ── Build messages ─────────────────────────────────────────────────
        messages = self._build_messages(question, context, chat_history)

        # ── Simulated word-by-word streaming ──────────────────────────────
        # The course Qwen endpoint buffers the full response server-side before
        # sending, so stream=True still delivers one large chunk — all text
        # appears at once. We get the full answer via _call_llm() and then
        # yield it word by word with a small delay so st.write_stream() renders
        # it progressively, exactly like ChatGPT-style typewriter output.
        def _token_gen() -> Generator[str, None, None]:
            answer = self._call_llm(messages)
            words  = answer.split(" ")
            for i, word in enumerate(words):
                # Preserve trailing space on every word except the last
                yield word if i == len(words) - 1 else word + " "
                time.sleep(0.02)   # ~50 words/sec — natural reading pace

        return StreamingSetup(
            sources=top_chunks,
            token_gen=_token_gen(),
            start_time=start,
            model=self.settings.llm_model,
            prompt_version=self.prompt_version,
            metadata={
                "chunks_retrieved": len(chunks),
                "chunks_used":      len(top_chunks),
                "top_similarity":   chunks[0].similarity_score if chunks else 0.0,
            },
        )

    def suggest_followups(self, question: str, answer: str) -> List[str]:
        """
        Generate 3 follow-up question suggestions based on the Q&A just shown.
        Returns a list of up to 3 strings, or [] on any failure (feature degrades gracefully).
        """
        followup_cfg = self.prompts.get("followup_prompt")
        if not followup_cfg:
            return []

        user_text = followup_cfg["user_template"].format(
            question=question,
            answer=answer[:800],   # cap context to avoid token overflow
        )
        messages = [
            {"role": "system", "content": followup_cfg["system"]},
            {"role": "user",   "content": user_text},
        ]

        try:
            raw = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            ).choices[0].message.content.strip()

            questions = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip leading numbering (1. 2. 1) etc.) and bullets (- • →)
                line = re.sub(r'^[\d]+[.)]\s*', '', line)
                line = re.sub(r'^[-•→]\s*', '', line)
                # Strip surrounding quotes the LLM might add
                line = line.strip('"\'')
                if len(line) > 8:
                    questions.append(line)
                if len(questions) == 3:
                    break

            logger.debug(f"Follow-up suggestions: {questions}")
            return questions

        except Exception as exc:
            logger.warning(f"suggest_followups failed: {exc}")
            return []
