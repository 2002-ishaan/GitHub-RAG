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

        normalized_answer = _normalize_inline_citations(self.answer, self.sources)
        linked_answer = _linkify_answer(normalized_answer)

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
_INLINE_SOURCE_CITE_RE = re.compile(r'\[Source:\s*([^,\]]+),\s*(https?://[^\]\s]+)\]')
_PLACEHOLDER_CITE_RE = re.compile(r'\[Source:\s*<title>,\s*<url>\]')
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ALIGNMENT_STOPWORDS = {
    "how", "what", "when", "where", "why", "who", "which", "is", "are", "am",
    "to", "for", "of", "on", "in", "with", "my", "your", "the", "a", "an",
    "do", "does", "can", "i", "we", "it", "that", "this", "and", "or",
}

_FALLBACK_BLOCKLIST = {
    "salary", "salaries", "employee", "employees", "internal", "confidential",
    "payroll", "compensation", "benefits", "password", "private",
}

_GITHUB_DOC_HINTS = {
    "repository", "repositories", "organization", "organizations", "team", "teams",
    "pull", "request", "issue", "actions", "workflow", "billing", "plan",
    "authentication", "2fa", "security", "dependabot", "codeowners", "branch",
    "token", "ssh", "release", "permissions", "projects",
    "secret", "secrets", "key", "keys", "credential", "credentials",
}


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


def _normalize_inline_citations(text: str, sources: List[SearchResult]) -> str:
    """
    Normalize inline citation variants into markdown links.
    Also replaces placeholder citations with concrete links from retrieved sources.
    """
    if not text:
        return text

    normalized = _INLINE_SOURCE_CITE_RE.sub(
        lambda m: f"[Source: {m.group(1).strip()}]({m.group(2).strip()})",
        text,
    )

    unique_urls = []
    seen = set()
    for src in sources:
        if src.source_file in seen:
            continue
        seen.add(src.source_file)
        unique_urls.append(src.source_file)

    idx = {"i": 0}

    def _replace_placeholder(_: re.Match) -> str:
        if unique_urls:
            url = unique_urls[min(idx["i"], len(unique_urls) - 1)]
            idx["i"] += 1
            return f"[Source: {_url_to_label(url)}]({url})"
        return "(see Sources below)"

    normalized = _PLACEHOLDER_CITE_RE.sub(_replace_placeholder, normalized)
    return normalized


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

    def _is_summary_request(self, question: str) -> bool:
        q = question.lower().strip().rstrip("?!.,")
        markers = [
            "summarize", "summarise", "summary", "recap",
            "tell me more", "more about that", "more about this",
            "what did you say", "what did you mention",
            "what were the steps", "what are the steps",
            "expand on", "elaborate", "go on", "what else",
            "what else should i know", "what else should",
            "more details", "more detail",
            "first point", "second point", "third point", "last point",
            "can you explain more", "explain more",
            "continue", "and then", "what was that",
            "anything else", "what more", "tell me anything",
        ]
        # Also match exact short phrases that are pure follow-ups
        exact = {"go on", "continue", "and then", "what else", "tell me more",
                 "what else should i know", "anything else"}
        if q in exact:
            return True
        return any(m in q for m in markers)

    def _filter_relevant_chunks(self, question: str, chunks: List[SearchResult]) -> List[SearchResult]:
        """
        Remove low-similarity or clearly off-topic chunks.
        Keeps behavior conservative to reduce irrelevant citations.
        """
        q = question.lower()
        token_count = len(question.strip().split())
        is_repo_visibility = (
            "private repo" in q
            or "repository visibility" in q
            or ("repository" in q and "private" in q)
        )
        # Short/noisy questions are naturally less semantically stable.
        min_similarity = 0.16 if token_count <= 5 else 0.20

        filtered: List[SearchResult] = []
        for idx, c in enumerate(chunks):
            # Always retain top-1 chunk unless it's extremely weak.
            if idx == 0 and c.similarity_score >= 0.10:
                filtered.append(c)
                continue

            if c.similarity_score < min_similarity:   # BGE/bge-base-en-v1.5 scores run lower than MiniLM
                continue
            txt = c.text.lower()
            if is_repo_visibility and "github actions" in txt and "repository" not in txt and "visibility" not in txt:
                continue
            filtered.append(c)
        return filtered

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
        content = response.choices[0].message.content
        return (content or "").strip()

    def _resolve_search_query(
        self,
        question: str,
        session_state,
        session_id: Optional[str],
    ) -> str:
        """
        For follow-up questions ("tell me more", "what were the steps?"), the
        raw question has no semantic content for the vector search. Replace or
        enrich it with the last substantive user question from history so the
        retriever finds the right chunks.

        - Summary/follow-up requests ("tell me more", "go on", "elaborate"):
          search using the PREVIOUS question directly — the current phrase has
          zero signal for the vector store.
        - Short vague questions (< 9 words, not a summary request):
          enrich by prepending the previous question.
        """
        if not session_state or not session_id:
            return question

        is_followup = self._is_summary_request(question)
        # Only treat truly vague short phrases (≤ 4 words) as needing enrichment.
        # Anything longer — even "How do I enable 2FA?" (6 words) — has enough
        # semantic content to search on its own and must NOT be polluted with
        # context from a previous unrelated question.
        is_short    = len(question.strip().split()) <= 4

        if not is_followup and not is_short:
            return question

        history = session_state.get_history(session_id)

        # Walk backwards to find the last substantive user message that is not
        # the current follow-up itself.
        for msg in reversed(history):
            if msg["role"] != "user":
                continue
            content = msg["content"].strip()
            if content.lower() == question.lower():
                continue           # skip if it's the current question echoed back
            if len(content.split()) > 5:
                if is_followup:
                    # "Tell me more" → search with the previous question verbatim
                    logger.debug(f"Follow-up (summary): reusing prior query '{content[:60]}'")
                    return content
                else:
                    # Truly vague short phrase → enrich with prior context
                    enriched = f"{content} {question}"
                    logger.debug(f"Follow-up (enriched): '{enriched[:80]}'")
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

    def _build_extractive_fallback(self, question: str, chunks: List[SearchResult]) -> str:
        """
        Build a safe extractive answer directly from retrieved chunks.

        This is used when the model returns empty/insufficient output even
        though retrieval is highly relevant.
        """
        if not chunks:
            return "INSUFFICIENT_EVIDENCE"

        top = chunks[0].text.strip()
        if not top:
            return "INSUFFICIENT_EVIDENCE"

        compact = re.sub(r"\s+", " ", top)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]

        selected = sentences[:2] if len(sentences) >= 2 else [compact[:420]]
        bullet_lines = [f"- {line}" for line in selected if line]

        return (
            f"I found relevant guidance in GitHub documentation for: {question}\n\n"
            + "\n".join(bullet_lines)
            + "\n\n"
            + "If you want, I can provide a shorter step-by-step version."
        )

    def _has_lexical_alignment(self, question: str, top_chunk_text: str) -> bool:
        """Check whether query and top chunk share enough content words."""
        q_tokens = {
            t for t in _TOKEN_RE.findall((question or "").lower())
            if t not in _ALIGNMENT_STOPWORDS and len(t) > 2
        }
        if not q_tokens:
            return False

        d_tokens = set(_TOKEN_RE.findall((top_chunk_text or "").lower()))
        overlap_ratio = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        return overlap_ratio >= 0.20

    def _is_safe_for_extractive_fallback(self, question: str) -> bool:
        """Allow fallback only for likely GitHub documentation queries."""
        q_tokens = set(_TOKEN_RE.findall((question or "").lower()))
        if not q_tokens:
            return False

        # Never use extractive fallback for likely internal/sensitive requests.
        if q_tokens & _FALLBACK_BLOCKLIST:
            return False

        # Require at least one GitHub-doc keyword beyond generic "github".
        return bool(q_tokens & _GITHUB_DOC_HINTS)

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
        rel = self._filter_relevant_chunks(question, chunks)
        if rel:
            chunks = rel

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
            max_turns = 12 if self._is_summary_request(question) else 6
            chat_history = session_state.format_history_for_prompt(session_id, max_turns=max_turns)

        # ── Build messages ─────────────────────────────────────────────────
        messages = self._build_messages(question, context, chat_history)

        # ── Call Qwen ──────────────────────────────────────────────────────
        logger.info(f"Calling {self.settings.llm_model}...")
        answer    = self._call_llm(messages)
        supported = self._is_supported(answer)

        if (
            not supported
            and chunks
            and chunks[0].similarity_score >= 0.75
            and self._is_safe_for_extractive_fallback(question)
            and self._has_lexical_alignment(question, top_chunks[0].text)
        ):
            logger.warning(
                "LLM returned unsupported output despite high retrieval score; "
                "using extractive fallback."
            )
            answer = self._build_extractive_fallback(question, top_chunks)
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
        rel = self._filter_relevant_chunks(question, chunks)
        if rel:
            chunks = rel

        if not chunks:
            logger.warning("No chunks retrieved — streaming unavailable")
            return None

        top_chunks = chunks[:k]

        # ── Build context ──────────────────────────────────────────────────
        context = build_context(top_chunks)

        # ── Get conversation history ───────────────────────────────────────
        chat_history = ""
        if session_state and session_id:
            max_turns = 12 if self._is_summary_request(question) else 6
            chat_history = session_state.format_history_for_prompt(session_id, max_turns=max_turns)

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
            supported = self._is_supported(answer)
            if (
                not supported
                and top_chunks
                and top_chunks[0].similarity_score >= 0.75
                and self._is_safe_for_extractive_fallback(question)
                and self._has_lexical_alignment(question, top_chunks[0].text)
            ):
                logger.warning(
                    "Streaming LLM output unsupported despite high retrieval score; "
                    "using extractive fallback."
                )
                answer = self._build_extractive_fallback(question, top_chunks)

            delay_sec = max(0.0, float(getattr(self.settings, "streaming_word_delay_sec", 0.0)))
            if delay_sec <= 0:
                yield answer
                return

            words  = answer.split(" ")
            for i, word in enumerate(words):
                # Preserve trailing space on every word except the last
                yield word if i == len(words) - 1 else word + " "
                time.sleep(delay_sec)

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

    def generate_new_info(self, previous_answer: str, current_answer: str) -> str:
        """
        Compare two RAG answers and return genuinely new information points.

        Returns bullet-point text, or the sentinel "NOTHING_NEW" when nothing
        new was added (or if comparison fails). Callers should show output only
        when the return value is NOT "NOTHING_NEW".
        """
        messages = [
            {
                "role": "system",
                "content": "You identify new information. Be extremely brief.",
            },
            {
                "role": "user",
                "content": (
                    f"Previous answer: {previous_answer}\n\n"
                    f"New answer: {current_answer}\n\n"
                    "What facts appear in the new answer that were NOT mentioned "
                    "in the previous answer? List only genuinely new points in "
                    "1-2 bullet points. If nothing is new or there is no previous "
                    "answer, respond with exactly: NOTHING_NEW"
                ),
            },
        ]
        try:
            raw = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=120,
            ).choices[0].message.content.strip()
            logger.debug(f"generate_new_info result: {raw[:80]}")
            return raw
        except Exception as exc:
            logger.warning(f"generate_new_info failed: {exc}")
            return "NOTHING_NEW"

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

        def _default_followups(seed_question: str) -> List[str]:
            q = (seed_question or "").lower()

            if any(k in q for k in ["2fa", "two-factor", "authentication", "token", "ssh", "security"]):
                return [
                    "How can I verify 2FA is enabled correctly?",
                    "What recovery options should I set up for account access?",
                    "Where can I manage authentication settings on GitHub?",
                ]

            if any(k in q for k in ["billing", "plan", "subscription", "refund", "invoice", "payment"]):
                return [
                    "How can I review my current plan details and limits?",
                    "Where can I update billing contacts and payment methods?",
                    "What changes when upgrading or downgrading plans?",
                ]

            if any(k in q for k in ["ticket", "support", "issue", "tkt-"]):
                return [
                    "How can I check the status of my support ticket?",
                    "What details should I include to speed up ticket resolution?",
                    "How do I close a resolved ticket?",
                ]

            if any(k in q for k in ["repository", "repo", "branch", "pull request", "pr", "issue", "actions"]):
                return [
                    "What are common mistakes when doing this on GitHub?",
                    "How can I verify this was configured correctly?",
                    "Where can I manage this setting in GitHub?",
                ]

            return [
                "What is the recommended workflow for this on GitHub?",
                "How can I validate this setup after making changes?",
                "What related GitHub setting should I review next?",
            ]

        try:
            raw = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            ).choices[0].message.content

            raw_text = (raw or "").strip()
            if not raw_text:
                logger.warning("suggest_followups got empty LLM output; using defaults")
                return _default_followups(question)

            questions = []
            for line in raw_text.splitlines():
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

            # Some models return one paragraph instead of line-separated questions.
            if len(questions) < 3:
                sentence_candidates = re.split(r'\?\s+|\n|;\s+', raw_text)
                for candidate in sentence_candidates:
                    c = candidate.strip().strip('"\'')
                    c = re.sub(r'^[\d]+[.)]\s*', '', c)
                    c = re.sub(r'^[-•→]\s*', '', c)
                    if not c:
                        continue
                    if not c.endswith("?"):
                        c = c + "?"
                    if len(c) > 8 and c not in questions:
                        questions.append(c)
                    if len(questions) == 3:
                        break

            if not questions:
                return _default_followups(question)

            logger.debug(f"Follow-up suggestions: {questions}")
            return questions[:3]

        except Exception as exc:
            logger.warning(f"suggest_followups failed: {exc}")
            return _default_followups(question)
