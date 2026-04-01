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

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

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

        # Deduplicate citations
        seen  = set()
        cites = []
        for src in self.sources:
            cite = src.citation()
            if cite not in seen:
                cites.append(f"  • {cite}  (score: {src.similarity_score:.3f})")
                seen.add(cite)

        return (
            f"{self.answer}\n\n"
            f"---\n"
            f"**Sources:**\n" + "\n".join(cites) + "\n"
            f"\n*Response time: {self.response_time_sec:.1f}s*"
        )


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

        # ── Retrieve ───────────────────────────────────────────────────────
        chunks = self.retriever.search(question, top_k=self.settings.top_k_retrieval)

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
            with open(log_path, "a") as f:
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
