"""
agent/guardrails.py
────────────────────────────────────────────────────────────────
Guardrails for the GitHub Documentation Assistant.

CHECKS (in order):
    1. Prompt injection detection
    2. Out-of-scope detection (for things intent router misses)
    3. Insufficient evidence handler

All guardrail responses are loaded from prompts.yaml so they
can be tuned without touching code.
────────────────────────────────────────────────────────────────
"""

import re

from agent.intent_router import IntentResult
from loguru import logger


_BILLING_CONTEXT_RE = re.compile(
    r"\b(billing|payment|payments|refund|invoice|receipt|charge|charged|pricing|plan|subscription|cancel)\b",
    re.IGNORECASE,
)


def get_guardrail_response(intent_result: IntentResult, prompts: dict) -> str | None:
    """
    Check if a guardrail should fire.
    Returns a response string if guardrail fires, else None.

    Call this BEFORE routing to RAG or actions.
    """
    guardrail = prompts.get("guardrail_prompt", {})

    if intent_result.intent == "prompt_injection":
        logger.warning(f"Prompt injection blocked | confidence={intent_result.confidence}")
        return guardrail.get(
            "injection_detected",
            "I cannot follow those instructions. Please ask a GitHub-related question."
        )

    if intent_result.intent == "out_of_scope":
        logger.info("Out-of-scope query rejected")
        return guardrail.get(
            "out_of_scope",
            "I can only help with GitHub-related questions. Is there something about GitHub I can help with?"
        )

    return None


def handle_insufficient_evidence(prompts: dict, user_message: str = "") -> str:
    """Return a context-aware insufficient evidence message."""
    guardrail = prompts.get("guardrail_prompt", {})

    # Keep billing-specific escalation language only for billing/payment topics.
    if _BILLING_CONTEXT_RE.search(user_message or ""):
        return guardrail.get(
            "insufficient_evidence",
            "I couldn't find enough information in the GitHub documentation to answer that. "
            "Try checking https://docs.github.com directly."
        )

    return (
        "I couldn't find enough information in the available GitHub documentation "
        "to answer that reliably.\n\n"
        "Try rephrasing your question, or ask for a specific GitHub feature/page "
        "(for example: repositories, Actions, 2FA, organizations, or security settings)."
    )
