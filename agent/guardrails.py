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

from agent.intent_router import IntentResult
from loguru import logger


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


def handle_insufficient_evidence(prompts: dict) -> str:
    """Return the standard insufficient evidence message."""
    guardrail = prompts.get("guardrail_prompt", {})
    return guardrail.get(
        "insufficient_evidence",
        "I couldn't find enough information in the GitHub documentation to answer that. "
        "Try checking https://docs.github.com directly."
    )
