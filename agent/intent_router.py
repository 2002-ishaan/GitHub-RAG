"""
agent/intent_router.py
────────────────────────────────────────────────────────────────
Classifies every user message into one of 6 intents before
any action is taken.

INTENTS:
    rag_query        → answer from GitHub Docs knowledge base
    create_ticket    → multi-turn ticket creation action
    check_ticket     → look up existing ticket by ID
    check_billing    → look up mock account plan
    out_of_scope     → politely reject
    prompt_injection → block and warn

HOW IT WORKS:
    1. Fast regex pre-check (no LLM needed for obvious cases)
    2. LLM classification for anything ambiguous
    3. Returns IntentResult with intent + confidence
────────────────────────────────────────────────────────────────
"""

import re
import json
from dataclasses import dataclass
from openai import OpenAI
from loguru import logger


@dataclass
class IntentResult:
    intent:     str    # one of the 6 intents above
    confidence: float  # 0.0 to 1.0
    raw:        str    # raw LLM response (for debugging)


# ── Regex patterns for fast pre-classification ─────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+your\s+(system\s+)?prompt",
    r"pretend\s+you\s+are",
    r"you\s+are\s+now\s+a",
    r"forget\s+everything",
    r"new\s+persona",
    r"jailbreak",
    r"dan\s+mode",
    r"override\s+your",
]

TICKET_PATTERNS = [
    r"\bcreate\s+(a\s+)?(support\s+)?ticket\b",
    r"\bopen\s+(a\s+)?ticket\b",
    r"\bsubmit\s+(a\s+)?request\b",
    r"\bi\s+need\s+help\s+with\s+my\s+account\b",
    r"\bfile\s+(a\s+)?complaint\b",
]

CHECK_TICKET_PATTERNS = [
    r"\b(check|view|see|status\s+of)\s+(ticket|tkt)[- ]?\w*\b",
    r"\btkt[-\s]?\d+\b",
    r"\bmy\s+ticket\s+status\b",
]

BILLING_PATTERNS = [
    r"\b(check|view|see)\s+(my\s+)?(billing|plan|subscription)\b",
    r"\bwhat\s+plan\s+(am\s+i|is\s+\w+)\s+(on|using)\b",
    r"\bmy\s+account\s+plan\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(weather|forecast|temperature|rain)\b",
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|song|story|essay)\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b",
    r"\b(recipe|cook|food|restaurant)\b",
    r"\b(sports|football|basketball|soccer|nba|nfl)\b",
    r"\b(celebrity|actor|movie|netflix|spotify)\b",
]

CLOSE_TICKET_PATTERNS = [
    r"\bclose\s+(all\s+)?(my\s+)?(active\s+)?tickets?\b",
    r"\bmark\s+(all\s+)?tickets?\s+(as\s+)?closed\b",
    r"\bclose\s+ticket\b",
    r"\bclose\s+active\b",
    r"\bshut\s+(all\s+)?tickets?\b",
]


def _regex_check(message: str) -> str | None:
    """Fast regex pre-check. Returns intent string or None."""
    msg_lower = message.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, msg_lower):
            return "prompt_injection"
    
    for pattern in CLOSE_TICKET_PATTERNS:
        if re.search(pattern, msg_lower):
            return "close_tickets"

    for pattern in CHECK_TICKET_PATTERNS:
        if re.search(pattern, msg_lower):
            return "check_ticket"

    for pattern in TICKET_PATTERNS:
        if re.search(pattern, msg_lower):
            return "create_ticket"

    for pattern in BILLING_PATTERNS:
        if re.search(pattern, msg_lower):
            return "check_billing"

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, msg_lower):
            return "out_of_scope"

    return None


class IntentRouter:
    """
    Classifies user messages to route them to the right handler.

    Usage:
        router = IntentRouter(settings, prompts)
        result = router.classify("How do I create a private repo?")
        # result.intent == "rag_query"
    """

    def __init__(self, settings, prompts: dict):
        self.settings = settings
        self.prompts  = prompts

        # OpenAI-compatible client pointed at the course Qwen endpoint
        self.client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )

        logger.info("IntentRouter ready")

    def classify(self, message: str) -> IntentResult:
        """
        Classify a user message into one of the 6 intents.
        Tries regex first (fast), then falls back to LLM.
        """
        if not message or not message.strip():
            return IntentResult(intent="out_of_scope", confidence=1.0, raw="empty message")

        # ── Fast regex check ───────────────────────────────────────────────
        regex_intent = _regex_check(message)
        if regex_intent:
            logger.info(f"Intent (regex): {regex_intent} | '{message[:60]}'")
            return IntentResult(
                intent=regex_intent,
                confidence=0.95,
                raw=f"regex match"
            )

        # ── LLM classification ─────────────────────────────────────────────
        intent_config = self.prompts["intent_prompt"]
        user_content  = intent_config["user_template"].format(message=message)

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": intent_config["system"]},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0.0,
                max_tokens=50,
            )

            raw = response.choices[0].message.content.strip()

            # Parse JSON response
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(clean)

            intent     = parsed.get("intent", "rag_query")
            confidence = float(parsed.get("confidence", 0.8))

            # Validate intent is one of our known types
            valid_intents = {
                "rag_query", "create_ticket", "check_ticket",
                "check_billing", "close_tickets", "out_of_scope", "prompt_injection"
            }
            if intent not in valid_intents:
                intent = "rag_query"

            logger.info(
                f"Intent (LLM): {intent} ({confidence:.2f}) | '{message[:60]}'"
            )
            return IntentResult(intent=intent, confidence=confidence, raw=raw)

        except Exception as e:
            # If LLM classification fails, default to RAG query
            logger.warning(f"Intent classification failed: {e} — defaulting to rag_query")
            return IntentResult(intent="rag_query", confidence=0.5, raw=str(e))
